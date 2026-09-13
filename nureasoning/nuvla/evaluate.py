"""
VLA Model Evaluation Script.

Two evaluation modes:
  1. Reasoning generation — feed multi-view images, generate text via the VLM.
  2. Planning benchmark  — evaluate trajectory prediction (ADE/FDE/Heading)
     over the full test set; reasoning text is NOT generated, only VLM features
     are extracted and fed to the action expert.

Add --visualize to save per-sample figures (input images, prompt, reasoning /
trajectory) alongside the normal evaluation output.

Usage:
    # Planning benchmark with visualizations
    python -m nureasoning.nuvla.evaluate \
        --checkpoint_dir ./nureasoning_vla_workspace/final \
        --mode planning --visualize

    # Reasoning generation with visualizations
    python -m nureasoning.nuvla.evaluate \
        --checkpoint_dir ./nureasoning_vla_workspace/final \
        --mode reasoning --visualize \
        --num_reasoning_samples 5

    # Both modes, no visualizations
    python -m nureasoning.nuvla.evaluate \
        --checkpoint_dir ./nureasoning_vla_workspace/final \
        --mode both
"""

import argparse
import json
import logging
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES, VLMBackbone, VLMBackboneConfig
from nureasoning.nuvla.models.action_expert import (
    FlowMatchingDiTActionExpert,
    ActionExpertConfig,
    compute_trajectory_metrics,
)
from nureasoning.nuvla.models.data_loader import (
    DEFAULT_FRAME_RATE_HZ,
    VLADataConfig,
    NuReasoningVLADataset,
    vla_collate_fn,
    waypoint_dt_s,
)
from nureasoning.nuvla.trajectory_provider import (
    PLANNING_PROMPT_AUTO,
    add_planning_prompt_arguments,
    build_planning_user_prompt,
    load_vla_training_config,
    resolve_planning_prompt_kind,
)

logger = logging.getLogger(__name__)
DEFAULT_KEYFRAME_TIME_S = 10.0


class VLATester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_cfg = load_vla_training_config(args.checkpoint_dir)
        logger.info("Loaded training config from workspace")
        planning_prompt = getattr(args, "planning_prompt", PLANNING_PROMPT_AUTO)
        logger.info(
            "Planning prompt: %s -> %s (checkpoint reasoning_mode=%s, "
            "format override=%s)",
            planning_prompt,
            resolve_planning_prompt_kind(
                planning_prompt,
                str(self.train_cfg.get("reasoning_mode") or "structured"),
            ),
            self.train_cfg.get("reasoning_mode", "structured"),
            getattr(args, "planning_reasoning_format", None),
        )

        self._build_models()
        self._load_checkpoint()
        self._build_test_data()

    def _build_models(self):
        cfg = self.train_cfg

        vlm_config = VLMBackboneConfig(
            model_name_or_path=cfg["vlm_model_path"],
            freeze_vision_encoder=True,
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.0),
            current_resolution=(cfg["current_res_w"], cfg["current_res_h"]),
            history_resolution=(cfg["history_res_w"], cfg["history_res_h"]),
            reasoning_format=cfg.get("reasoning_format", "spatial_driving_counterfactual"),
        )

        self.vlm = VLMBackbone(vlm_config).to(self.device)
        self.vlm.eval()

        actual_vlm_feature_dim = self.vlm.get_feature_dim()

        action_config = ActionExpertConfig(
            vlm_feature_dim=actual_vlm_feature_dim,
            ego_state_dim=4,
            max_history_traj_points=cfg.get("max_history_traj_points", 6),
            history_traj_dim=3,
            num_waypoints=cfg.get("num_trajectory_points", 10),
            trajectory_dim=3,
            hidden_dim=cfg.get("action_hidden_dim", 512),
            num_dit_layers=cfg.get("num_dit_layers", cfg.get("num_dit_blocks", 12)),
            num_heads=cfg.get("num_dit_heads", 8),
            dropout=0.0,
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            interleave_self_attention=cfg.get("interleave_self_attention", True),
            num_inference_steps=cfg.get("num_inference_steps", 5),
            num_timestep_buckets=cfg.get("num_timestep_buckets", 1000),
            noise_beta_alpha=cfg.get("noise_beta_alpha", 1.5),
            noise_beta_beta=cfg.get("noise_beta_beta", 2.5),
        )
        self.action_expert = FlowMatchingDiTActionExpert(action_config).to(self.device)
        self.action_expert.eval()

        logger.info("VLM feature dim: %d", actual_vlm_feature_dim)
        logger.info(
            "Action expert params: %s",
            f"{sum(p.numel() for p in self.action_expert.parameters()):,}",
        )

    def _load_checkpoint(self):
        ckpt_dir = self.args.checkpoint_dir

        vlm_adapter_dir = os.path.join(ckpt_dir, "vlm_adapter")
        if os.path.isdir(vlm_adapter_dir):
            self.vlm.load_adapter(vlm_adapter_dir, device=self.device)
            logger.info("Loaded VLM adapter from %s", vlm_adapter_dir)
            self._verify_lora_load(vlm_adapter_dir)
        else:
            logger.warning("No vlm_adapter directory found in %s", ckpt_dir)

        action_path = os.path.join(ckpt_dir, "action_expert.pt")
        if os.path.isfile(action_path):
            state = torch.load(action_path, map_location=self.device)
            self.action_expert.load_state_dict(state)
            logger.info("Loaded action expert from %s", action_path)
        else:
            logger.warning("No action_expert.pt found in %s", ckpt_dir)

    def _verify_lora_load(self, adapter_dir: str) -> None:
        """Compare on-device LoRA params against the saved safetensors file.

        Hard-fails if fewer than 95% of saved tensors are byte-identical (up to
        dtype cast) with the live parameters on the model, to guarantee that
        inference is actually running with the fine-tuned adapter.
        """
        sft_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(sft_path):
            logger.warning("LoRA audit skipped: %s not found", sft_path)
            return

        from safetensors.torch import load_file
        saved = load_file(sft_path)

        live = {n: p for n, p in self.vlm.model.named_parameters() if "lora_" in n}

        def _norm(key: str) -> str:
            # saved keys look like "base_model.model.model.<...>.lora_A.weight"
            # live keys look like        "base_model.model.<...>.lora_A.default.weight"
            k = key
            k = k.replace(".default.weight", ".weight")
            if k.startswith("base_model.model.model."):
                k = "base_model.model." + k[len("base_model.model.model."):]
            return k

        live_norm = {_norm(n): p for n, p in live.items()}

        matches = 0
        mismatches = 0
        not_found = 0
        total = 0
        zero_in_file = 0
        for k, t_file in saved.items():
            total += 1
            k_norm = _norm(k)
            p = live_norm.get(k_norm)
            if p is None:
                not_found += 1
                continue
            if t_file.float().abs().max().item() < 1e-12:
                zero_in_file += 1
            t_live = p.detach().to("cpu", dtype=t_file.dtype)
            if t_live.shape == t_file.shape and torch.allclose(
                t_live, t_file, atol=1e-6, rtol=1e-5,
            ):
                matches += 1
            else:
                mismatches += 1

        logger.info(
            "LoRA load audit: %d/%d tensors match (mismatch=%d, not_found=%d, zero_in_file=%d)",
            matches, total, mismatches, not_found, zero_in_file,
        )
        if matches < int(0.95 * total):
            raise RuntimeError(
                f"LoRA weights were NOT properly loaded: only {matches}/{total} "
                f"saved tensors match live model params. Inference would run with "
                f"unfine-tuned weights. mismatches={mismatches} not_found={not_found}."
            )

    def _build_test_data(self):
        cfg = self.train_cfg
        test_root = self.args.test_data_root or cfg.get("test_data_root", "")
        if not test_root or not os.path.isdir(test_root):
            logger.warning("Test data root not found: %s", test_root)
            self.test_loader = None
            return

        data_config = VLADataConfig(
            data_root=test_root,
            num_history_steps=cfg.get("num_history_steps", 1),
            history_stride=cfg.get("history_stride", 10),
            current_resolution=(cfg["current_res_w"], cfg["current_res_h"]),
            history_resolution=(cfg["history_res_w"], cfg["history_res_h"]),
            trajectory_future_seconds=cfg.get("trajectory_future_seconds", 5.0),
            frame_rate_hz=cfg.get("frame_rate_hz", DEFAULT_FRAME_RATE_HZ),
            num_waypoints=cfg.get("num_trajectory_points", 10),
            max_history_traj_points=cfg.get("max_history_traj_points", 6),
            reasoning_format=cfg.get("reasoning_format", "spatial_driving_counterfactual"),
            reasoning_max_items_per_list=cfg.get("reasoning_max_items_per_list", 10),
        )
        test_dataset = NuReasoningVLADataset(data_config, split="test")
        self._filter_test_dataset_to_keyframes(test_dataset)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=vla_collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("Test samples: %d", len(test_dataset))

    def _sample_clip_time_s(self, sample: Dict[str, Any]) -> Optional[float]:
        frame_idx = sample.get("frame_index", -1)
        frames = sample.get("frames", [])
        metadata = sample.get("metadata", {})
        if 0 <= frame_idx < len(frames):
            relative_time = frames[frame_idx].get("relative_time_s")
            if isinstance(relative_time, (int, float)):
                return float(relative_time)

        start_ts = metadata.get("start_timestamp_us")
        timestamp_us = sample.get("timestamp_us")
        if start_ts is not None and timestamp_us is not None:
            return (float(timestamp_us) - float(start_ts)) / 1e6

        frame_rate_hz = metadata.get("frame_rate_hz")
        if isinstance(frame_rate_hz, (int, float)) and frame_rate_hz > 0:
            return float(frame_idx) / float(frame_rate_hz)
        return None

    def _filter_test_dataset_to_keyframes(self, dataset: NuReasoningVLADataset) -> None:
        target_time_s = self.args.keyframe_time_s
        if target_time_s < 0:
            logger.info("Keeping all test samples (keyframe filter disabled).")
            return

        original_count = len(dataset.samples)
        if original_count == 0:
            return

        clip_count = len({sample["clip_name"] for sample in dataset.samples})
        best_by_clip: Dict[str, tuple[tuple[float, int], int]] = {}
        for idx, sample in enumerate(dataset.samples):
            clip_name = sample["clip_name"]
            clip_time_s = self._sample_clip_time_s(sample)
            if clip_time_s is None:
                continue
            score = (abs(clip_time_s - target_time_s), sample["frame_index"])
            best = best_by_clip.get(clip_name)
            if best is None or score < best[0]:
                best_by_clip[clip_name] = (score, idx)

        if not best_by_clip:
            logger.warning(
                "Could not identify %.1fs key frames; keeping all %d test samples.",
                target_time_s,
                original_count,
            )
            return

        selected_indices = {best_idx for _, best_idx in best_by_clip.values()}
        dataset.samples = [
            sample for idx, sample in enumerate(dataset.samples)
            if idx in selected_indices
        ]
        logger.info(
            "Filtered test samples to one key frame per clip near %.1fs: %d -> %d samples across %d/%d clips",
            target_time_s,
            original_count,
            len(dataset.samples),
            len(best_by_clip),
            clip_count,
        )

    # ------------------------------------------------------------------
    # VLM input preparation (mirrors VLATrainer._prepare_vlm_inputs)
    # ------------------------------------------------------------------

    def _prepare_vlm_inputs(
        self, batch: Dict[str, Any], *, for_generation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Build VLM inputs.

        - Reasoning generation (``for_generation=True``): structured multi-view
          user turn, ``add_generation_prompt=True``.
        - Planning / feature extraction (``for_generation=False``): planning
          user prompt (``--planning_prompt``) plus GT reasoning as the assistant
          response. Labels are unused; ``prompt_length`` selects prefix hidden
          states for the action expert.
        """
        cfg = self.train_cfg
        num_history_steps = cfg.get("num_history_steps", 1)

        images = batch["images"][0]
        images = [img for img in images if img is not None]

        image_contexts = []
        for t in range(-num_history_steps, 1):
            t_label = f"t={t}" if t < 0 else "t=0 (current)"
            for cam in CAMERA_NAMES:
                image_contexts.append(
                    f"{t_label}, {cam} camera."
                )
        image_contexts = image_contexts[: len(images)]

        if for_generation:
            prompt = self.vlm.build_multiview_prompt(
                num_history_steps=num_history_steps,
                num_current_cameras=len(CAMERA_NAMES),
                mission_command=batch["mission_commands"][0],
            )
        else:
            prompt = self._build_planning_prompt_text(batch)

        assistant_response = None if for_generation else batch["reasoning_texts"][0]

        inputs = self.vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=image_contexts,
            assistant_response=assistant_response,
        )

        result: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v

        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop("prompt_length").to(self.device)

        return result

    # ------------------------------------------------------------------
    # Reasoning generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reasoning(self, batch: Dict[str, Any]) -> str:
        vlm_inputs = self._prepare_vlm_inputs(batch, for_generation=True)

        generate_kwargs: Dict[str, Any] = {
            "input_ids": vlm_inputs["input_ids"],
            "attention_mask": vlm_inputs["attention_mask"],
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "temperature": self.args.temperature if self.args.temperature > 0 else None,
            "top_p": self.args.top_p if self.args.temperature > 0 else None,
        }
        if "pixel_values" in vlm_inputs:
            generate_kwargs["pixel_values"] = vlm_inputs["pixel_values"]
        if "image_grid_thw" in vlm_inputs:
            generate_kwargs["image_grid_thw"] = vlm_inputs["image_grid_thw"]
        if "mm_token_type_ids" in vlm_inputs:
            generate_kwargs["mm_token_type_ids"] = vlm_inputs["mm_token_type_ids"]

        output_ids = self.vlm.model.generate(**generate_kwargs)

        prompt_len = vlm_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_len:]
        return self.vlm.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        )

    @torch.no_grad()
    def run_reasoning_evaluation(self):
        if self.test_loader is None:
            logger.error("No test data available for reasoning evaluation.")
            return

        num_samples = min(self.args.num_reasoning_samples, len(self.test_loader.dataset))
        vis = self.args.visualize
        vis_dir = self._get_vis_dir("reasoning") if vis else None

        logger.info("=" * 60)
        logger.info("Reasoning Generation (%d samples, visualize=%s)", num_samples, vis)
        logger.info("=" * 60)

        results = []
        for i, batch in enumerate(self.test_loader):
            if i >= num_samples:
                break

            t0 = time.time()
            generated = self.generate_reasoning(batch)
            elapsed = time.time() - t0

            clip_name = batch["clip_names"][0]
            frame_idx = batch["frame_indices"][0]
            gt_reasoning = batch["reasoning_texts"][0]

            logger.info(
                "\n--- Sample %d/%d [%s frame=%d] (%.1fs) ---",
                i + 1, num_samples, clip_name, frame_idx, elapsed,
            )
            logger.info("GT reasoning:\n%s", gt_reasoning)
            logger.info("Generated:\n%s", generated)

            if vis:
                self._plot_reasoning_sample(batch, generated, i, vis_dir)

            results.append({
                "clip_name": clip_name,
                "frame_index": frame_idx,
                "ground_truth": gt_reasoning,
                "generated": generated,
                "time_s": elapsed,
            })

        output_path = os.path.join(self.args.output_dir, "reasoning_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Reasoning results saved to %s", output_path)

    # ------------------------------------------------------------------
    # Planning benchmark
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_trajectory(
        self, batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pred_traj, gt_traj), both [B, T, 3] on device."""
        vlm_inputs = self._prepare_vlm_inputs(batch, for_generation=False)

        vlm_outputs = self.vlm(
            input_ids=vlm_inputs["input_ids"],
            attention_mask=vlm_inputs["attention_mask"],
            pixel_values=vlm_inputs.get("pixel_values"),
            image_grid_thw=vlm_inputs.get("image_grid_thw"),
            mm_token_type_ids=vlm_inputs.get("mm_token_type_ids"),
            labels=None,
            prompt_lengths=vlm_inputs.get("prompt_lengths"),
            return_features=True,
        )
        vlm_features = vlm_outputs["vlm_features"]

        ego_history_traj = batch["ego_history_trajectories"].to(self.device)
        ego_velocities = batch["ego_velocities"].to(self.device)
        ego_accelerations = batch["ego_accelerations"].to(self.device)
        ego_state = torch.cat([ego_velocities, ego_accelerations], dim=-1)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            pred_traj = self.action_expert.sample(
                vlm_features, ego_state, ego_history_traj,
                num_steps=self.args.num_inference_steps,
            )
        gt_traj = batch["ego_trajectories"].to(self.device)

        return pred_traj, gt_traj

    @torch.no_grad()
    def run_planning_benchmark(self):
        if self.test_loader is None:
            logger.error("No test data available for planning benchmark.")
            return

        vis = self.args.visualize
        vis_dir = self._get_vis_dir("planning") if vis else None
        vis_count = 0
        max_vis = self.args.max_vis_samples

        logger.info("=" * 60)
        logger.info("Planning Benchmark (%d samples, %d ODE steps, visualize=%s)",
                     len(self.test_loader.dataset), self.args.num_inference_steps, vis)
        logger.info("=" * 60)

        agg: Dict[str, float] = {}
        count = 0
        t_start = time.time()

        num_traj_pts = self.train_cfg.get("num_trajectory_points", 10)
        per_horizon_pos_errors: List[List[float]] = [[] for _ in range(num_traj_pts)]

        for i, batch in enumerate(self.test_loader):
            pred_traj, gt_traj = self.predict_trajectory(batch)

            metrics = compute_trajectory_metrics(pred_traj, gt_traj)
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1

            pos_err = torch.sqrt(
                (pred_traj[..., 0] - gt_traj[..., 0]) ** 2
                + (pred_traj[..., 1] - gt_traj[..., 1]) ** 2
            )
            for t_idx in range(pos_err.shape[1]):
                per_horizon_pos_errors[t_idx].append(pos_err[0, t_idx].item())

            if vis and vis_count < max_vis:
                self._plot_planning_sample(batch, pred_traj, gt_traj, i, vis_dir)
                vis_count += 1

            if (i + 1) % self.args.log_interval == 0:
                elapsed = time.time() - t_start
                running = {k: v / count for k, v in agg.items()}
                logger.info(
                    "  [%d/%d] ADE=%.3fm | FDE=%.3fm | Heading=%.2f° (%.1fs)",
                    i + 1, len(self.test_loader),
                    running["ADE_m"], running["FDE_m"],
                    running["heading_error_deg"], elapsed,
                )

        total_time = time.time() - t_start
        final = {k: v / max(count, 1) for k, v in agg.items()}

        logger.info("=" * 60)
        logger.info("Planning Benchmark Results (%d samples, %.1fs)", count, total_time)
        logger.info("  ADE:           %.3f m", final["ADE_m"])
        logger.info("  FDE:           %.3f m", final["FDE_m"])
        logger.info("  Heading error: %.2f° (%.4f rad)",
                     final["heading_error_deg"], final["heading_error_rad"])

        dt_s = waypoint_dt_s(
            float(self.train_cfg.get("trajectory_future_seconds", 5.0)),
            int(self.train_cfg.get("num_trajectory_points", 10) or 10),
        )
        logger.info("  Per-horizon displacement error (m):")
        for t_idx, errs in enumerate(per_horizon_pos_errors):
            if errs:
                t_sec = (t_idx + 1) * dt_s
                mean_err = sum(errs) / len(errs)
                logger.info("    t=%.1fs: %.3f m", t_sec, mean_err)

        logger.info("=" * 60)

        output_path = os.path.join(self.args.output_dir, "planning_results.json")
        horizon_breakdown = {}
        for t_idx, errs in enumerate(per_horizon_pos_errors):
            if errs:
                t_sec = (t_idx + 1) * dt_s
                horizon_breakdown[f"{t_sec:.1f}s"] = sum(errs) / len(errs)

        with open(output_path, "w") as f:
            json.dump({
                "num_samples": count,
                "num_inference_steps": self.args.num_inference_steps,
                "total_time_s": total_time,
                "ADE_m": final["ADE_m"],
                "FDE_m": final["FDE_m"],
                "heading_error_deg": final["heading_error_deg"],
                "heading_error_rad": final["heading_error_rad"],
                "per_horizon_error_m": horizon_breakdown,
            }, f, indent=2)
        logger.info("Planning results saved to %s", output_path)

    # ------------------------------------------------------------------
    # Visualization helpers (called inline during eval loops)
    # ------------------------------------------------------------------

    def _get_vis_dir(self, subfolder: str) -> str:
        d = os.path.join(self.args.output_dir, "visualizations", subfolder)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _extract_image_grid(
        batch: Dict[str, Any], num_hist: int, num_cams: int,
    ) -> List[list]:
        raw_images = batch["images"][0]
        all_images = [img for img in raw_images if img is not None]
        grid: List[list] = []
        for t_idx in range(num_hist + 1):
            start = t_idx * num_cams
            grid.append(all_images[start : start + num_cams])
        return grid

    def _plot_camera_rows(
        self,
        fig,
        gridspec_mod,
        subplot_spec,
        image_grid: List[list],
        num_hist: int,
        num_cams: int,
    ):
        from PIL import Image as PILImage

        num_rows = max(num_hist + 1, 1)
        gs_images = gridspec_mod.GridSpecFromSubplotSpec(
            num_rows, num_cams, subplot_spec=subplot_spec, wspace=0.03, hspace=0.08,
        )

        current_imgs = image_grid[-1] if image_grid else []
        for c in range(min(num_cams, len(current_imgs))):
            ax = fig.add_subplot(gs_images[0, c])
            img = current_imgs[c]
            if isinstance(img, PILImage.Image):
                ax.imshow(np.array(img))
            else:
                ax.set_facecolor("#333")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("t=0", fontsize=7, rotation=0, labelpad=25)
            ax.set_title(
                CAMERA_NAMES[c].replace("_", " ").title(),
                fontsize=8, fontweight="bold",
            )

        for t in range(num_hist):
            row_imgs = image_grid[t] if t < len(image_grid) else []
            for c in range(min(num_cams, len(row_imgs))):
                ax = fig.add_subplot(gs_images[t + 1, c])
                img = row_imgs[c]
                if isinstance(img, PILImage.Image):
                    ax.imshow(np.array(img))
                else:
                    ax.set_facecolor("#333")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(
                        f"t={t - num_hist}", fontsize=7, rotation=0, labelpad=25,
                    )

    def _build_planning_prompt_text(self, batch: Dict[str, Any]) -> str:
        cfg = self.train_cfg
        return build_planning_user_prompt(
            self.vlm,
            num_history_steps=cfg.get("num_history_steps", 1),
            num_current_cameras=len(CAMERA_NAMES),
            mission_command=batch["mission_commands"][0],
            planning_prompt=getattr(self.args, "planning_prompt", PLANNING_PROMPT_AUTO),
            reasoning_mode=str(cfg.get("reasoning_mode") or "structured"),
            reasoning_format=getattr(self.args, "planning_reasoning_format", None),
        )

    def _build_prompt_text(self, batch: Dict[str, Any], *, for_planning: bool = False) -> str:
        if for_planning:
            return self._build_planning_prompt_text(batch)
        cfg = self.train_cfg
        return self.vlm.build_multiview_prompt(
            num_history_steps=cfg.get("num_history_steps", 1),
            num_current_cameras=len(CAMERA_NAMES),
            mission_command=batch["mission_commands"][0],
        )

    def _plot_reasoning_sample(
        self,
        batch: Dict[str, Any],
        generated_text: str,
        sample_idx: int,
        save_dir: str,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        cfg = self.train_cfg
        num_hist = cfg.get("num_history_steps", 1)
        num_cams = len(CAMERA_NAMES)
        clip_name = batch["clip_names"][0]
        frame_idx = batch["frame_indices"][0]
        gt_reasoning = batch["reasoning_texts"][0]
        prompt = self._build_prompt_text(batch)
        image_grid = self._extract_image_grid(batch, num_hist, num_cams)
        fig_height = max(18, 12 + 3 * (num_hist + 1))

        fig = plt.figure(figsize=(28, fig_height))
        outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[max(num_hist + 1, 2), 1.25],
            hspace=0.25,
        )

        self._plot_camera_rows(fig, gridspec, outer[0], image_grid, num_hist, num_cams)

        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[1],
            width_ratios=[1.0, 1.2, 1.2],
            wspace=0.12,
        )

        ax_prompt = fig.add_subplot(gs_bottom[0, 0])
        ax_prompt.axis("off")
        prompt_short = prompt[:600] + "..." if len(prompt) > 600 else prompt
        ax_prompt.text(
            0, 1, f"PROMPT:\n{prompt_short}",
            transform=ax_prompt.transAxes,
            fontsize=6, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef6ff", alpha=0.9),
        )
        ax_prompt.set_title("Prompt", fontsize=9, fontweight="bold")

        ax_gt = fig.add_subplot(gs_bottom[0, 1])
        ax_gt.axis("off")
        wrapped_gt = textwrap.fill(gt_reasoning, width=72)
        if len(wrapped_gt) > 1500:
            wrapped_gt = wrapped_gt[:1500] + "\n..."
        ax_gt.text(
            0, 1, wrapped_gt,
            transform=ax_gt.transAxes,
            fontsize=6.5, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eeffee", alpha=0.9),
        )
        ax_gt.set_title("GT Reasoning", fontsize=9, fontweight="bold")

        ax_gen = fig.add_subplot(gs_bottom[0, 2])
        ax_gen.axis("off")
        wrapped_gen = textwrap.fill(generated_text, width=72)
        if len(wrapped_gen) > 1500:
            wrapped_gen = wrapped_gen[:1500] + "\n..."
        ax_gen.text(
            0, 1, wrapped_gen,
            transform=ax_gen.transAxes,
            fontsize=6.5, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5ee", alpha=0.9),
        )
        ax_gen.set_title("Generated Reasoning", fontsize=9, fontweight="bold")

        fig.suptitle(
            f"Reasoning — sample {sample_idx}  |  {clip_name}  frame={frame_idx}  "
            f"mission={batch['mission_commands'][0]}",
            fontsize=11, fontweight="bold", y=0.98,
        )

        path = os.path.join(save_dir, f"reasoning_{sample_idx:04d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved reasoning visualization to %s", path)

    def _plot_planning_sample(
        self,
        batch: Dict[str, Any],
        pred_traj: torch.Tensor,
        gt_traj: torch.Tensor,
        sample_idx: int,
        save_dir: str,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        cfg = self.train_cfg
        num_hist = cfg.get("num_history_steps", 1)
        num_cams = len(CAMERA_NAMES)
        num_traj_pts = cfg.get("num_trajectory_points", 10)
        dt_s = waypoint_dt_s(
            float(cfg.get("trajectory_future_seconds", 5.0)),
            int(num_traj_pts or 10),
        )
        clip_name = batch["clip_names"][0]
        frame_idx = batch["frame_indices"][0]
        prompt = self._build_prompt_text(batch, for_planning=True)
        image_grid = self._extract_image_grid(batch, num_hist, num_cams)

        pred_np = pred_traj[0].float().cpu().numpy()
        gt_np = gt_traj[0].float().cpu().numpy()
        fig_height = max(18, 12 + 3 * (num_hist + 1))

        fig = plt.figure(figsize=(28, fig_height))
        outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[max(num_hist + 1, 2), 1.25],
            hspace=0.25,
        )

        self._plot_camera_rows(fig, gridspec, outer[0], image_grid, num_hist, num_cams)

        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[1.2, 1.0],
            wspace=0.12,
        )

        # ── Trajectory BEV ──
        ax_traj = fig.add_subplot(gs_bottom[0, 0])
        ax_traj.set_aspect("equal")
        ax_traj.set_facecolor("#f7f7f7")

        time_labels = [(i + 1) * dt_s for i in range(num_traj_pts)]

        ax_traj.plot(
            gt_np[:, 0], gt_np[:, 1], "o-",
            color="#2ecc71", markersize=5, linewidth=2.0,
            label="GT trajectory", zorder=5,
        )
        ax_traj.plot(
            pred_np[:, 0], pred_np[:, 1], "s-",
            color="#e74c3c", markersize=5, linewidth=2.0,
            label="Predicted trajectory", zorder=5,
        )
        for idx in range(0, min(num_traj_pts, len(gt_np)), 2):
            ax_traj.annotate(
                f"{time_labels[idx]:.1f}s",
                (gt_np[idx, 0], gt_np[idx, 1]),
                fontsize=6, color="#27ae60",
                textcoords="offset points", xytext=(4, 4),
            )
            ax_traj.annotate(
                f"{time_labels[idx]:.1f}s",
                (pred_np[idx, 0], pred_np[idx, 1]),
                fontsize=6, color="#c0392b",
                textcoords="offset points", xytext=(4, -8),
            )

        ax_traj.plot(
            0, 0, "D", color="royalblue", markersize=10,
            zorder=10, label="Ego (origin)",
        )

        ego_hist = batch["ego_history_trajectories"][0].numpy()
        valid_hist = ego_hist[np.abs(ego_hist).sum(axis=-1) > 1e-8]
        if len(valid_hist) > 0:
            ax_traj.plot(
                valid_hist[:, 0], valid_hist[:, 1], "^--",
                color="#9b59b6", markersize=4, linewidth=1.2,
                label="History", alpha=0.7, zorder=3,
            )

        metrics = compute_trajectory_metrics(
            pred_traj[:1].to(gt_traj.device), gt_traj[:1],
        )
        ax_traj.set_title(
            f"Ego-Frame Trajectory\n"
            f"ADE={metrics['ADE_m']:.2f}m  FDE={metrics['FDE_m']:.2f}m  "
            f"Heading={metrics['heading_error_deg']:.1f}°",
            fontsize=9, fontweight="bold",
        )
        ax_traj.set_xlabel("x (m) — forward", fontsize=8)
        ax_traj.set_ylabel("y (m) — left", fontsize=8)
        ax_traj.legend(fontsize=7, loc="best")
        ax_traj.grid(True, alpha=0.3)

        # ── Prompt text ──
        ax_prompt = fig.add_subplot(gs_bottom[0, 1])
        ax_prompt.axis("off")
        prompt_short = prompt[:800] + "..." if len(prompt) > 800 else prompt
        ax_prompt.text(
            0, 1, f"PROMPT:\n{prompt_short}",
            transform=ax_prompt.transAxes,
            fontsize=6, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef6ff", alpha=0.9),
        )
        ax_prompt.set_title("Prompt", fontsize=9, fontweight="bold")

        fig.suptitle(
            f"Planning — sample {sample_idx}  |  {clip_name}  frame={frame_idx}  "
            f"mission={batch['mission_commands'][0]}",
            fontsize=11, fontweight="bold", y=0.98,
        )

        path = os.path.join(save_dir, f"planning_{sample_idx:04d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved planning visualization to %s", path)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        mode = self.args.mode
        if mode in ("reasoning", "both"):
            self.run_reasoning_evaluation()
        if mode in ("planning", "both"):
            self.run_planning_benchmark()


def parse_args():
    parser = argparse.ArgumentParser(description="VLA Model Evaluation")

    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to a saved epoch checkpoint (e.g. nureasoning_vla_workspace/epoch_3)",
    )
    parser.add_argument(
        "--test_data_root", type=str, default=None,
        help="Override test data root (default: use training_config.json value)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output files (default: the workspace folder, parent of --checkpoint_dir)",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["reasoning", "planning", "both"],
    )

    parser.add_argument("--visualize", action="store_true",
                        help="Save per-sample visualization figures during evaluation")
    parser.add_argument("--max_vis_samples", type=int, default=50,
                        help="Max samples to visualize (planning runs over full set)")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--keyframe_time_s",
        type=float,
        default=DEFAULT_KEYFRAME_TIME_S,
        help=(
            "Only evaluate one sample per clip: the sample nearest this many "
            "seconds into the clip. Set to a negative value to evaluate all frames."
        ),
    )

    parser.add_argument("--num_reasoning_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.1)
    add_planning_prompt_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        workspace = os.path.dirname(os.path.normpath(args.checkpoint_dir))
        args.output_dir = workspace if workspace else "."
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "eval.log"), mode="a"),
        ],
    )

    logger.info("Evaluation args: %s", json.dumps(vars(args), indent=2, default=str))

    tester = VLATester(args)
    tester.run()


if __name__ == "__main__":
    main()

"""
VLA Training Script.

Architecture:
  ┌─────────────┐          ┌──────────────────┐
  │  Multi-view │          │  VLM Backbone    │
  │  Multi-frame├─────────>│  (Qwen3-VL)      │
  │  Images     │          │                  │
  └─────────────┘          │ Reasoning Loss   │
                           │ (text generation)│
                           └───────┬──────────┘
                                   │
                              VLM features
                                   │
                           ┌───────▼───────────┐
                           │  Action Expert    │
                           │  (Flow-Match DiT) │
                           │                   │
                           │  Action Loss      │
                           │  (trajectory MSE) │
                           └───────────────────┘

Reasoning modes (VLM text target; the action expert always trains on trajectories):

  structured — Spatial / Driving / Counterfactual annotations
    python -m nureasoning.nuvla.train --reasoning_mode structured

  vqa — generated VQA (all question types; requires --vqa_root)
    python -m nureasoning.nuvla.train --reasoning_mode vqa --vqa_root ./vqa_output_train

Single-GPU:
    python -m nureasoning.nuvla.train

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=8 -m nureasoning.nuvla.train
"""

import argparse
import contextlib
import datetime
import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nureasoning.common.pretrained import resolve_pretrained_path
from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES, VLMBackbone, VLMBackboneConfig
from nureasoning.nuvla.models.action_expert import (
    FlowMatchingDiTActionExpert,
    ActionExpertConfig,
    compute_trajectory_metrics,
)
from nureasoning.nuvla.models.data_loader import (
    VLADataConfig,
    NuReasoningVLADataset,
    build_dataloader,
    vla_collate_fn,
)

logger = logging.getLogger(__name__)

REASONING_MODE_STRUCTURED = "structured"
REASONING_MODE_VQA = "vqa"
REASONING_MODES = (REASONING_MODE_STRUCTURED, REASONING_MODE_VQA)


# ======================================================================
# Distributed helpers
# ======================================================================


def setup_distributed() -> tuple:
    """
    Initialize DDP from environment variables set by torchrun / torch.distributed.launch.
    Returns (local_rank, global_rank, world_size).  Falls back to single-GPU if
    the environment is not configured for distributed training.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ======================================================================
# Trainer
# ======================================================================


class VLATrainer:
    """
    Joint VLA trainer with multi-GPU DDP support.

    Training losses:
      - VLM parameters are updated by the reasoning loss (text generation)
        in one of two modes: structured annotations, or VQA Q&A
      - Action expert parameters are updated by the action loss (flow matching)

    PEFT / LoRA:
      - When lora_rank > 0 the VLM backbone automatically applies LoRA adapters
        via the peft library.  Only adapter weights are saved/loaded in checkpoints.

    Multi-GPU:
      - Both models are wrapped in DistributedDataParallel when world_size > 1.
      - Gradient accumulation uses DDP no_sync to skip redundant all-reduces.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        self.args = args
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.is_distributed = world_size > 1
        self.is_main = global_rank == 0
        self.device = torch.device("cuda", local_rank)

        self._build_models()
        self._build_data()
        self._build_optimizers()

        self.global_step = 0
        self.best_val_loss = float("inf")

        if self.is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            self._save_config()

        if self.is_distributed:
            dist.barrier()

    # ------------------------------------------------------------------
    # Unwrapped module accessors
    # ------------------------------------------------------------------

    @property
    def vlm_unwrapped(self) -> VLMBackbone:
        return self.vlm.module if self.is_distributed else self.vlm

    @property
    def action_expert_unwrapped(self) -> FlowMatchingDiTActionExpert:
        return self.action_expert.module if self.is_distributed else self.action_expert

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_models(self):
        args = self.args

        args.vlm_model_path = resolve_pretrained_path(args.vlm_model_path)
        logger.info(
            "VLM weights: %s  (HF_HOME=%s)",
            args.vlm_model_path,
            os.environ.get("HF_HOME"),
        )

        vlm_config = VLMBackboneConfig(
            model_name_or_path=args.vlm_model_path,
            freeze_vision_encoder=args.freeze_vision_encoder,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            current_resolution=(args.current_res_w, args.current_res_h),
            history_resolution=(args.history_res_w, args.history_res_h),
            reasoning_format=args.reasoning_format,
        )
        self.vlm = VLMBackbone(vlm_config).to(self.device)

        vlm_feature_dim = self.vlm.feature_dim

        action_config = ActionExpertConfig(
            vlm_feature_dim=vlm_feature_dim,
            ego_state_dim=4,
            max_history_traj_points=args.max_history_traj_points,
            num_waypoints=args.num_trajectory_points,
            trajectory_dim=3,
            hidden_dim=args.action_hidden_dim,
            num_heads=args.num_dit_heads,
            num_dit_layers=args.num_dit_layers,
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            interleave_self_attention=args.interleave_self_attention,
            num_inference_steps=args.num_inference_steps,
            num_timestep_buckets=args.num_timestep_buckets,
            noise_beta_alpha=args.noise_beta_alpha,
            noise_beta_beta=args.noise_beta_beta,
        )
        self.action_expert = FlowMatchingDiTActionExpert(action_config).to(self.device)

        if self.is_main:
            logger.info(f"VLM parameters: {sum(p.numel() for p in self.vlm.parameters()):,}")
            logger.info(
                f"VLM trainable (LoRA rank={args.lora_rank}): "
                f"{sum(p.numel() for p in self.vlm.parameters() if p.requires_grad):,}"
            )
            logger.info(f"VLM feature dim (inferred): {vlm_feature_dim}")
            logger.info(
                f"Action expert parameters: "
                f"{sum(p.numel() for p in self.action_expert.parameters()):,}"
            )

        if self.is_distributed:
            self.vlm = DDP(
                self.vlm,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )
            self.action_expert = DDP(
                self.action_expert,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

    def _vla_data_config(self, data_root: str) -> VLADataConfig:
        """Build a dataloader config using the same reasoning target as training."""
        args = self.args
        use_vqa = args.reasoning_mode == REASONING_MODE_VQA
        return VLADataConfig(
            data_root=data_root,
            num_history_steps=args.num_history_steps,
            history_stride=args.history_stride,
            current_resolution=(args.current_res_w, args.current_res_h),
            history_resolution=(args.history_res_w, args.history_res_h),
            trajectory_future_seconds=args.trajectory_future_seconds,
            frame_rate_hz=args.frame_rate_hz,
            num_waypoints=args.num_trajectory_points,
            max_history_traj_points=args.max_history_traj_points,
            reasoning_format=args.reasoning_format,
            reasoning_max_items_per_list=args.reasoning_max_items_per_list,
            reasoning_mode=REASONING_MODE_VQA if use_vqa else REASONING_MODE_STRUCTURED,
            vqa_root=args.vqa_root if use_vqa else None,
            qa_seed=args.qa_seed,
        )

    def _build_data(self):
        args = self.args
        data_config = self._vla_data_config(args.data_root)

        dataset = NuReasoningVLADataset(data_config, split="train")

        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                dataset, num_replicas=self.world_size,
                rank=self.global_rank, shuffle=True,
            )
            self.train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=self.train_sampler,
                num_workers=args.num_workers,
                collate_fn=vla_collate_fn,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.train_sampler = None
            self.train_loader = build_dataloader(
                data_config,
                split="train",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )

        self.test_loader = None
        if args.test_data_root:
            test_config = self._vla_data_config(args.test_data_root)

            if self.is_distributed:
                test_dataset = NuReasoningVLADataset(test_config, split="test")
                self.test_sampler = DistributedSampler(
                    test_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                    drop_last=False,
                )
                self.test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    sampler=self.test_sampler,
                    num_workers=args.num_workers,
                    collate_fn=vla_collate_fn,
                    pin_memory=True,
                    drop_last=False,
                )
            else:
                self.test_sampler = None
                self.test_loader = build_dataloader(
                    test_config,
                    split="test",
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                )
        else:
            self.test_sampler = None

        if self.is_main:
            logger.info(f"Training samples: {len(self.train_loader.dataset)}")
            if self.test_loader is not None:
                logger.info(f"Testing samples: {len(self.test_loader.dataset)}")
            elif args.eval_interval > 0:
                logger.warning("No test data configured; test evaluation will be skipped.")
            if self.is_distributed:
                logger.info(
                    f"  Per-GPU batch size: {args.batch_size} × "
                    f"{self.world_size} GPUs = {args.batch_size * self.world_size} global"
                )

    def _build_optimizers(self):
        args = self.args

        vlm_params = [p for p in self.vlm_unwrapped.parameters() if p.requires_grad]
        self.vlm_optimizer = AdamW(
            vlm_params,
            lr=args.vlm_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        self.action_optimizer = AdamW(
            self.action_expert_unwrapped.parameters(),
            lr=args.action_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        total_steps = max(total_steps, 1)
        min_lr = 1e-6

        def build_warmup_cosine_lambda(base_lr: float):
            min_ratio = min(min_lr / max(base_lr, 1e-12), 1.0)

            def lr_lambda(current_step: int) -> float:
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step + 1) / float(warmup_steps)

                if total_steps <= warmup_steps:
                    return 1.0

                progress = (current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            return lr_lambda

        self.vlm_scheduler = LambdaLR(
            self.vlm_optimizer,
            lr_lambda=build_warmup_cosine_lambda(args.vlm_lr),
        )
        self.action_scheduler = LambdaLR(
            self.action_optimizer,
            lr_lambda=build_warmup_cosine_lambda(args.action_lr),
        )

        if self.is_main:
            logger.info(
                "Scheduler: warmup+cosine with %d warmup steps out of %d total",
                warmup_steps,
                total_steps,
            )

        if not torch.cuda.is_available():
            raise RuntimeError("BF16-only training requires CUDA.")
        self.use_bf16 = True
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA device reports bf16 may be unsupported; training may fail.")

    def _save_config(self):
        config_path = os.path.join(self.args.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    def _prepare_single_vlm_input(
        self, batch: Dict[str, Any], batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Prepare one sample's VLM inputs with the reasoning text placed in the
        assistant turn (so training matches the chat template used at inference)
        and labels masked to score only the assistant response.
        """
        vlm = self.vlm_unwrapped
        images = batch["images"][batch_idx]
        images = [img for img in images if img is not None]

        image_contexts = []
        for t in range(-self.args.num_history_steps, 1):
            t_label = f"t={t}" if t < 0 else "t=0 (current)"

            for cam in CAMERA_NAMES:
                image_contexts.append(
                    f"{t_label}, {cam} camera."
                )
        image_contexts = image_contexts[: len(images)]

        prompt = (batch.get("user_prompts") or [None] * (batch_idx + 1))[batch_idx]
        if not prompt:
            prompt = vlm.build_multiview_prompt(
                num_history_steps=self.args.num_history_steps,
                num_current_cameras=len(CAMERA_NAMES),
                mission_command=batch["mission_commands"][batch_idx],
            )

        reasoning_text = batch["reasoning_texts"][batch_idx]

        inputs = vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=image_contexts,
            assistant_response=reasoning_text,
        )

        result: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v

        # Rename `prompt_length` -> `prompt_lengths` for forward()'s contract.
        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop("prompt_length").to(self.device)

        return result

    def _prepare_vlm_inputs(
        self, batch: Dict[str, Any],
    ) -> list[Dict[str, torch.Tensor]]:
        """Prepare per-sample VLM inputs for a batch."""
        return [
            self._prepare_single_vlm_input(batch, b)
            for b in range(len(batch["images"]))
        ]

    @staticmethod
    def _stack_vlm_features(
        features_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pad variable-length prompt features to a dense batch tensor."""
        if not features_list:
            raise ValueError("features_list must not be empty")

        batch_size = len(features_list)
        max_seq_len = max(feat.shape[1] for feat in features_list)
        hidden_dim = features_list[0].shape[2]
        stacked = features_list[0].new_zeros(batch_size, max_seq_len, hidden_dim)

        for idx, feat in enumerate(features_list):
            seq_len = feat.shape[1]
            stacked[idx, :seq_len] = feat[0]

        return stacked

    def _extract_vlm_features(
        self,
        batch: Dict[str, Any],
        *,
        compute_reasoning_loss: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the VLM sample-by-sample and re-batch the prompt features."""
        sample_inputs = self._prepare_vlm_inputs(batch)
        feature_list = []
        reasoning_losses = []

        vlm_module = self.vlm if compute_reasoning_loss else self.vlm_unwrapped

        for sample_input in sample_inputs:
            with torch.amp.autocast("cuda", enabled=self.use_bf16, dtype=torch.bfloat16):
                vlm_outputs = vlm_module(
                    input_ids=sample_input["input_ids"],
                    attention_mask=sample_input["attention_mask"],
                    pixel_values=sample_input.get("pixel_values"),
                    image_grid_thw=sample_input.get("image_grid_thw"),
                    mm_token_type_ids=sample_input.get("mm_token_type_ids"),
                    labels=sample_input.get("labels") if compute_reasoning_loss else None,
                    prompt_lengths=sample_input.get("prompt_lengths"),
                    return_features=True,
                )

            feature_list.append(vlm_outputs["vlm_features"])
            if compute_reasoning_loss:
                reasoning_losses.append(
                    vlm_outputs.get("loss", torch.tensor(0.0, device=self.device))
                )

        vlm_features = self._stack_vlm_features(feature_list)
        reasoning_loss = None
        if compute_reasoning_loss:
            reasoning_loss = torch.stack(reasoning_losses).mean()

        return vlm_features, reasoning_loss

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step.

        Step 1: Forward VLM, compute reasoning loss, extract features
        Step 2: Forward action expert with VLM features, compute action loss
        Step 3: Backward reasoning loss -> update VLM
        Step 4: Backward action loss -> update action expert
        """
        self.vlm.train()
        self.action_expert.train()

        # === Step 1: VLM Forward ===
        vlm_features, reasoning_loss = self._extract_vlm_features(
            batch,
            compute_reasoning_loss=True,
        )
        if reasoning_loss is None:
            reasoning_loss = torch.tensor(0.0, device=self.device)

        # === Step 2: Action Expert Forward ===
        ego_trajectories = batch["ego_trajectories"].to(self.device)
        ego_history_trajectories = batch["ego_history_trajectories"].to(self.device)
        ego_velocities = batch["ego_velocities"].to(self.device)
        ego_accelerations = batch["ego_accelerations"].to(self.device)
        ego_state = torch.cat([ego_velocities, ego_accelerations], dim=-1)

        with torch.amp.autocast("cuda", enabled=self.use_bf16, dtype=torch.bfloat16):
            action_outputs = self.action_expert(
                x_1=ego_trajectories,
                vlm_features=vlm_features,
                ego_state=ego_state,
                history_trajectory=ego_history_trajectories,
            )

        action_loss = action_outputs["loss"]

        # === Step 3: joint loss ===
        total_loss = reasoning_loss + self.args.action_loss_weight * action_loss
        total_loss.backward()

        metrics = {
            "reasoning_loss": reasoning_loss.item(),
            "action_loss": action_loss.item(),
            "total_loss": reasoning_loss.item() + self.args.action_loss_weight * action_loss.item(),
            "mse_x": action_outputs["mse_x"].item(),
            "mse_y": action_outputs["mse_y"].item(),
            "mse_theta": action_outputs["mse_theta"].item(),
        }

        return metrics

    def optimizer_step(self):
        """Perform optimizer step for both VLM and action expert."""
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.vlm_unwrapped.parameters() if p.requires_grad],
                self.args.max_grad_norm,
            )
            torch.nn.utils.clip_grad_norm_(
                self.action_expert_unwrapped.parameters(),
                self.args.max_grad_norm,
            )

        self.vlm_optimizer.step()
        self.action_optimizer.step()

        self.vlm_optimizer.zero_grad()
        self.action_optimizer.zero_grad()

        self.vlm_scheduler.step()
        self.action_scheduler.step()

        self.global_step += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_trajectory(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate trajectory prediction quality."""
        self.vlm_unwrapped.eval()
        self.action_expert_unwrapped.eval()

        vlm_features, _ = self._extract_vlm_features(
            batch,
            compute_reasoning_loss=False,
        )

        ego_history_trajectories = batch["ego_history_trajectories"].to(self.device)
        ego_velocities = batch["ego_velocities"].to(self.device)
        ego_accelerations = batch["ego_accelerations"].to(self.device)
        ego_state = torch.cat([ego_velocities, ego_accelerations], dim=-1)

        action_expert = self.action_expert_unwrapped
        with torch.amp.autocast("cuda", enabled=self.use_bf16, dtype=torch.bfloat16):
            pred_traj = action_expert.sample(
                vlm_features, ego_state, ego_history_trajectories,
            )
        pred_traj = pred_traj.float()
        gt_traj = batch["ego_trajectories"].to(self.device).float()

        metrics = compute_trajectory_metrics(pred_traj, gt_traj)
        return metrics

    @torch.no_grad()
    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate trajectory metrics on the configured testing data.

        Runs on every rank (the test set is sharded via ``DistributedSampler``
        when distributed) and aggregates the weighted sums across ranks with a
        single all-reduce so every rank returns identical averages.
        """
        if self.test_loader is None:
            raise RuntimeError("Testing data loader is not available.")

        agg: Dict[str, float] = {}
        total_samples = 0

        for batch_idx, batch in enumerate(self.test_loader):
            metrics = self.evaluate_trajectory(batch)
            batch_size = int(batch["ego_trajectories"].shape[0])
            total_samples += batch_size

            for key, value in metrics.items():
                agg[key] = agg.get(key, 0.0) + value * batch_size

        if self.is_distributed:
            # All ranks must agree on the metric keys for the all-reduce to
            # line up; sort so iteration order is deterministic across ranks.
            metric_keys = sorted(agg.keys())
            stacked = torch.tensor(
                [agg.get(k, 0.0) for k in metric_keys] + [float(total_samples)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            reduced = stacked.tolist()
            agg = {k: reduced[i] for i, k in enumerate(metric_keys)}
            total_samples = int(round(reduced[-1]))

        if total_samples == 0:
            raise RuntimeError("Testing data loader produced no samples.")

        averaged = {key: value / total_samples for key, value in agg.items()}
        averaged["num_eval_samples"] = float(total_samples)
        return averaged

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        if self.is_main:
            logger.info("=" * 60)
            logger.info("Starting VLA Training")
            logger.info(f"  Epochs: {args.epochs}")
            logger.info(f"  Per-GPU batch size: {args.batch_size}")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
            eff_bs = args.batch_size * self.world_size * args.gradient_accumulation_steps
            logger.info(f"  Effective batch size: {eff_bs}")
            logger.info(f"  VLM LR: {args.vlm_lr}")
            logger.info(f"  Action LR: {args.action_lr}")
            logger.info(f"  LoRA rank: {args.lora_rank}  alpha: {args.lora_alpha}  dropout: {args.lora_dropout}")
            logger.info(f"  Reasoning mode: {args.reasoning_mode}")
            if args.reasoning_mode == REASONING_MODE_VQA:
                logger.info(
                    "  VQA root: %s  seed: %d",
                    args.vqa_root,
                    args.qa_seed,
                )
            else:
                logger.info(f"  Reasoning format: {args.reasoning_format}")
            logger.info(f"  Action loss weight: {args.action_loss_weight}")
            logger.info("  Precision: bf16-only (autocast enabled on CUDA)")
            logger.info("=" * 60)

        self.vlm_optimizer.zero_grad()
        self.action_optimizer.zero_grad()

        for epoch in range(args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            epoch_metrics = {
                "reasoning_loss": 0.0,
                "action_loss": 0.0,
                "total_loss": 0.0,
            }
            num_steps = 0

            for step, batch in enumerate(self.train_loader):
                is_sync_step = (step + 1) % args.gradient_accumulation_steps == 0

                if self.is_distributed and not is_sync_step:
                    vlm_ctx = self.vlm.no_sync()
                    act_ctx = self.action_expert.no_sync()
                else:
                    vlm_ctx = contextlib.nullcontext()
                    act_ctx = contextlib.nullcontext()

                with vlm_ctx, act_ctx:
                    step_metrics = self.train_step(batch)

                for k, v in step_metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k] += v

                if is_sync_step:
                    self.optimizer_step()

                num_steps += 1

                if self.is_main and (step + 1) % args.log_interval == 0:
                    avg_reasoning = epoch_metrics["reasoning_loss"] / num_steps
                    avg_action = epoch_metrics["action_loss"] / num_steps
                    vlm_lr = self.vlm_scheduler.get_last_lr()[0]
                    action_lr = self.action_scheduler.get_last_lr()[0]

                    logger.info(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {step+1}/{len(self.train_loader)} | "
                        f"Reasoning Loss: {avg_reasoning:.4f} | "
                        f"Action Loss: {avg_action:.4f} | "
                        f"VLM LR: {vlm_lr:.2e} | "
                        f"Action LR: {action_lr:.2e} | "
                        f"x_mse: {step_metrics['mse_x']:.4f} | "
                        f"y_mse: {step_metrics['mse_y']:.4f}"
                    )

            epoch_time = time.time() - epoch_start
            avg_metrics = {k: v / max(num_steps, 1) for k, v in epoch_metrics.items()}

            if self.is_main:
                logger.info(
                    f"\nEpoch {epoch+1} complete in {epoch_time:.1f}s | "
                    f"Avg Reasoning Loss: {avg_metrics['reasoning_loss']:.4f} | "
                    f"Avg Action Loss: {avg_metrics['action_loss']:.4f}"
                )

            if (epoch + 1) % args.eval_interval == 0:
                if self.test_loader is None:
                    if self.is_main:
                        logger.warning(
                            "Skipping evaluation because no testing data loader is configured."
                        )
                else:
                    traj_metrics = self.evaluate_test_set()
                    if self.is_main:
                        logger.info(
                            f"  [Test] ADE={traj_metrics['ADE_m']:.2f}m | "
                            f"FDE={traj_metrics['FDE_m']:.2f}m | "
                            f"Heading={traj_metrics['heading_error_deg']:.1f}° "
                            f"over {int(traj_metrics['num_eval_samples'])} samples"
                        )

            if (epoch + 1) % args.save_interval == 0:
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(args.epochs, is_final=True)
        if self.is_main:
            logger.info("Training complete!")

    # ------------------------------------------------------------------
    # Checkpoint save / load 
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        if self.is_main:
            suffix = "final" if is_final else f"epoch_{epoch}"
            save_dir = os.path.join(self.args.output_dir, suffix)
            os.makedirs(save_dir, exist_ok=True)

            vlm = self.vlm_unwrapped
            vlm.save_adapter(os.path.join(save_dir, "vlm_adapter"))

            action_state = {
                k: v.cpu() for k, v in self.action_expert_unwrapped.state_dict().items()
            }
            torch.save(action_state, os.path.join(save_dir, "action_expert.pt"))

            torch.save({
                "vlm_optimizer": self.vlm_optimizer.state_dict(),
                "action_optimizer": self.action_optimizer.state_dict(),
                "vlm_scheduler": self.vlm_scheduler.state_dict(),
                "action_scheduler": self.action_scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": epoch,
            }, os.path.join(save_dir, "training_state.pt"))

            logger.info(f"Checkpoint saved to {save_dir}")

        if self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, checkpoint_dir: str):
        vlm = self.vlm_unwrapped

        vlm_adapter_dir = os.path.join(checkpoint_dir, "vlm_adapter")
        if os.path.isdir(vlm_adapter_dir):
            vlm.load_adapter(vlm_adapter_dir, device=self.device)
        else:
            vlm_path = os.path.join(checkpoint_dir, "vlm_backbone.pt")
            if os.path.isfile(vlm_path):
                state = torch.load(vlm_path, map_location=self.device)
                vlm.load_state_dict(state)

        action_path = os.path.join(checkpoint_dir, "action_expert.pt")
        if os.path.isfile(action_path):
            self.action_expert_unwrapped.load_state_dict(
                torch.load(action_path, map_location=self.device)
            )

        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.vlm_optimizer.load_state_dict(state["vlm_optimizer"])
            self.action_optimizer.load_state_dict(state["action_optimizer"])
            self.vlm_scheduler.load_state_dict(state["vlm_scheduler"])
            self.action_scheduler.load_state_dict(state["action_scheduler"])
            self.global_step = state["global_step"]
            logger.info(f"Resumed from step {self.global_step}")

        if self.is_distributed:
            dist.barrier()


def resolve_reasoning_training_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Pick one exclusive VLM text-target mode: structured annotations or VQA."""
    mode = args.reasoning_mode
    if mode is None:
        mode = REASONING_MODE_VQA if args.vqa_root else REASONING_MODE_STRUCTURED

    if mode == REASONING_MODE_VQA:
        if not args.vqa_root:
            parser.error(
                "--reasoning_mode vqa requires --vqa_root "
                "(directory from nureasoning.vqa.generate)"
            )
        if not os.path.isdir(args.vqa_root):
            parser.error(f"--vqa_root is not a directory: {args.vqa_root}")
    else:
        if args.vqa_root:
            parser.error(
                "--reasoning_mode structured does not use --vqa_root; "
                "omit it, or pass --reasoning_mode vqa"
            )
        args.vqa_root = None

    args.reasoning_mode = mode


def parse_args():
    parser = argparse.ArgumentParser(description="VLA Training")

    parser.add_argument("--data_root", type=str, default="./dataset/data/train")
    parser.add_argument("--test_data_root", type=str, default="./dataset/data/validation")
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hub repo id or local snapshot directory (also checks ./models/<name>). "
             "Hub downloads go to ./models/.hf unless HF_HOME is set.",
    )
    parser.add_argument("--output_dir", type=str, default="./nureasoning_vla_workspace_spatial_driving_counterfactual")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--vlm_lr", type=float, default=5e-5)
    parser.add_argument("--action_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)

    # PEFT / LoRA
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (0 to disable LoRA and train full model)")
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default=None,
        choices=list(REASONING_MODES),
        help="VLM reasoning text target. 'structured' uses Spatial / "
             "Driving / Counterfactual annotations. 'vqa' uses generated "
             "VQA of every question type (requires --vqa_root). If omitted, 'vqa' is "
             "selected automatically when --vqa_root is set, otherwise 'structured'.",
    )
    parser.add_argument(
        "--reasoning_format",
        type=str,
        default="spatial_driving_counterfactual",
        choices=["driving", "spatial_driving", "spatial_driving_counterfactual", "spatial", "driving_counterfactual"],
        help="Which annotation sections to serialize in --reasoning_mode structured "
             "(ignored as the VLM target when --reasoning_mode vqa).",
    )
    parser.add_argument("--reasoning_max_items_per_list", type=int, default=10)
    parser.add_argument(
        "--vqa_root",
        type=str,
        default=None,
        help="Output of nureasoning.vqa.generate. Required for --reasoning_mode vqa. "
             "Passing this flag without --reasoning_mode selects vqa mode.",
    )
    parser.add_argument("--qa_seed", type=int, default=42)

    # Action expert (GR00T-style DiT)
    parser.add_argument("--action_hidden_dim", type=int, default=512)
    parser.add_argument("--num_dit_layers", type=int, default=12,
                        help="Number of DiT transformer layers (interleaved cross/self-attention)")
    parser.add_argument("--num_dit_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--interleave_self_attention", action="store_true", default=True,
                        help="Alternate cross-attention and self-attention blocks (GR00T pattern)")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--num_timestep_buckets", type=int, default=1000)
    parser.add_argument("--noise_beta_alpha", type=float, default=2.5,
                        help="Alpha for Beta distribution time sampling")
    parser.add_argument("--noise_beta_beta", type=float, default=1.5,
                        help="Beta for Beta distribution time sampling")

    # Data / trajectory
    parser.add_argument("--num_history_steps", type=int, default=1)
    parser.add_argument(
        "--history_stride",
        type=int,
        default=10,
        help="Camera-history stride in 10 Hz frames (10 ⇒ 1 s). "
             "If clip fps differs, this is converted by wall-clock time.",
    )
    parser.add_argument("--current_res_w", type=int, default=448)
    parser.add_argument("--current_res_h", type=int, default=448)
    parser.add_argument("--history_res_w", type=int, default=448)
    parser.add_argument("--history_res_h", type=int, default=448)
    parser.add_argument("--trajectory_future_seconds", type=float, default=5.0)
    parser.add_argument(
        "--frame_rate_hz",
        type=float,
        default=10.0,
        help="Native clip frame rate (official data is 10 Hz). Used as the "
             "unit of --history_stride and as a fallback when metadata has no fps.",
    )
    parser.add_argument(
        "--num_trajectory_points",
        type=int,
        default=10,
        help="Future waypoints from the action expert (default 10 over 5 s).",
    )
    parser.add_argument(
        "--max_history_traj_points",
        type=int,
        default=6,
        help="Ego-history waypoints on the 0.5 s grid (default 6 × 0.5 s = 3 s).",
    )

    # Logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()
    resolve_reasoning_training_args(args, parser)
    return args


# ======================================================================
# Entry point
# ======================================================================

def main():
    local_rank, global_rank, world_size = setup_distributed()

    args = parse_args()

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            *(
                [logging.FileHandler(os.path.join(args.output_dir, "train.log"), mode="a")]
                if global_rank == 0 and os.path.isdir(args.output_dir)
                else []
            ),
        ],
    )

    if global_rank == 0:
        logger.info(f"Arguments: {json.dumps(vars(args), indent=2, default=str)}")
        if world_size > 1:
            logger.info(f"Distributed training: {world_size} GPUs")

    trainer = VLATrainer(
        args,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nureasoning.nuvla.models.data_loader import (
    DEFAULT_FRAME_RATE_HZ,
    camera_history_stride,
    frame_stride_for_dt,
    infer_clip_frame_rate_hz,
    waypoint_dt_s,
)

logger = logging.getLogger(__name__)

PLANNING_PROMPT_AUTO = "auto"
PLANNING_PROMPT_STRUCTURED = "structured"
PLANNING_PROMPT_MINIMAL = "minimal"
PLANNING_PROMPT_MODES = (
    PLANNING_PROMPT_AUTO,
    PLANNING_PROMPT_STRUCTURED,
    PLANNING_PROMPT_MINIMAL,
)
REASONING_FORMATS = (
    "driving",
    "spatial_driving",
    "spatial_driving_counterfactual",
    "spatial",
    "driving_counterfactual",
)

# Fallbacks match ``nureasoning.nuvla.train.parse_args`` defaults.
_TRAIN_DEFAULTS: Dict[str, Any] = {
    "num_history_steps": 1,
    "history_stride": 10,
    "current_res_w": 448,
    "current_res_h": 448,
    "history_res_w": 448,
    "history_res_h": 448,
    "frame_rate_hz": DEFAULT_FRAME_RATE_HZ,
    "trajectory_future_seconds": 5.0,
    "num_trajectory_points": 10,
    "max_history_traj_points": 6,
    "num_inference_steps": 5,
    "noise_beta_alpha": 2.5,
    "noise_beta_beta": 1.5,
    "reasoning_format": "spatial_driving_counterfactual",
    "reasoning_mode": "structured",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_dropout": 0.0,
}


def resolve_planning_prompt_kind(
    planning_prompt: str,
    reasoning_mode: str = "structured",
) -> str:
    """Return ``structured`` or ``minimal`` for a planning user prompt.

    ``auto`` follows the checkpoint: structured-trained VLMs keep the
    annotation-style prompt; VQA-trained VLMs use the scene-only prompt
    because they never saw Spatial / Driving / Counterfactual as the user turn.
    """
    mode = str(planning_prompt or PLANNING_PROMPT_AUTO).lower().strip()
    if mode == PLANNING_PROMPT_AUTO:
        trained = str(reasoning_mode or "structured").lower().strip()
        return (
            PLANNING_PROMPT_MINIMAL
            if trained == "vqa"
            else PLANNING_PROMPT_STRUCTURED
        )
    if mode in (PLANNING_PROMPT_STRUCTURED, PLANNING_PROMPT_MINIMAL):
        return mode
    raise ValueError(
        f"Unknown planning_prompt {planning_prompt!r}; "
        f"expected one of {PLANNING_PROMPT_MODES}"
    )


def build_planning_user_prompt(
    vlm: Any,
    *,
    num_history_steps: int,
    num_current_cameras: int,
    mission_command: str,
    planning_prompt: str = PLANNING_PROMPT_AUTO,
    reasoning_mode: str = "structured",
    reasoning_format: Optional[str] = None,
) -> str:
    """Build the planning user prompt for a VLM backbone instance."""
    kind = resolve_planning_prompt_kind(planning_prompt, reasoning_mode)
    return vlm.build_multiview_prompt(
        num_history_steps=num_history_steps,
        num_current_cameras=num_current_cameras,
        mission_command=mission_command,
        include_reasoning_prefix=(kind == PLANNING_PROMPT_STRUCTURED),
        reasoning_format=reasoning_format,
    )


def add_planning_prompt_arguments(
    parser: argparse.ArgumentParser,
    *,
    hyphen: bool = False,
) -> None:
    """Add ``--planning_prompt`` / ``--planning_reasoning_format`` to a CLI."""
    prompt_flag = "--planning-prompt" if hyphen else "--planning_prompt"
    format_flag = "--planning-reasoning-format" if hyphen else "--planning_reasoning_format"
    parser.add_argument(
        prompt_flag,
        dest="planning_prompt",
        default=PLANNING_PROMPT_AUTO,
        choices=list(PLANNING_PROMPT_MODES),
        help=(
            "User prompt for planning inference. 'auto' uses the structured "
            "multi-view prompt when the checkpoint was trained with "
            "--reasoning_mode structured, and the scene-only prompt when it "
            "was trained with vqa. 'structured' always requests Spatial / "
            "Driving / Counterfactual sections. 'minimal' is cameras + mission "
            "only."
        ),
    )
    parser.add_argument(
        format_flag,
        dest="planning_reasoning_format",
        default=None,
        choices=list(REASONING_FORMATS),
        help=(
            "Override which structured sections appear in the planning prompt "
            "(default: the checkpoint's --reasoning_format)."
        ),
    )


def _load_metadata(clip_path: str) -> Dict[str, Any]:
    with open(os.path.join(clip_path, "metadata.json"), "r") as f:
        return json.load(f)


def load_vla_training_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Load ``training_config.json`` saved by ``nureasoning.nuvla.train``.

    The trainer writes the file next to checkpoints (workspace root). Older
    runs sometimes copied it into the checkpoint directory itself.
    """
    ckpt_dir = os.path.normpath(checkpoint_dir)
    parent_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(parent_dir, "training_config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(ckpt_dir, "training_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"nuVLA training_config.json not found in {parent_dir} or {ckpt_dir}"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return {**_TRAIN_DEFAULTS, **cfg}


def vlm_backbone_config_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword args for ``VLMBackboneConfig`` from a training config dict."""
    return {
        "model_name_or_path": cfg.get("vlm_model_path", "Qwen/Qwen3-VL-2B-Instruct"),
        "freeze_vision_encoder": True,
        "lora_rank": int(cfg.get("lora_rank", 0) or 0),
        "lora_alpha": int(cfg.get("lora_alpha", 0) or 0),
        "lora_dropout": float(cfg.get("lora_dropout", 0.0) or 0.0),
        "current_resolution": (
            int(cfg.get("current_res_w", 448)),
            int(cfg.get("current_res_h", 448)),
        ),
        "history_resolution": (
            int(cfg.get("history_res_w", 448)),
            int(cfg.get("history_res_h", 448)),
        ),
        "reasoning_format": cfg.get(
            "reasoning_format", "spatial_driving_counterfactual"
        ),
    }


def _load_frame_images(
    clip_path: str,
    frame: Dict[str, Any],
    resolution: Tuple[int, int],
    cam_names: Sequence[str],
) -> Dict[str, Any]:
    from PIL import Image

    cameras = (frame.get("sensors") or {}).get("cameras") or {}
    result: Dict[str, Any] = {}
    for cam in cam_names:
        rel = cameras.get(cam, "")
        if not rel:
            result[cam] = Image.new("RGB", resolution, (0, 0, 0))
            continue
        abs_path = os.path.join(clip_path, rel)
        try:
            with Image.open(abs_path) as raw:
                result[cam] = raw.convert("RGB").resize(resolution, Image.LANCZOS)
        except Exception:
            result[cam] = Image.new("RGB", resolution, (0, 0, 0))
    return result


def load_vlm_observation(
    clip_path: str,
    key_frame_idx: int,
    cfg: Optional[Dict[str, Any]] = None,
    cam_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Any], List[str], str]:
    """Load multi-view history+current images the same way ``VLATrainer`` does.

    Returns ``(images, image_contexts, mission_command)`` with time-major then
    camera-major order: ``[t=-H ... t=0] × CAMERA_NAMES``.
    """
    from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES

    cfg = {**_TRAIN_DEFAULTS, **(cfg or {})}
    names = list(cam_names or CAMERA_NAMES)
    meta = _load_metadata(clip_path)
    frames = meta.get("frames") or []
    if not frames or key_frame_idx < 0 or key_frame_idx >= len(frames):
        return [], [], "LANE_FOLLOW"

    num_hist = int(cfg.get("num_history_steps", 1) or 0)
    hist_stride = int(cfg.get("history_stride", 10) or 1)
    current_res = (
        int(cfg.get("current_res_w", 448)),
        int(cfg.get("current_res_h", 448)),
    )
    history_res = (
        int(cfg.get("history_res_w", 448)),
        int(cfg.get("history_res_h", 448)),
    )
    clip_hz = infer_clip_frame_rate_hz(
        meta, frames, float(cfg.get("frame_rate_hz", DEFAULT_FRAME_RATE_HZ)),
    )
    scaled_hist_stride = camera_history_stride(hist_stride, clip_hz)

    frame = frames[key_frame_idx]
    current_images = _load_frame_images(clip_path, frame, current_res, names)
    flat_images: List[Any] = []
    for step in range(num_hist, 0, -1):
        hist_idx = max(key_frame_idx - step * scaled_hist_stride, 0)
        hist_images = _load_frame_images(
            clip_path, frames[hist_idx], history_res, names
        )
        for cam in names:
            flat_images.append(hist_images.get(cam))
    for cam in names:
        flat_images.append(current_images.get(cam))
    flat_images = [img for img in flat_images if img is not None]

    image_contexts: List[str] = []
    for t in range(-num_hist, 1):
        t_label = f"t={t}" if t < 0 else "t=0 (current)"
        for cam in names:
            image_contexts.append(f"{t_label}, {cam} camera.")
    image_contexts = image_contexts[: len(flat_images)]

    mission = frame.get("mission_goal") or {}
    mission_cmd = (
        mission.get("command", "LANE_FOLLOW")
        if isinstance(mission, dict)
        else "LANE_FOLLOW"
    )
    return flat_images, image_contexts, str(mission_cmd or "LANE_FOLLOW")


class VLATrajectoryProvider:
    """
    Loads a trained VLA checkpoint and produces global-frame trajectories
    compatible with the benchmark's ``trajectory_provider`` callback.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        num_inference_steps: int = 5,
        device: str = "cuda",
        planning_prompt: str = PLANNING_PROMPT_AUTO,
        reasoning_format: Optional[str] = None,
    ):
        import torch
        from nureasoning.nuvla.models.action_expert import ActionExpertConfig, FlowMatchingDiTActionExpert
        from nureasoning.nuvla.models.vlm_backbone import CAMERA_NAMES as _CAM_NAMES
        from nureasoning.nuvla.models.vlm_backbone import VLMBackbone, VLMBackboneConfig

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = num_inference_steps
        self._cam_names = _CAM_NAMES
        self._torch = torch
        self.planning_prompt = planning_prompt or PLANNING_PROMPT_AUTO
        self.planning_reasoning_format = reasoning_format

        checkpoint_dir = os.path.normpath(checkpoint_dir)
        self.cfg = load_vla_training_config(checkpoint_dir)
        cfg = self.cfg
        vlm_config = VLMBackboneConfig(**vlm_backbone_config_kwargs(cfg))
        self.vlm = VLMBackbone(vlm_config).to(self.device)
        self.vlm.eval()
        self.planning_prompt_kind = resolve_planning_prompt_kind(
            self.planning_prompt,
            cfg.get("reasoning_mode", "structured"),
        )

        vlm_adapter_dir = os.path.join(checkpoint_dir, "vlm_adapter")
        if os.path.isdir(vlm_adapter_dir):
            self.vlm.load_adapter(vlm_adapter_dir, device=self.device)
            logger.info("VLA: loaded VLM adapter from %s", vlm_adapter_dir)
        elif int(cfg.get("lora_rank", 0) or 0) > 0:
            raise FileNotFoundError(
                f"nuVLA checkpoint requires a LoRA adapter, but "
                f"{vlm_adapter_dir} does not exist"
            )
        else:
            logger.info("VLA: checkpoint uses no VLM adapter")

        action_config = ActionExpertConfig(
            vlm_feature_dim=self.vlm.get_feature_dim(),
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
            noise_beta_alpha=cfg.get("noise_beta_alpha", 2.5),
            noise_beta_beta=cfg.get("noise_beta_beta", 1.5),
        )
        self.action_expert = FlowMatchingDiTActionExpert(action_config).to(self.device)
        self.action_expert.eval()

        action_path = os.path.join(checkpoint_dir, "action_expert.pt")
        if os.path.isfile(action_path):
            state = torch.load(action_path, map_location=self.device)
            self.action_expert.load_state_dict(state)
            logger.info("VLA: loaded action expert from %s", action_path)
        else:
            raise FileNotFoundError(
                f"nuVLA action expert checkpoint not found: {action_path}"
            )

        logger.info(
            "VLA: ready (feature_dim=%d, inference_steps=%d, "
            "planning_prompt=%s -> %s, reasoning_format=%s)",
            self.vlm.get_feature_dim(),
            self.num_inference_steps,
            self.planning_prompt,
            self.planning_prompt_kind,
            self.planning_reasoning_format
            or self.vlm.config.reasoning_format,
        )

    def build_planning_prompt(
        self,
        num_history_steps: int,
        num_current_cameras: int,
        mission_command: str,
    ) -> str:
        return build_planning_user_prompt(
            self.vlm,
            num_history_steps=num_history_steps,
            num_current_cameras=num_current_cameras,
            mission_command=mission_command,
            planning_prompt=self.planning_prompt,
            reasoning_mode=str(self.cfg.get("reasoning_mode") or "structured"),
            reasoning_format=self.planning_reasoning_format,
        )

    def _load_frame_images(
        self, clip_path: str, frame: Dict, resolution: Tuple[int, int]
    ) -> Dict[str, Any]:
        return _load_frame_images(clip_path, frame, resolution, self._cam_names)

    @staticmethod
    def _ego_to_global(
        ego_traj: np.ndarray,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
    ) -> np.ndarray:
        """Convert [N, 3] ego-frame (dx, dy, dtheta) waypoints to global (x, y, yaw)."""
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        out = np.zeros_like(ego_traj)
        for i in range(len(ego_traj)):
            dx, dy, dtheta = ego_traj[i]
            out[i, 0] = ego_x + dx * cos_y - dy * sin_y
            out[i, 1] = ego_y + dx * sin_y + dy * cos_y
            out[i, 2] = ego_yaw + dtheta
        return out

    @staticmethod
    def _global_points_to_ego(
        global_points: list,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
    ) -> np.ndarray:
        cos_neg = math.cos(-ego_yaw)
        sin_neg = math.sin(-ego_yaw)
        ego_points = []
        for pt in global_points:
            if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 3:
                ego_points.append([0.0, 0.0, 0.0])
                continue
            gx, gy, gyaw = float(pt[0]), float(pt[1]), float(pt[2])
            dx_g, dy_g = gx - ego_x, gy - ego_y
            ego_points.append([
                dx_g * cos_neg - dy_g * sin_neg,
                dx_g * sin_neg + dy_g * cos_neg,
                ((gyaw - ego_yaw) + math.pi) % (2 * math.pi) - math.pi,
            ])
        return np.asarray(ego_points, dtype=np.float32)

    @classmethod
    def _build_history_trajectory(
        cls,
        hist_raw: Any,
        max_hist_pts: int,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        *,
        frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
        waypoint_interval_s: float = 0.5,
    ) -> np.ndarray:
        """Mirror NuReasoningVLADataset history padding and ego-frame conversion."""
        zeros = np.zeros((max_hist_pts, 3), dtype=np.float32)
        traj_history = list(hist_raw or [])
        if not traj_history:
            return zeros

        stride = frame_stride_for_dt(waypoint_interval_s, frame_rate_hz)
        if stride > 1:
            traj_history = traj_history[::stride]

        if len(traj_history) > max_hist_pts:
            traj_history = traj_history[-max_hist_pts:]
        elif len(traj_history) < max_hist_pts and len(traj_history) >= 2:
            while len(traj_history) < max_hist_pts:
                p_first = traj_history[0]
                p_second = traj_history[1]
                dx = float(p_first[0]) - float(p_second[0])
                dy = float(p_first[1]) - float(p_second[1])
                dyaw = float(p_first[2]) - float(p_second[2])
                dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
                traj_history.insert(0, [
                    float(p_first[0]) + dx,
                    float(p_first[1]) + dy,
                    float(p_first[2]) + dyaw,
                ])

        ego_history = cls._global_points_to_ego(traj_history, ego_x, ego_y, ego_yaw)
        if len(ego_history) >= max_hist_pts:
            return ego_history[:max_hist_pts]

        zeros[: len(ego_history)] = ego_history
        return zeros

    @staticmethod
    def _smooth_ego_trajectory(
        ego_traj: np.ndarray,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        """Upsample sparse model waypoints to a smooth ego-frame trajectory."""
        traj = np.asarray(ego_traj, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 3 or len(traj) == 0:
            return traj

        horizon_s = float(cfg.get("trajectory_future_seconds", 5.0))
        smooth_dt_s = float(cfg.get("trajectory_smoothing_dt_s", 0.1))
        if horizon_s <= 0.0 or smooth_dt_s <= 0.0:
            return traj[:, :3]

        source_times = np.arange(1, len(traj) + 1, dtype=np.float64) * (horizon_s / len(traj))
        source_times = np.concatenate(([0.0], source_times))
        source_traj = np.vstack((np.zeros((1, 3), dtype=np.float64), traj[:, :3]))
        source_traj[:, 2] = np.unwrap(source_traj[:, 2])

        num_steps = int(round(horizon_s / smooth_dt_s)) + 1
        target_times = np.linspace(0.0, horizon_s, max(num_steps, 2), dtype=np.float64)

        if len(source_times) < 4:
            smoothed = np.column_stack([
                np.interp(target_times, source_times, source_traj[:, dim])
                for dim in range(3)
            ])
        else:
            from scipy.interpolate import CubicSpline

            smoothed = np.column_stack([
                CubicSpline(source_times, source_traj[:, dim], bc_type="natural")(target_times)
                for dim in range(3)
            ])

        smoothed[:, 2] = (smoothed[:, 2] + math.pi) % (2 * math.pi) - math.pi
        smoothed[0] = 0.0
        return smoothed

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        torch = self._torch
        cfg = self.cfg
        meta = _load_metadata(clip_path)
        frames = meta.get("frames", [])
        if key_frame_idx >= len(frames):
            return None

        num_hist = int(cfg.get("num_history_steps", 1) or 0)
        clip_hz = infer_clip_frame_rate_hz(
            meta, frames, float(cfg.get("frame_rate_hz", DEFAULT_FRAME_RATE_HZ)),
        )
        flat_images, image_contexts, mission_cmd = load_vlm_observation(
            clip_path, key_frame_idx, cfg, cam_names=self._cam_names
        )
        if not flat_images:
            return None
        prompt = self.build_planning_prompt(
            num_history_steps=num_hist,
            num_current_cameras=len(self._cam_names),
            mission_command=mission_cmd,
        )

        inputs = self.vlm.prepare_inputs(
            images=flat_images,
            text_prompt=prompt,
            image_contexts=image_contexts,
        )
        vlm_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "prompt_length" in vlm_inputs:
            vlm_inputs["prompt_lengths"] = vlm_inputs.pop("prompt_length").to(self.device)

        with torch.no_grad():
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
        action_dtype = next(self.action_expert.parameters()).dtype
        vlm_features = vlm_outputs["vlm_features"].to(dtype=action_dtype)

        def state_field(name: str) -> Any:
            if isinstance(ego_state, dict):
                return ego_state.get(name)
            return getattr(ego_state, name, None)

        vel_raw = state_field("velocity")
        acc_raw = state_field("acceleration")
        vel = vel_raw if isinstance(vel_raw, dict) else {}
        acc = acc_raw if isinstance(acc_raw, dict) else {}
        ego_dyn = torch.tensor(
            [[
                float(vel.get("vx", 0)),
                float(vel.get("vy", 0)),
                float(acc.get("ax", 0)),
                float(acc.get("ay", 0)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        max_hist_pts = cfg.get("max_history_traj_points", 6)
        hist_raw = state_field("trajectory_history") or []
        pose_raw = state_field("pose")
        pose = pose_raw if isinstance(pose_raw, dict) else {}
        ego_x = float(pose.get("x", 0))
        ego_y = float(pose.get("y", 0))
        ego_yaw = float(pose.get("yaw", 0))

        hist_pts = self._build_history_trajectory(
            hist_raw,
            max_hist_pts,
            ego_x,
            ego_y,
            ego_yaw,
            frame_rate_hz=clip_hz,
            waypoint_interval_s=waypoint_dt_s(
                float(cfg.get("trajectory_future_seconds", 5.0)),
                int(cfg.get("num_trajectory_points", 10) or 10),
            ),
        )
        ego_hist_traj = torch.tensor(
            np.array([hist_pts]),
            dtype=torch.float32,
            device=self.device,
        )

        use_cuda_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=torch.bfloat16):
            pred_traj = self.action_expert.sample(
                vlm_features,
                ego_dyn,
                ego_hist_traj,
                num_steps=self.num_inference_steps,
            )

        ego_traj_np = pred_traj[0].float().cpu().numpy()
        ego_traj_np = self._smooth_ego_trajectory(ego_traj_np, cfg)
        expected_steps = int(round(5.0 / 0.1)) + 1
        if ego_traj_np.shape != (expected_steps, 3):
            raise ValueError(
                f"nuVLA produced trajectory shape {ego_traj_np.shape}; "
                f"expected ({expected_steps}, 3)"
            )
        if not np.all(np.isfinite(ego_traj_np)):
            raise ValueError("nuVLA produced non-finite trajectory values")

        return self._ego_to_global(ego_traj_np, ego_x, ego_y, ego_yaw)

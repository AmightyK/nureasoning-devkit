"""
NuReasoning VLA Dataset & DataLoader.

Loads multi-view, multi-frame camera images at differentiated resolutions
alongside structured reasoning annotations and ego trajectories for VLA
training.

Data layout (per clip):
  clip_dir/
    metadata.json          — clip-level metadata with per-frame entries
    cameras/{CAM}/...jpg   — camera images
    ego_state/{ts}.pkl     — ego pose / velocity / trajectory
    reasoning/{ts}.json    — structured Spatial / Driving / Counterfactual
    map.pkl                — static map (optional)
"""

import json
import logging
import math
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

CAMERA_NAMES = [
    "front", "front_left", "front_right",
    "left", "right",
    "back", "back_left", "back_right",
]

# Official / HuggingFace / challenge clips are 10 Hz. ``history_stride`` is
# specified in frames at this rate (10 ⇒ 1 s). The action expert still uses
# 10 waypoints over 5 s (0.5 s spacing).
DEFAULT_FRAME_RATE_HZ = 10.0


def infer_clip_frame_rate_hz(
    metadata: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    fallback_hz: float = DEFAULT_FRAME_RATE_HZ,
) -> float:
    """Read clip fps from metadata, else from frame timestamps, else ``fallback_hz``."""
    frame_rate = metadata.get("frame_rate_hz")
    if isinstance(frame_rate, (int, float)) and frame_rate > 0:
        return float(frame_rate)

    for i in range(1, len(frames)):
        t0 = frames[i - 1].get("relative_time_s")
        t1 = frames[i].get("relative_time_s")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0:
            return 1.0 / (float(t1) - float(t0))
    return float(fallback_hz)


def frame_stride_for_dt(dt_s: float, frame_rate_hz: float) -> int:
    """Convert a time interval to a clip-frame stride (10 Hz × 0.5 s → 5)."""
    if dt_s <= 0.0 or frame_rate_hz <= 0.0:
        return 1
    return max(1, int(round(dt_s * frame_rate_hz)))


def waypoint_dt_s(horizon_s: float, num_waypoints: int) -> float:
    """Seconds between action-expert waypoints (default 5 s / 10 pts = 0.5 s)."""
    return float(horizon_s) / max(int(num_waypoints), 1)


def camera_history_stride(history_stride: int, clip_frame_rate_hz: float) -> int:
    """``history_stride`` is in 10 Hz frames; return the matching clip-frame stride."""
    return frame_stride_for_dt(
        float(history_stride) / DEFAULT_FRAME_RATE_HZ,
        clip_frame_rate_hz,
    )


class _EgoStateUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles the ``data_schema.EgoState`` class which
    may not be importable in the training environment.  We replace it with
    a lightweight SimpleNamespace-like object so that attribute access works
    as expected.
    """

    def find_class(self, module: str, name: str):
        if module.startswith("data_schema"):
            return type(name, (), {})
        return super().find_class(module, name)


def _load_pickle_safe(path: str) -> Any:
    """Load a pickle file, falling back to the custom unpickler when needed."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, "rb") as f:
            return _EgoStateUnpickler(f).load()


@dataclass
class VLADataConfig:
    data_root: str = "./dataset/data/train"
    num_history_steps: int = 1
    history_stride: int = 10  # frames at DEFAULT_FRAME_RATE_HZ (10 ⇒ 1 s)
    current_resolution: Tuple[int, int] = (448, 448)
    history_resolution: Tuple[int, int] = (224, 224)
    trajectory_future_seconds: float = 5.0
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ
    num_waypoints: int = 10  # 5 s trajectory, 0.5 s spacing
    max_history_traj_points: int = 6  # 3 s of history on the 0.5 s grid
    cameras: List[str] = field(default_factory=lambda: CAMERA_NAMES)
    reasoning_format: str = "spatial_driving_counterfactual"
    reasoning_max_items_per_list: int = 10
    reasoning_mode: str = "structured"  # "structured" | "vqa"
    vqa_root: Optional[str] = None
    qa_seed: int = 42


class NuReasoningVLADataset(Dataset):
    """
    PyTorch Dataset for nuReasoning VLA training.

    Each sample provides:
      - Multi-view images at current time (high-res) and history (low-res)
      - A reasoning text target (structured annotations, or VQA Q&A)
      - 5-second future ego trajectory as waypoints (x, y, θ) in ego frame

    ``reasoning_mode``:
      - ``structured``: Spatial / Driving / Counterfactual annotation text
      - ``vqa``: generated VQA under ``vqa_root`` (all question types)
    """

    def __init__(self, config: VLADataConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        self._vqa_index: Dict[Tuple[str, str], str] = {}
        self._discover_samples()
        self._index_vqa()
        logger.info(
            "[%s] Reasoning mode: %s",
            split,
            str(self.config.reasoning_mode).lower().strip() or "structured",
        )

    # ------------------------------------------------------------------
    # Reasoning format helpers
    # ------------------------------------------------------------------

    def _include_spatial(self) -> bool:
        fmt = str(self.config.reasoning_format).lower().strip()
        return fmt in {"spatial", "spatial_driving", "spatial_driving_counterfactual", "all", "full"}

    def _include_counterfactual(self) -> bool:
        fmt = str(self.config.reasoning_format).lower().strip()
        return fmt in {"spatial_driving_counterfactual", "driving_counterfactual", "all", "full"}

    def _use_vqa_targets(self) -> bool:
        mode = str(self.config.reasoning_mode).lower().strip()
        return mode == "vqa"

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------

    def _discover_samples(self):
        data_root = self.config.data_root
        if not os.path.isdir(data_root):
            logger.warning(f"Data root not found: {data_root}")
            return

        from nureasoning.common.clips import discover_clips

        clip_dirs = discover_clips(data_root)

        for clip_dir in clip_dirs:
            try:
                self._index_clip(clip_dir)
            except Exception as e:
                logger.warning(f"Failed to index clip {clip_dir}: {e}")

        logger.info(
            f"[{self.split}] Discovered {len(self.samples)} samples "
            f"from {len(clip_dirs)} clips"
        )

    def _index_clip(self, clip_dir: str):
        with open(os.path.join(clip_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        frames = metadata.get("frames", [])
        clip_name = os.path.basename(clip_dir)

        clip_frame_rate_hz = infer_clip_frame_rate_hz(
            metadata, frames, self.config.frame_rate_hz,
        )
        hist_stride = camera_history_stride(
            self.config.history_stride, clip_frame_rate_hz,
        )
        min_history_frames = self.config.num_history_steps * hist_stride

        for frame in frames:
            reasoning_relpath = frame.get("reasoning", "")
            if not reasoning_relpath:
                continue

            reasoning_abs = os.path.join(clip_dir, reasoning_relpath)
            if not os.path.isfile(reasoning_abs):
                continue

            frame_idx = frame.get("frame_index", -1)
            if frame_idx < min_history_frames:
                continue

            ego_state_relpath = frame.get("ego_state", "")
            if not ego_state_relpath:
                continue

            self.samples.append({
                "clip_dir": clip_dir,
                "clip_name": clip_name,
                "frame_index": frame_idx,
                "timestamp_us": frame.get("timestamp_us", 0),
                "reasoning_file": reasoning_abs,
                "metadata": metadata,
                "frames": frames,
            })

    def _index_vqa(self) -> None:
        """Map (clip_name, timestamp) to generated VQA files under ``vqa_root``."""
        if not self._use_vqa_targets():
            return
        root = self.config.vqa_root
        if not root or not os.path.isdir(root):
            if root:
                logger.warning("vqa_root is set but does not exist: %s", root)
            return
        count = 0
        for dirpath, _, filenames in os.walk(root):
            clip_name = os.path.basename(dirpath)
            for name in filenames:
                if not name.endswith("_vqa.json"):
                    continue
                timestamp = name[: -len("_vqa.json")]
                self._vqa_index[(clip_name, timestamp)] = os.path.join(dirpath, name)
                count += 1
        logger.info(
            "[%s] Indexed %d VQA files under %s",
            self.split, count, root,
        )

    def _select_qa_target(
        self, clip_name: str, timestamp_us: int,
    ) -> Optional[Tuple[str, str]]:
        """Use a VQA question/answer as the text target in VQA reasoning mode.

        Samples uniformly across question types (choice, numerical, text, …)
        so spatial multiple-choice does not crowd out other answer formats.
        """
        if not self._use_vqa_targets() or not self._vqa_index:
            return None
        path = self._vqa_index.get((clip_name, str(timestamp_us)))
        if not path:
            return None
        from nureasoning.reasoning.modules.prompt_format import (
            format_assistant_answer,
            format_question_prompt,
        )
        from nureasoning.reasoning.modules.sampling import frame_rng

        rng = frame_rng(self.config.qa_seed, clip_name, str(timestamp_us))
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        by_type: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
        for question in payload.get("questions") or []:
            if not isinstance(question, dict):
                continue
            answer = format_assistant_answer(question)
            if not answer:
                continue
            qtype = str(question.get("question_type") or "unknown").lower()
            by_type.setdefault(qtype, []).append((question, answer))
        if not by_type:
            return None
        qtype = rng.choice(sorted(by_type))
        question, answer = rng.choice(by_type[qtype])
        return format_question_prompt(question), answer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]
        clip_dir = info["clip_dir"]
        frame_idx = info["frame_index"]
        frames = info["frames"]
        clip_frame_rate_hz = infer_clip_frame_rate_hz(
            info.get("metadata", {}),
            frames,
            self.config.frame_rate_hz,
        )
        hist_stride = camera_history_stride(
            self.config.history_stride, clip_frame_rate_hz,
        )

        current_frame = frames[frame_idx]

        # --- images ---
        current_images = self._load_frame_images(
            clip_dir, current_frame, self.config.current_resolution,
        )
        history_images: List[Dict[str, Image.Image]] = []
        history_times: List[int] = []
        for step in range(self.config.num_history_steps, 0, -1):
            hist_idx = max(frame_idx - step * hist_stride, 0)
            hist_frame = frames[hist_idx]
            imgs = self._load_frame_images(
                clip_dir, hist_frame, self.config.history_resolution,
            )
            history_images.append(imgs)
            history_times.append(-step)

        # --- reasoning ---
        with open(info["reasoning_file"], "r") as f:
            reasoning_data = json.load(f)

        spatial, driving, counterfactual = self._extract_reasoning_sections(reasoning_data)
        reasoning_text = self._format_reasoning_text(
            spatial=spatial, driving=driving, counterfactual=counterfactual,
        )
        driving_decision = self._extract_driving_decision(driving)
        scene_description = self._extract_scene_description(driving)
        user_prompt = None
        qa_pair = self._select_qa_target(info["clip_name"], info["timestamp_us"])
        if qa_pair is not None:
            user_prompt, reasoning_text = qa_pair

        # --- ego state & trajectory ---
        ego_raw = self._load_ego_state_raw(clip_dir, current_frame)
        ego_traj = self._extract_ego_trajectory_from_raw(
            ego_raw, clip_dir, frames, frame_idx, clip_frame_rate_hz,
        )
        ego_hist_traj = self._extract_ego_history_trajectory_from_raw(
            ego_raw,
            clip_frame_rate_hz,
        )
        ego_vel, ego_acc = self._ego_dynamics_from_raw(ego_raw)

        # --- mission ---
        mission = current_frame.get("mission_goal", {})
        mission_command = mission.get("command", "LANE_FOLLOW") if mission else "LANE_FOLLOW"

        # --- flatten images: [hist_t-N cam0..cam7, ..., current cam0..cam7] ---
        flat_images: List[Optional[Image.Image]] = []
        for hist_dict in history_images:
            for cam in self.config.cameras:
                flat_images.append(hist_dict.get(cam))
        for cam in self.config.cameras:
            flat_images.append(current_images.get(cam))

        return {
            "clip_name": info["clip_name"],
            "frame_index": frame_idx,
            "timestamp_us": info["timestamp_us"],
            "images": flat_images,
            "history_times": history_times,
            "reasoning_text": reasoning_text,
            "user_prompt": user_prompt,
            "driving_decision": driving_decision,
            "scene_description": scene_description,
            "spatial_reasoning": spatial,
            "counterfactual_reasoning": counterfactual,
            "ego_trajectory": torch.tensor(ego_traj, dtype=torch.float32),
            "ego_history_trajectory": torch.tensor(ego_hist_traj, dtype=torch.float32),
            "ego_velocity": torch.tensor(ego_vel, dtype=torch.float32),
            "ego_acceleration": torch.tensor(ego_acc, dtype=torch.float32),
            "mission_command": mission_command,
        }

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_frame_images(
        self,
        clip_dir: str,
        frame: Dict[str, Any],
        target_resolution: Tuple[int, int],
    ) -> Dict[str, Image.Image]:
        cameras = frame.get("sensors", {}).get("cameras", {})
        result: Dict[str, Image.Image] = {}
        for cam_name in self.config.cameras:
            rel_path = cameras.get(cam_name, "")
            if not rel_path:
                result[cam_name] = Image.new("RGB", target_resolution, (0, 0, 0))
                continue
            abs_path = os.path.join(clip_dir, rel_path)
            try:
                img = Image.open(abs_path).convert("RGB")
                img = img.resize(target_resolution, Image.LANCZOS)
                result[cam_name] = img
            except Exception as e:
                logger.debug(f"Failed to load {abs_path}: {e}")
                result[cam_name] = Image.new("RGB", target_resolution, (0, 0, 0))
        return result

    # ------------------------------------------------------------------
    # Ego state / trajectory
    # ------------------------------------------------------------------

    def _load_ego_state_raw(
        self, clip_dir: str, frame: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Load ego state pickle into a plain dict."""
        ego_relpath = frame.get("ego_state", "")
        if not ego_relpath:
            return None
        abs_path = os.path.join(clip_dir, ego_relpath)
        if not os.path.isfile(abs_path):
            return None
        try:
            obj = _load_pickle_safe(abs_path)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            if isinstance(obj, dict):
                return obj
            return None
        except Exception as e:
            logger.debug(f"Failed to load ego state {abs_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Global → ego-frame conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_from_pose(pose: Dict[str, Any]) -> float:
        if "yaw" in pose:
            return float(pose["yaw"])
        qw = float(pose.get("qw", 1.0))
        qx = float(pose.get("qx", 0.0))
        qy = float(pose.get("qy", 0.0))
        qz = float(pose.get("qz", 0.0))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _global_traj_to_ego(
        self,
        raw_points: List,
        current_x: float,
        current_y: float,
        current_yaw: float,
        num_pts: int,
    ) -> np.ndarray:
        """Convert a list of [x_global, y_global, yaw] to [num_pts, 3] in ego frame."""
        cos_yaw = math.cos(-current_yaw)
        sin_yaw = math.sin(-current_yaw)

        points: List[np.ndarray] = []
        for pt in raw_points:
            fx, fy, fyaw = float(pt[0]), float(pt[1]), float(pt[2])
            dx_global = fx - current_x
            dy_global = fy - current_y
            dx_ego = dx_global * cos_yaw - dy_global * sin_yaw
            dy_ego = dx_global * sin_yaw + dy_global * cos_yaw

            dtheta = fyaw - current_yaw
            dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

            points.append(np.array([dx_ego, dy_ego, dtheta], dtype=np.float32))

        if not points:
            return np.zeros((num_pts, 3), dtype=np.float32)

        stacked = np.stack(points, axis=0)
        if len(stacked) >= num_pts:
            return stacked[:num_pts]

        padded = np.zeros((num_pts, 3), dtype=np.float32)
        padded[: len(stacked)] = stacked
        return padded

    def _ego_pose_components(
        self, ego: Optional[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Return (x, y, yaw) from the raw ego dict; zeros on failure."""
        if ego is None:
            return 0.0, 0.0, 0.0
        pose = ego.get("pose", {})
        if not isinstance(pose, dict):
            return 0.0, 0.0, 0.0
        return (
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            self._yaw_from_pose(pose),
        )

    # ------------------------------------------------------------------
    # Trajectory extraction (from pre-loaded ego dict)
    # ------------------------------------------------------------------

    def _extract_ego_trajectory_from_raw(
        self,
        ego: Optional[Dict[str, Any]],
        clip_dir: str,
        frames: List[Dict[str, Any]],
        frame_idx: int,
        frame_rate_hz: float,
    ) -> np.ndarray:
        """Build future trajectory [num_points, 3] in ego frame from actual
        ego poses at subsequent timesteps.  This is more reliable than the
        pre-computed ``trajectory_future`` stored in each ego-state pickle."""
        num_pts = self.config.num_waypoints
        zeros = np.zeros((num_pts, 3), dtype=np.float32)

        cx, cy, cyaw = self._ego_pose_components(ego)

        future_global: List[List[float]] = []
        future_stride = frame_stride_for_dt(
            waypoint_dt_s(self.config.trajectory_future_seconds, self.config.num_waypoints),
            frame_rate_hz,
        )
        for k in range(1, num_pts + 1):
            fwd_idx = frame_idx + k * future_stride
            if fwd_idx >= len(frames):
                break
            future_ego = self._load_ego_state_raw(clip_dir, frames[fwd_idx])
            if future_ego is None:
                break
            fx, fy, fyaw = self._ego_pose_components(future_ego)
            future_global.append([fx, fy, fyaw])

        if not future_global:
            return zeros

        # Pad by linear extrapolation from the last two points when the clip
        # is too short to provide all requested future poses.
        while len(future_global) < num_pts and len(future_global) >= 2:
            p_prev = future_global[-2]
            p_last = future_global[-1]
            dx = p_last[0] - p_prev[0]
            dy = p_last[1] - p_prev[1]
            dyaw = p_last[2] - p_prev[2]
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
            future_global.append([
                p_last[0] + dx,
                p_last[1] + dy,
                p_last[2] + dyaw,
            ])

        return self._global_traj_to_ego(future_global, cx, cy, cyaw, num_pts)

    def _extract_ego_history_trajectory_from_raw(
        self,
        ego: Optional[Dict[str, Any]],
        frame_rate_hz: float,
    ) -> np.ndarray:
        """History trajectory [max_history_traj_points, 3] in ego frame, zero-padded."""
        num_pts = self.config.max_history_traj_points
        zeros = np.zeros((num_pts, 3), dtype=np.float32)
        if ego is None:
            return zeros

        traj_history = list(ego.get("trajectory_history", []))
        if not traj_history:
            return zeros

        pose_stride = frame_stride_for_dt(
            waypoint_dt_s(self.config.trajectory_future_seconds, self.config.num_waypoints),
            frame_rate_hz,
        )
        if pose_stride > 1:
            traj_history = traj_history[::pose_stride]

        n_raw = len(traj_history)
        if n_raw > num_pts:
            traj_history = traj_history[-num_pts:]
        elif n_raw < num_pts and n_raw >= 2:
            while len(traj_history) < num_pts:
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

        cx, cy, cyaw = self._ego_pose_components(ego)
        return self._global_traj_to_ego(traj_history, cx, cy, cyaw, num_pts)

    def _ego_dynamics_from_raw(
        self, ego: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (velocity[2], acceleration[2]) from pre-loaded ego dict."""
        if ego is None:
            return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        vel = ego.get("velocity", {})
        acc = ego.get("acceleration", {})
        if not isinstance(vel, dict):
            vel = {}
        if not isinstance(acc, dict):
            acc = {}
        velocity = np.array([
            float(vel.get("vx", 0.0)),
            float(vel.get("vy", 0.0)),
        ], dtype=np.float32)
        acceleration = np.array([
            float(acc.get("ax", 0.0)),
            float(acc.get("ay", 0.0)),
        ], dtype=np.float32)
        return velocity, acceleration

    # ------------------------------------------------------------------
    # Reasoning extraction & formatting
    # ------------------------------------------------------------------

    def _canonicalize_key(self, key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(key).lower())

    def _get_with_aliases(
        self, mapping: Dict[str, Any], aliases: List[str], default: Any = None
    ) -> Any:
        if not isinstance(mapping, dict):
            return default
        normalized = {self._canonicalize_key(k): v for k, v in mapping.items()}
        for alias in aliases:
            v = normalized.get(self._canonicalize_key(alias))
            if v is not None:
                return v
        return default

    def _extract_reasoning_sections(
        self, reasoning_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if not isinstance(reasoning_data, dict):
            return {}, {}, {}

        spatial = self._get_with_aliases(
            reasoning_data,
            aliases=["Spatial", "spatial", "environment", "scene_spatial"],
            default={},
        )
        driving = self._get_with_aliases(
            reasoning_data,
            aliases=["Driving", "driving", "planning", "decision"],
            default={},
        )
        counterfactual = self._get_with_aliases(
            reasoning_data,
            aliases=["Counterfactual", "counterfactual", "counter_factual", "what_if"],
            default={},
        )
        return (
            spatial if isinstance(spatial, dict) else {},
            driving if isinstance(driving, dict) else {},
            counterfactual if isinstance(counterfactual, dict) else {},
        )

    def _extract_driving_decision(self, driving: Dict[str, Any]) -> Dict[str, str]:
        decision = self._get_with_aliases(
            driving,
            aliases=["Driving decision", "driving_decision", "decision"],
            default={},
        )
        if not isinstance(decision, dict):
            return {}
        lon = self._get_with_aliases(
            decision,
            aliases=["Longitudinal", "longitudinal", "speed", "throttle_brake"],
            default="",
        )
        lat = self._get_with_aliases(
            decision,
            aliases=["Lateral", "lateral", "steering", "direction"],
            default="",
        )
        return {
            "Longitudinal": str(lon) if lon is not None else "",
            "Lateral": str(lat) if lat is not None else "",
        }

    def _extract_scene_description(self, driving: Dict[str, Any]) -> str:
        scene = self._get_with_aliases(
            driving,
            aliases=["Scene description", "scene_description", "scene", "summary"],
            default="",
        )
        return str(scene) if scene is not None else ""

    def _format_spatial_section(self, spatial: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["[Spatial]"]
        summary = self._get_with_aliases(
            spatial, aliases=["summary", "scene_summary", "description"], default=None,
        )
        if isinstance(summary, str) and summary.strip():
            lines.append(f"Summary: {summary.strip()}")
        elif isinstance(summary, dict):
            compact = ", ".join(f"{k}: {v}" for k, v in summary.items())
            if compact:
                lines.append(f"Summary: {compact}")

        obj_rel = self._get_with_aliases(
            spatial, aliases=["object_relations", "objects", "agents", "detections"], default=[],
        )
        limit = max(1, int(self.config.reasoning_max_items_per_list))
        if isinstance(obj_rel, list) and obj_rel:
            def _distance_to_ego(rel: Dict[str, Any]) -> float:
                geo = rel.get("geometric_relations", {})
                if isinstance(geo, dict):
                    dist = geo.get("euclidean_distance_m")
                    if isinstance(dist, (float, int)):
                        return float(dist)

                pos3d = rel.get("position_3d_ego", {})
                if isinstance(pos3d, dict):
                    coords = [
                        float(coord)
                        for coord in (pos3d.get("x"), pos3d.get("y"), pos3d.get("z"))
                        if isinstance(coord, (float, int))
                    ]
                    if len(coords) >= 2:
                        return math.sqrt(sum(coord ** 2 for coord in coords))

                return float("inf")

            sorted_obj_rel = sorted(
                (rel for rel in obj_rel if isinstance(rel, dict)),
                key=_distance_to_ego,
            )
            for idx, rel in enumerate(sorted_obj_rel[:limit]):
                cat = rel.get("category", "unknown")
                lines.append(f"- object[{idx}] category={cat}")

                pos3d = rel.get("position_3d_ego", {})
                if isinstance(pos3d, dict):
                    x = pos3d.get("x")
                    y = pos3d.get("y")
                    z = pos3d.get("z")
                    if all(isinstance(v, (float, int)) for v in (x, y, z)):
                        lines.append(f"  3d_center_ego=({x:.2f}, {y:.2f}, {z:.2f})")

                camera_observations = rel.get("camera_observations", {})
                if isinstance(camera_observations, dict) and camera_observations:
                    view_parts: List[str] = []
                    for view_name, view_obs in list(camera_observations.items())[:8]:
                        if not isinstance(view_obs, dict):
                            continue
                        bbox2d = view_obs.get("detection_bbox_2d")
                        if isinstance(bbox2d, list) and len(bbox2d) >= 4:
                            x1, y1, x2, y2 = bbox2d[:4]
                            if all(isinstance(v, (float, int)) for v in (x1, y1, x2, y2)):
                                view_parts.append(
                                    f"{view_name}=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                                )
                    if view_parts:
                        lines.append(f"  2d_boxes_by_view: {'; '.join(view_parts)}")

                geo = rel.get("geometric_relations", {})
                if not isinstance(geo, dict):
                    geo = {}
                dist = geo.get("euclidean_distance_m")
                if isinstance(dist, (float, int)):
                    lines.append(f"  distance_m={dist:.2f}")

                fc = rel.get("future_conflict", {})
                if isinstance(fc, dict):
                    conflict = bool(fc.get("conflict_with_ego", False))
                    ttc = fc.get("ttc_s")
                    min_dist = fc.get("min_center_distance_m")
                    conflict_parts = [f"conflict_with_ego={conflict}"]
                    if isinstance(ttc, (float, int)):
                        conflict_parts.append(f"ttc_s={ttc:.2f}")
                    if isinstance(min_dist, (float, int)):
                        conflict_parts.append(f"min_center_distance_m={min_dist:.2f}")
                    lines.append(f"  future_conflict: {', '.join(conflict_parts)}")

                future_traj = rel.get("future_trajectory_3d_ego", [])
                if isinstance(future_traj, list) and future_traj:
                    traj_points: List[str] = []
                    for pt in future_traj[:5]:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                            px, py, pz = pt[:3]
                            if all(isinstance(v, (float, int)) for v in (px, py, pz)):
                                traj_points.append(f"({px:.2f},{py:.2f},{pz:.2f})")
                    if traj_points:
                        lines.append(
                            f"  future_trajectory_3d_ego(first_5): {' -> '.join(traj_points)}"
                        )

        # map_info = self._get_with_aliases(
        #     spatial, aliases=["map", "map_data", "hd_map"], default={},
        # )
        # if isinstance(map_info, dict) and map_info:
        #     lines.append("Map context available")
        return lines

    def _format_driving_section(self, driving: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["[Driving]"]
        scene = self._extract_scene_description(driving)
        if scene:
            lines.append(f"Scene description: {scene}")

        critical = self._get_with_aliases(
            driving,
            aliases=["Critical components", "critical_components", "critical_objects"],
            default={},
        )
        limit = max(1, int(self.config.reasoning_max_items_per_list))
        if isinstance(critical, dict) and critical:
            lines.append("Critical components:")
            for name, comp in list(critical.items())[:limit]:
                if isinstance(comp, dict):
                    comp_str = ", ".join(f"{k}: {v}" for k, v in comp.items())
                    lines.append(f"- {name}: {comp_str}")
                else:
                    lines.append(f"- {name}: {comp}")

        decision = self._extract_driving_decision(driving)
        if decision.get("Longitudinal") or decision.get("Lateral"):
            lines.append(
                f"Driving decision: Longitudinal={decision.get('Longitudinal', '')}, "
                f"Lateral={decision.get('Lateral', '')}"
            )

        trace = self._get_with_aliases(
            driving,
            aliases=["Reasoning trace", "reasoning_trace", "trace", "justification"],
            default="",
        )
        if isinstance(trace, str) and trace.strip():
            lines.append(f"Reasoning trace: {trace.strip()}")
        return lines

    def _format_action_item(self, item: Any) -> str:
        if isinstance(item, dict):
            lon = self._get_with_aliases(item, aliases=["Longitudinal", "longitudinal"], default="")
            lat = self._get_with_aliases(item, aliases=["Lateral", "lateral"], default="")
            risk = self._get_with_aliases(item, aliases=["Risk level", "risk_level", "risk"], default="")
            reason = self._get_with_aliases(item, aliases=["Reason", "reason", "rationale"], default="")
            parts: List[str] = []
            if lon or lat:
                parts.append(f"Longitudinal={lon}, Lateral={lat}")
            if risk:
                parts.append(f"Risk={risk}")
            if reason:
                parts.append(f"Reason={reason}")
            return "; ".join(parts) if parts else str(item)
        return str(item)

    def _format_counterfactual_section(self, counterfactual: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["[Counterfactual]"]
        limit = max(1, int(self.config.reasoning_max_items_per_list))

        alt_actions = self._get_with_aliases(
            counterfactual,
            aliases=["Alternative actions", "alternative_actions", "safe_actions"],
            default=[],
        )
        if isinstance(alt_actions, list) and alt_actions:
            lines.append("Alternative actions:")
            for a in alt_actions[:limit]:
                lines.append(f"- {self._format_action_item(a)}")

        critical_actions = self._get_with_aliases(
            counterfactual,
            aliases=["Top safety-critical actions", "top_safety_critical_actions", "unsafe_actions"],
            default=[],
        )
        if isinstance(critical_actions, list) and critical_actions:
            lines.append("Top safety-critical actions:")
            for a in critical_actions[:limit]:
                lines.append(f"- {self._format_action_item(a)}")

        explanation = self._get_with_aliases(
            counterfactual,
            aliases=["summary", "reasoning", "counterfactual_reasoning"],
            default="",
        )
        if isinstance(explanation, str) and explanation.strip():
            lines.append(f"Counterfactual summary: {explanation.strip()}")
        return lines

    def _format_reasoning_text(
        self,
        spatial: Dict[str, Any],
        driving: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        if self._include_spatial():
            parts.extend(self._format_spatial_section(spatial))
            parts.append("")
        parts.extend(self._format_driving_section(driving))
        if self._include_counterfactual():
            parts.append("")
            parts.extend(self._format_counterfactual_section(counterfactual))
        return "\n".join(parts).strip()


# ======================================================================
# Collate & DataLoader builder
# ======================================================================


def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that handles mixed PIL-image / tensor / string fields.
    Images are left as lists for the VLM processor to handle.
    """
    return {
        "clip_names": [s["clip_name"] for s in batch],
        "frame_indices": [s["frame_index"] for s in batch],
        "timestamps": [s["timestamp_us"] for s in batch],
        "images": [s["images"] for s in batch],
        "history_times": [s["history_times"] for s in batch],
        "reasoning_texts": [s["reasoning_text"] for s in batch],
        "user_prompts": [s.get("user_prompt") for s in batch],
        "driving_decisions": [s["driving_decision"] for s in batch],
        "scene_descriptions": [s["scene_description"] for s in batch],
        "ego_trajectories": torch.stack([s["ego_trajectory"] for s in batch]),
        "ego_history_trajectories": torch.stack([s["ego_history_trajectory"] for s in batch]),
        "ego_velocities": torch.stack([s["ego_velocity"] for s in batch]),
        "ego_accelerations": torch.stack([s["ego_acceleration"] for s in batch]),
        "mission_commands": [s["mission_command"] for s in batch],
    }


def build_dataloader(
    config: VLADataConfig,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = NuReasoningVLADataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=vla_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

"""Planning-only data loading and causal LeVJEPA video preprocessing.

This module intentionally has no dependency on reasoning annotations.  It
constructs all temporal inputs by timestamp (never by assuming a frame rate or
using a positional stride), expresses poses in the current ego frame, and
leaves future trajectories in physical units.  Trajectory normalization is a
model concern and must not be applied here.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nureasoning.common.clips import discover_clips
from nureasoning.common.pickle_io import load_pickle

from .config import DataConfig
from .contracts import ObservationBatch, TrainingBatch


logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class PlanningDataError(ValueError):
    """A sample cannot satisfy the fixed, fully-valid planning contract."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class CoverageReport:
    """Explicit accounting for samples excluded while indexing a split."""

    split: str
    clips_total: int = 0
    clips_selected: int = 0
    candidates: int = 0
    included: int = 0
    excluded_by_reason: Counter[str] = field(default_factory=Counter)
    examples: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def excluded(self) -> int:
        return sum(self.excluded_by_reason.values())

    @property
    def coverage(self) -> float:
        return self.included / self.candidates if self.candidates else 0.0

    def exclude(self, reason: str, sample: str, *, max_examples: int = 5) -> None:
        self.excluded_by_reason[reason] += 1
        values = self.examples.setdefault(reason, [])
        if len(values) < max_examples:
            values.append(sample)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "clips_total": self.clips_total,
            "clips_selected": self.clips_selected,
            "candidates": self.candidates,
            "included": self.included,
            "excluded": self.excluded,
            "coverage": self.coverage,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "examples": {key: list(value) for key, value in self.examples.items()},
        }


@dataclass(frozen=True)
class PlanningSample:
    """One unbatched sample. ``future`` is absent for observation-only data."""

    video: torch.Tensor  # [V,3,F,H,W], ImageNet normalized
    camera_ids: torch.Tensor  # [V]
    frame_times_s: torch.Tensor  # [V,F], relative to current anchor
    ego_state: torch.Tensor  # [4] = vx,vy,ax,ay in current ego axes
    history: torch.Tensor  # [6,3], raw current-ego-frame cumulative poses
    command_id: torch.Tensor  # scalar long
    sample_id: str
    future: torch.Tensor | None = None  # [10,3], raw current-ego-frame poses


@dataclass(frozen=True)
class _TimedFrame:
    metadata_index: int
    time_s: float
    frame: Mapping[str, Any]


@dataclass(frozen=True)
class _IndexedSample:
    clip_dir: Path
    clip_name: str
    anchor: _TimedFrame
    video_frames: Tuple[Tuple[_TimedFrame, ...], ...]
    history_frames: Tuple[_TimedFrame, ...]
    future_frames: Tuple[_TimedFrame, ...]
    sample_id: str


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians into ``[-pi, pi)`` without changing array shape."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def global_pose_to_ego(
    pose: Sequence[float], anchor_pose: Sequence[float]
) -> np.ndarray:
    """Transform a global ``(x,y,heading)`` pose into the fixed anchor frame."""
    x, y, heading = (float(value) for value in pose[:3])
    ax, ay, aheading = (float(value) for value in anchor_pose[:3])
    dx, dy = x - ax, y - ay
    cosine, sine = math.cos(aheading), math.sin(aheading)
    return np.asarray(
        [
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            wrap_angle(heading - aheading),
        ],
        dtype=np.float32,
    )


def ego_pose_to_global(
    pose: Sequence[float], anchor_pose: Sequence[float]
) -> np.ndarray:
    """Inverse of :func:`global_pose_to_ego`."""
    x, y, heading = (float(value) for value in pose[:3])
    ax, ay, aheading = (float(value) for value in anchor_pose[:3])
    cosine, sine = math.cos(aheading), math.sin(aheading)
    return np.asarray(
        [
            ax + cosine * x - sine * y,
            ay + sine * x + cosine * y,
            wrap_angle(aheading + heading),
        ],
        dtype=np.float32,
    )


def global_vector_to_ego(vector: Sequence[float], anchor_heading: float) -> np.ndarray:
    """Rotate a global planar vector into current ego axes (no translation)."""
    x, y = float(vector[0]), float(vector[1])
    cosine, sine = math.cos(anchor_heading), math.sin(anchor_heading)
    return np.asarray([cosine * x + sine * y, -sine * x + cosine * y], dtype=np.float32)


def ego_vector_to_global(vector: Sequence[float], anchor_heading: float) -> np.ndarray:
    """Rotate a current-ego planar vector into global axes."""
    x, y = float(vector[0]), float(vector[1])
    cosine, sine = math.cos(anchor_heading), math.sin(anchor_heading)
    return np.asarray([cosine * x - sine * y, sine * x + cosine * y], dtype=np.float32)


def _plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlanningDataError("invalid_numeric_value", f"{name}={value!r}") from error
    if not math.isfinite(result):
        raise PlanningDataError("invalid_numeric_value", f"{name}={value!r}")
    return result


def _yaw_from_pose(pose: Mapping[str, Any]) -> float:
    if "yaw" in pose and pose["yaw"] is not None:
        return _finite_float(pose["yaw"], name="pose.yaw")
    quaternion_keys = ("qw", "qx", "qy", "qz")
    if not all(key in pose for key in quaternion_keys):
        raise PlanningDataError("missing_pose", "pose has neither yaw nor a full quaternion")
    qw, qx, qy, qz = (_finite_float(pose[key], name=f"pose.{key}") for key in quaternion_keys)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _pose_from_state(state: Any) -> np.ndarray:
    state_mapping = _plain_mapping(state)
    pose = _plain_mapping(state_mapping.get("pose"))
    if "x" not in pose or "y" not in pose:
        raise PlanningDataError("missing_pose", "ego state pose must contain x and y")
    return np.asarray(
        [
            _finite_float(pose["x"], name="pose.x"),
            _finite_float(pose["y"], name="pose.y"),
            _yaw_from_pose(pose),
        ],
        dtype=np.float64,
    )


def _state_vector(
    state: Any,
    field_name: str,
    component_names: Tuple[str, str],
    anchor_heading: float,
) -> np.ndarray:
    state_mapping = _plain_mapping(state)
    value = _plain_mapping(state_mapping.get(field_name))
    alternatives = ((component_names[0], component_names[1]), ("x", "y"))
    components: Tuple[float, float] | None = None
    for x_name, y_name in alternatives:
        if x_name in value and y_name in value:
            components = (
                _finite_float(value[x_name], name=f"{field_name}.{x_name}"),
                _finite_float(value[y_name], name=f"{field_name}.{y_name}"),
            )
            break
    if components is None:
        raise PlanningDataError(
            f"missing_{field_name}",
            f"ego state {field_name} must contain {component_names}",
        )
    coordinate_frame = str(
        value.get("frame", state_mapping.get(f"{field_name}_frame", "ego"))
    ).strip().lower()
    if coordinate_frame in {"ego", "vehicle", "body", "current_ego"}:
        return np.asarray(components, dtype=np.float32)
    if coordinate_frame in {"global", "world", "map"}:
        return global_vector_to_ego(components, anchor_heading)
    raise PlanningDataError(
        f"invalid_{field_name}_frame", f"unsupported coordinate frame {coordinate_frame!r}"
    )


def _load_metadata(clip_dir: Path) -> Mapping[str, Any]:
    path = clip_dir / "metadata.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningDataError("invalid_metadata", f"cannot load {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PlanningDataError("invalid_metadata", f"{path} must contain an object")
    return value


def _timed_frames(metadata: Mapping[str, Any]) -> List[_TimedFrame]:
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PlanningDataError("missing_frames", "metadata.frames is empty")

    use_timestamps = all(
        isinstance(frame, Mapping) and isinstance(frame.get("timestamp_us"), (int, float))
        for frame in frames
    )
    use_relative = all(
        isinstance(frame, Mapping) and isinstance(frame.get("relative_time_s"), (int, float))
        for frame in frames
    )
    if not use_timestamps and not use_relative:
        raise PlanningDataError(
            "missing_frame_timestamps",
            "every frame needs timestamp_us or every frame needs relative_time_s",
        )
    result: List[_TimedFrame] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise PlanningDataError("invalid_metadata", f"frames[{index}] is not an object")
        value = float(frame["timestamp_us"]) / 1e6 if use_timestamps else float(frame["relative_time_s"])
        if not math.isfinite(value):
            raise PlanningDataError("invalid_frame_timestamp", f"frames[{index}] time is not finite")
        result.append(_TimedFrame(index, value, frame))
    result.sort(key=lambda item: (item.time_s, item.metadata_index))
    for previous, current in zip(result, result[1:]):
        if current.time_s <= previous.time_s:
            raise PlanningDataError(
                "non_unique_frame_timestamps",
                f"frame times must be strictly increasing; duplicate {current.time_s}",
            )
    return result


def _sensor_camera_path(frame: Mapping[str, Any], camera: str) -> str:
    sensors = _plain_mapping(frame.get("sensors"))
    cameras = _plain_mapping(sensors.get("cameras"))
    value = cameras.get(camera)
    return str(value) if value else ""


def _resolve_asset_path(clip_dir: Path, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        # Released metadata can retain an extraction-machine prefix.  Recover
        # the clip-relative cameras path without accepting an arbitrary basename.
        normalized = path.as_posix()
        marker = "/cameras/"
        if marker in normalized:
            path = Path("cameras") / normalized.split(marker, 1)[1]
        else:
            return path
    return clip_dir / path


def _ego_state_path(clip_dir: Path, frame: Mapping[str, Any]) -> Path:
    value = frame.get("ego_state")
    if not value:
        raise PlanningDataError("missing_ego_state", "frame has no ego_state path")
    return _resolve_asset_path(clip_dir, str(value))


def _load_ego_state(clip_dir: Path, frame: Mapping[str, Any]) -> Any:
    path = _ego_state_path(clip_dir, frame)
    if not path.is_file():
        raise PlanningDataError("missing_ego_state", f"file does not exist: {path}")
    try:
        return load_pickle(str(path))
    except Exception as error:
        raise PlanningDataError("invalid_ego_state", f"cannot load {path}: {error}") from error


def _nearest_frame(
    frames: Sequence[_TimedFrame],
    times: Sequence[float],
    target_s: float,
    tolerance_s: float,
    *,
    reason: str,
) -> _TimedFrame:
    position = bisect.bisect_left(times, target_s)
    candidates = []
    if position < len(frames):
        candidates.append(frames[position])
    if position:
        candidates.append(frames[position - 1])
    if not candidates:
        raise PlanningDataError(reason, f"no frame near target time {target_s:.6f}")
    selected = min(candidates, key=lambda item: (abs(item.time_s - target_s), item.time_s))
    distance = abs(selected.time_s - target_s)
    if distance > tolerance_s + 1e-9:
        raise PlanningDataError(
            reason,
            f"nearest frame is {distance:.6f}s from target {target_s:.6f}s "
            f"(tolerance={tolerance_s:.6f}s)",
        )
    return selected


def _select_pose_grid(
    frames: Sequence[_TimedFrame],
    anchor_time_s: float,
    offsets_s: Sequence[float],
    tolerance_s: float,
    *,
    reason: str,
) -> Tuple[_TimedFrame, ...]:
    times = [item.time_s for item in frames]
    selected = tuple(
        _nearest_frame(
            frames, times, anchor_time_s + offset, tolerance_s, reason=reason
        )
        for offset in offsets_s
    )
    if len({item.metadata_index for item in selected}) != len(selected):
        raise PlanningDataError(reason, "multiple target times selected the same frame")
    return selected


def _select_camera_clip(
    clip_dir: Path,
    frames: Sequence[_TimedFrame],
    anchor: _TimedFrame,
    camera: str,
    config: DataConfig,
) -> Tuple[_TimedFrame, ...]:
    anchor_path = _sensor_camera_path(anchor.frame, camera)
    if not anchor_path or not _resolve_asset_path(clip_dir, anchor_path).is_file():
        raise PlanningDataError(
            "missing_anchor_camera", f"camera {camera!r} is unavailable at the anchor"
        )
    # Causality is enforced before nearest-neighbour matching.  In particular,
    # frame_tolerance_s never authorizes selecting a post-anchor observation.
    available = []
    for item in frames:
        if item.time_s > anchor.time_s + 1e-9:
            break
        value = _sensor_camera_path(item.frame, camera)
        if value and _resolve_asset_path(clip_dir, value).is_file():
            available.append(item)
    times = [item.time_s for item in available]
    target_offsets = np.linspace(
        -float(config.video_window_s), 0.0, int(config.num_video_frames)
    )
    selected: List[_TimedFrame] = []
    for offset in target_offsets[:-1]:
        selected.append(
            _nearest_frame(
                available,
                times,
                anchor.time_s + float(offset),
                config.frame_tolerance_s,
                reason="missing_video_frame",
            )
        )
    selected.append(anchor)
    if len({item.metadata_index for item in selected}) != len(selected):
        raise PlanningDataError("missing_video_frame", "video sampling produced duplicate frames")
    if any(current.time_s <= previous.time_s for previous, current in zip(selected, selected[1:])):
        raise PlanningDataError("nonchronological_video", "selected camera frames are not chronological")
    return tuple(selected)


def _sample_identifier(clip_name: str, anchor: _TimedFrame) -> str:
    timestamp = anchor.frame.get("timestamp_us")
    suffix = str(int(timestamp)) if isinstance(timestamp, (int, float)) else f"{anchor.time_s:.6f}"
    return f"{clip_name}:{suffix}"


class _ClipReader:
    def __init__(self, clip_dir: Path, config: DataConfig, observation_only: bool):
        self.clip_dir = clip_dir
        self.config = config
        self.observation_only = observation_only
        self.metadata = _load_metadata(clip_dir)
        self.frames = _timed_frames(self.metadata)
        self._state_cache: Dict[Path, Any] = {}

    def load_state(self, frame: Mapping[str, Any]) -> Any:
        path = _ego_state_path(self.clip_dir, frame)
        if path not in self._state_cache:
            self._state_cache[path] = _load_ego_state(self.clip_dir, frame)
        return self._state_cache[path]

    def index_anchor(self, anchor: _TimedFrame, *, validate_states: bool = True) -> _IndexedSample:
        video_frames = tuple(
            _select_camera_clip(self.clip_dir, self.frames, anchor, camera, self.config)
            for camera in self.config.cameras
        )
        history_offsets = tuple(
            -self.config.history_dt_s * step
            for step in range(self.config.history_steps, 0, -1)
        )
        history = _select_pose_grid(
            self.frames,
            anchor.time_s,
            history_offsets,
            self.config.frame_tolerance_s,
            reason="missing_history_frame",
        )
        future: Tuple[_TimedFrame, ...] = ()
        if not self.observation_only:
            future_offsets = tuple(
                self.config.future_dt_s * step
                for step in range(1, self.config.future_steps + 1)
            )
            future = _select_pose_grid(
                self.frames,
                anchor.time_s,
                future_offsets,
                self.config.frame_tolerance_s,
                reason="missing_future_frame",
            )

        if validate_states:
            anchor_state = self.load_state(anchor.frame)
            anchor_pose = _pose_from_state(anchor_state)
            _state_vector(anchor_state, "velocity", ("vx", "vy"), float(anchor_pose[2]))
            _state_vector(anchor_state, "acceleration", ("ax", "ay"), float(anchor_pose[2]))
            for item in (*history, *future):
                _pose_from_state(self.load_state(item.frame))

        return _IndexedSample(
            clip_dir=self.clip_dir,
            clip_name=self.clip_dir.name,
            anchor=anchor,
            video_frames=video_frames,
            history_frames=history,
            future_frames=future,
            sample_id=_sample_identifier(self.clip_dir.name, anchor),
        )


def _center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    """Deterministic resize-and-center-crop shared by every frame in a clip."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise PlanningDataError("invalid_image", f"invalid image size {(width, height)}")
    scale = image_size / min(width, height)
    resized_width = max(image_size, int(round(width * scale)))
    resized_height = max(image_size, int(round(height * scale)))
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (resized_width - image_size) // 2
    top = (resized_height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    try:
        with Image.open(path) as opened:
            image = _center_crop_resize(opened.convert("RGB"), image_size)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
    except (OSError, ValueError) as error:
        raise PlanningDataError("invalid_image", f"cannot load {path}: {error}") from error
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _materialize(indexed: _IndexedSample, config: DataConfig) -> PlanningSample:
    anchor_state = _load_ego_state(indexed.clip_dir, indexed.anchor.frame)
    anchor_pose = _pose_from_state(anchor_state)
    velocity = _state_vector(anchor_state, "velocity", ("vx", "vy"), float(anchor_pose[2]))
    acceleration = _state_vector(
        anchor_state, "acceleration", ("ax", "ay"), float(anchor_pose[2])
    )

    camera_vocabulary = {name: index for index, name in enumerate(config.camera_vocab)}
    videos: List[torch.Tensor] = []
    all_frame_times: List[torch.Tensor] = []
    for camera, selected in zip(config.cameras, indexed.video_frames):
        images = []
        for item in selected:
            relative = _sensor_camera_path(item.frame, camera)
            images.append(_load_image(_resolve_asset_path(indexed.clip_dir, relative), config.image_size))
        videos.append(torch.stack(images, dim=1))  # [3,F,H,W]
        all_frame_times.append(
            torch.tensor(
                [item.time_s - indexed.anchor.time_s for item in selected],
                dtype=torch.float32,
            )
        )

    history = torch.from_numpy(
        np.stack(
            [
                global_pose_to_ego(
                    _pose_from_state(_load_ego_state(indexed.clip_dir, item.frame)),
                    anchor_pose,
                )
                for item in indexed.history_frames
            ]
        )
    )
    future = None
    if indexed.future_frames:
        future = torch.from_numpy(
            np.stack(
                [
                    global_pose_to_ego(
                        _pose_from_state(_load_ego_state(indexed.clip_dir, item.frame)),
                        anchor_pose,
                    )
                    for item in indexed.future_frames
                ]
            )
        )

    mission = _plain_mapping(indexed.anchor.frame.get("mission_goal"))
    command = str(mission.get("command", "UNKNOWN") or "UNKNOWN").strip().upper()
    command_vocabulary = {name.upper(): index for index, name in enumerate(config.command_vocab)}
    command_id = command_vocabulary.get(command, command_vocabulary["UNKNOWN"])

    return PlanningSample(
        video=torch.stack(videos),
        camera_ids=torch.tensor(
            [camera_vocabulary[camera] for camera in config.cameras], dtype=torch.long
        ),
        frame_times_s=torch.stack(all_frame_times),
        ego_state=torch.from_numpy(np.concatenate((velocity, acceleration))).float(),
        history=history.float(),
        command_id=torch.tensor(command_id, dtype=torch.long),
        sample_id=indexed.sample_id,
        future=future.float() if future is not None else None,
    )


class PlanningDataset(Dataset[PlanningSample]):
    """Planning samples independent of reasoning labels.

    Args:
        config: Frozen data contract configuration.
        split: Used for reporting and to restrict clip subsampling to ``train``.
        observation_only: If true, index without looking for or loading future
            ego states.  This is the required inference mode.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        observation_only: bool = False,
    ) -> None:
        if not config.require_complete_inputs:
            raise ValueError(
                "the fixed-token MVP requires complete inputs; "
                "set data.require_complete_inputs=true"
            )
        if not 0.0 < config.train_clip_fraction <= 1.0 or not math.isfinite(
            config.train_clip_fraction
        ):
            raise ValueError("train_clip_fraction must be finite and in (0, 1]")
        if not config.cameras or len(set(config.cameras)) != len(config.cameras):
            raise ValueError("cameras must be non-empty and unique")
        unknown_cameras = set(config.cameras) - set(config.camera_vocab)
        if unknown_cameras:
            raise ValueError(f"cameras missing from camera_vocab: {sorted(unknown_cameras)}")
        if not config.command_vocab or config.command_vocab[0] != "UNKNOWN":
            raise ValueError("command_vocab[0] must be UNKNOWN")
        self.config = config
        self.split = str(split)
        self.observation_only = bool(observation_only)
        self.samples: List[_IndexedSample] = []
        self.coverage_report = CoverageReport(split=self.split)
        self._all_clip_ids: set[str] = set()
        self._all_log_ids: set[str] = set()
        self._discover()

    @property
    def coverage(self) -> Dict[str, Any]:
        return self.coverage_report.as_dict()

    def _discover(self) -> None:
        root = Path(self.config.train_root if self.split == "train" else self.config.val_root)
        all_clip_dirs = [Path(path) for path in discover_clips(str(root))]
        self.coverage_report.clips_total = len(all_clip_dirs)

        for clip_dir in all_clip_dirs:
            try:
                metadata = _load_metadata(clip_dir)
            except PlanningDataError as error:
                self.coverage_report.exclude(error.reason, str(clip_dir))
                continue
            self._all_clip_ids.add(str(metadata.get("clip_token") or clip_dir.name))
            log_name = metadata.get("log_name")
            if log_name:
                self._all_log_ids.add(str(log_name))

        selected = list(all_clip_dirs)
        if self.split == "train" and self.config.train_clip_fraction < 1.0 and selected:
            count = max(1, math.ceil(len(selected) * self.config.train_clip_fraction))
            selected = sorted(random.Random(self.config.clip_seed).sample(selected, count))
        self.coverage_report.clips_selected = len(selected)

        for clip_dir in selected:
            try:
                reader = _ClipReader(clip_dir, self.config, self.observation_only)
            except PlanningDataError as error:
                self.coverage_report.exclude(error.reason, str(clip_dir))
                continue
            for anchor in reader.frames:
                self.coverage_report.candidates += 1
                sample_name = _sample_identifier(clip_dir.name, anchor)
                try:
                    indexed = reader.index_anchor(anchor, validate_states=True)
                except PlanningDataError as error:
                    self.coverage_report.exclude(error.reason, sample_name)
                    continue
                self.samples.append(indexed)
                self.coverage_report.included += 1

        logger.info("JEPA planning data coverage: %s", self.coverage_report.as_dict())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PlanningSample:
        return _materialize(self.samples[index], self.config)


def assert_disjoint_planning_splits(*datasets: PlanningDataset) -> None:
    """Fail fast if any datasets share a clip token/name or log identity.

    Call this after constructing train and validation datasets.  Identities are
    collected before train-fraction subsampling so an overlap cannot be hidden
    merely because the duplicated clip was not selected for this run.
    """
    for index, left in enumerate(datasets):
        for right in datasets[index + 1 :]:
            clip_overlap = left._all_clip_ids & right._all_clip_ids
            log_overlap = left._all_log_ids & right._all_log_ids
            if clip_overlap or log_overlap:
                raise ValueError(
                    f"planning split overlap between {left.split!r} and {right.split!r}: "
                    f"clips={sorted(clip_overlap)} logs={sorted(log_overlap)}"
                )


def load_planning_sample(
    clip_dir: str | Path,
    config: DataConfig,
    *,
    anchor_frame_index: int | None = None,
    anchor_timestamp_us: int | None = None,
    observation_only: bool = True,
) -> PlanningSample:
    """Load one anchor for training or inference without dataset discovery.

    Exactly one anchor selector is required. ``anchor_frame_index`` selects an
    entry in ``metadata.frames`` only; every temporal grid is still rebuilt by
    timestamps, so nonuniform frame rates and reordered frame arrays are safe.
    Observation-only mode never reads or validates post-anchor ego states.
    """
    if (anchor_frame_index is None) == (anchor_timestamp_us is None):
        raise ValueError("provide exactly one of anchor_frame_index or anchor_timestamp_us")
    reader = _ClipReader(Path(clip_dir), config, observation_only)
    anchor: _TimedFrame | None = None
    if anchor_frame_index is not None:
        anchor = next(
            (item for item in reader.frames if item.metadata_index == anchor_frame_index), None
        )
    else:
        for item in reader.frames:
            try:
                matches = int(item.frame.get("timestamp_us")) == int(anchor_timestamp_us)
            except (TypeError, ValueError):
                matches = False
            if matches:
                anchor = item
                break
    if anchor is None:
        raise PlanningDataError("anchor_not_found", "requested anchor is not in metadata")
    return _materialize(reader.index_anchor(anchor, validate_states=True), config)


def planning_collate_fn(
    samples: Sequence[PlanningSample],
) -> ObservationBatch | TrainingBatch:
    """Collate homogeneous observation-only or supervised planning samples."""
    if not samples:
        raise ValueError("cannot collate an empty planning batch")
    futures_present = [sample.future is not None for sample in samples]
    if any(futures_present) and not all(futures_present):
        raise ValueError("cannot mix observation-only and supervised samples")
    common = {
        "video": torch.stack([sample.video for sample in samples]),
        "camera_ids": torch.stack([sample.camera_ids for sample in samples]),
        "frame_times_s": torch.stack([sample.frame_times_s for sample in samples]),
        "ego_state": torch.stack([sample.ego_state for sample in samples]),
        "history": torch.stack([sample.history for sample in samples]),
        "command_id": torch.stack([sample.command_id for sample in samples]),
        "sample_id": [sample.sample_id for sample in samples],
    }
    if all(futures_present):
        batch = TrainingBatch(
            **common,
            future=torch.stack([sample.future for sample in samples]),  # type: ignore[arg-type]
        )
    else:
        batch = ObservationBatch(**common)
    return batch.validate(num_cameras=samples[0].video.shape[0])


__all__ = [
    "CoverageReport",
    "PlanningDataError",
    "PlanningDataset",
    "PlanningSample",
    "assert_disjoint_planning_splits",
    "ego_pose_to_global",
    "ego_vector_to_global",
    "global_pose_to_ego",
    "global_vector_to_ego",
    "load_planning_sample",
    "planning_collate_fn",
    "wrap_angle",
]

"""Shared tensor contracts for planning data, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Union

import torch


TensorBatch = Union["ObservationBatch", "TrainingBatch"]


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tuple(tensor.shape)}")
    for actual, expected in zip(tensor.shape, shape):
        if expected >= 0 and actual != expected:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


@dataclass
class ObservationBatch:
    """Inputs available at planning time.

    ``frame_times_s`` are relative to the current planning anchor.  A positive
    value is a contract violation because it would leak future observations.
    All history poses are cumulative poses in the current ego frame.
    """

    video: torch.Tensor
    camera_ids: torch.Tensor
    frame_times_s: torch.Tensor
    ego_state: torch.Tensor
    history: torch.Tensor
    command_id: torch.Tensor
    sample_id: List[str]

    @property
    def batch_size(self) -> int:
        return int(self.video.shape[0])

    def validate(
        self,
        *,
        num_cameras: int | None = None,
        num_frames: int = 16,
        image_size: int = 224,
        num_history: int = 6,
        camera_vocab_size: int | None = None,
        command_vocab_size: int | None = None,
        causal_tolerance_s: float = 1e-6,
    ) -> "ObservationBatch":
        batch = self.batch_size
        cameras = int(self.video.shape[1]) if self.video.ndim >= 2 else -1
        if num_cameras is not None and cameras != num_cameras:
            raise ValueError(f"video has {cameras} cameras; expected {num_cameras}")
        _require_shape(
            "video", self.video,
            (batch, cameras, 3, num_frames, image_size, image_size),
        )
        _require_shape("camera_ids", self.camera_ids, (batch, cameras))
        _require_shape("frame_times_s", self.frame_times_s, (batch, cameras, num_frames))
        _require_shape("ego_state", self.ego_state, (batch, 4))
        _require_shape("history", self.history, (batch, num_history, 3))
        _require_shape("command_id", self.command_id, (batch,))
        if len(self.sample_id) != batch:
            raise ValueError(f"sample_id has {len(self.sample_id)} entries; expected {batch}")
        if self.camera_ids.dtype != torch.long:
            raise TypeError("camera_ids must use torch.long")
        if self.command_id.dtype != torch.long:
            raise TypeError("command_id must use torch.long")
        for name in ("video", "frame_times_s", "ego_state", "history"):
            _validate_finite(name, getattr(self, name))
        if torch.any(self.frame_times_s > causal_tolerance_s):
            latest = float(self.frame_times_s.max().item())
            raise ValueError(f"future-data leakage: frame time {latest:.6f}s is after anchor")
        if torch.any(self.frame_times_s[..., 1:] < self.frame_times_s[..., :-1]):
            raise ValueError("frame_times_s must be chronological within every camera")
        if camera_vocab_size is not None and (
            torch.any(self.camera_ids < 0) or torch.any(self.camera_ids >= camera_vocab_size)
        ):
            raise ValueError("camera_ids contain an out-of-vocabulary value")
        if command_vocab_size is not None and (
            torch.any(self.command_id < 0) or torch.any(self.command_id >= command_vocab_size)
        ):
            raise ValueError("command_id contains an out-of-vocabulary value")
        return self

    def to(self, *args, **kwargs) -> "ObservationBatch":
        values: Dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(*args, **kwargs) if torch.is_tensor(value) else value
        return type(self)(**values)


@dataclass
class TrainingBatch(ObservationBatch):
    """Observation inputs plus a raw ten-pose future supervision target."""

    future: torch.Tensor

    def validate(self, *, num_waypoints: int = 10, **kwargs) -> "TrainingBatch":
        super().validate(**kwargs)
        _require_shape("future", self.future, (self.batch_size, num_waypoints, 3))
        _validate_finite("future", self.future)
        return self

    def observations(self) -> ObservationBatch:
        return ObservationBatch(
            video=self.video,
            camera_ids=self.camera_ids,
            frame_times_s=self.frame_times_s,
            ego_state=self.ego_state,
            history=self.history,
            command_id=self.command_id,
            sample_id=list(self.sample_id),
        )

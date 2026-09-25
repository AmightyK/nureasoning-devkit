"""Observation-only JEPA planner adapter for the local planning benchmark.

The model predicts ten cumulative poses in the current ego frame at 0.5 s
intervals.  The benchmark callback consumes 51 global poses at 0.1 s intervals,
including the key-frame pose at ``t=0``.  Conversion lives here so neither the
model nor the official evaluator needs benchmark-specific behavior.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import CheckpointState, inspect_checkpoint, load_checkpoint
from .config import PlanningConfig, Stage
from .contracts import ObservationBatch
from .data import load_planning_sample, planning_collate_fn
from .model import PlanningModel, build_planning_model


MODEL_FUTURE_STEPS = 10
MODEL_FUTURE_DT_S = 0.5
BENCHMARK_HORIZON_S = 5.0
BENCHMARK_DT_S = 0.1
BENCHMARK_STEPS = 51


def wrap_to_pi(values: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to ``[-pi, pi)`` without changing array shape."""

    wrapped = (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(values):
        return float(wrapped)
    return wrapped


def interpolate_ego_trajectory(
    ego_waypoints: np.ndarray,
    *,
    source_dt_s: float = MODEL_FUTURE_DT_S,
    target_dt_s: float = BENCHMARK_DT_S,
    horizon_s: float = BENCHMARK_HORIZON_S,
) -> np.ndarray:
    """Convert ten future poses to the benchmark grid in the ego frame.

    ``ego_waypoints`` must omit the current pose and represent times
    ``source_dt_s, ..., horizon_s``.  A zero current-ego pose is prepended.
    Heading is unwrapped before linear interpolation and wrapped afterwards,
    preventing interpolation from taking the long route across ``+/-pi``.
    """

    trajectory = np.asarray(ego_waypoints, dtype=np.float64)
    if source_dt_s <= 0.0 or target_dt_s <= 0.0 or horizon_s <= 0.0:
        raise ValueError("trajectory intervals and horizon must be positive")
    expected_source_steps = int(round(horizon_s / source_dt_s))
    if not math.isclose(expected_source_steps * source_dt_s, horizon_s):
        raise ValueError("source interval must divide the planning horizon exactly")
    if trajectory.shape != (expected_source_steps, 3):
        raise ValueError(
            "model trajectory must have shape "
            f"({expected_source_steps}, 3), got {trajectory.shape}"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("model trajectory contains non-finite values")
    source_times = np.arange(
        expected_source_steps + 1, dtype=np.float64
    ) * source_dt_s
    source = np.vstack((np.zeros((1, 3), dtype=np.float64), trajectory))
    source[:, 2] = np.unwrap(source[:, 2])

    target_steps = int(round(horizon_s / target_dt_s)) + 1
    target_times = np.arange(target_steps, dtype=np.float64) * target_dt_s
    target_times[-1] = horizon_s
    interpolated = np.column_stack(
        [
            np.interp(target_times, source_times, source[:, dimension])
            for dimension in range(3)
        ]
    )
    interpolated[:, 2] = wrap_to_pi(interpolated[:, 2])
    interpolated[0] = 0.0
    return interpolated


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def ego_pose_from_state(ego_state: Any) -> tuple[float, float, float]:
    """Extract the benchmark key-frame ``(x, y, yaw)`` pose."""

    pose = _field(ego_state, "pose")
    if pose is None:
        raise ValueError("benchmark ego_state does not contain a key-frame pose")
    coordinates = tuple(_field(pose, name) for name in ("x", "y", "yaw"))
    if any(value is None for value in coordinates):
        raise ValueError("benchmark ego_state pose must contain x, y, and yaw")
    result = tuple(float(value) for value in coordinates)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("benchmark key-frame pose contains non-finite values")
    return result  # type: ignore[return-value]


def ego_to_global(
    ego_trajectory: np.ndarray,
    anchor_pose: tuple[float, float, float],
) -> np.ndarray:
    """Transform cumulative current-ego poses into the global map frame."""

    trajectory = np.asarray(ego_trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(
            f"ego trajectory must have shape (N, 3), got {trajectory.shape}"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("ego trajectory contains non-finite values")
    ego_x, ego_y, ego_yaw = anchor_pose
    cosine = math.cos(ego_yaw)
    sine = math.sin(ego_yaw)
    global_trajectory = np.empty_like(trajectory)
    global_trajectory[:, 0] = (
        ego_x + trajectory[:, 0] * cosine - trajectory[:, 1] * sine
    )
    global_trajectory[:, 1] = (
        ego_y + trajectory[:, 0] * sine + trajectory[:, 1] * cosine
    )
    global_trajectory[:, 2] = wrap_to_pi(ego_yaw + trajectory[:, 2])
    return global_trajectory


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is unavailable")
    return requested


def validate_evaluation_checkpoint(state: CheckpointState) -> None:
    """Reject checkpoints that cannot represent the requested planner."""

    if state.stage is not Stage.JOINT or not state.stage_complete:
        raise ValueError(
            "evaluation requires a completed joint JEPA-planning checkpoint; "
            f"got stage={state.stage.value!r}, stage_complete={state.stage_complete}"
        )
    data = state.config.data
    if data.future_steps != MODEL_FUTURE_STEPS or not math.isclose(
        data.future_dt_s, MODEL_FUTURE_DT_S
    ):
        raise ValueError(
            "benchmark provider requires ten future poses at 0.5 s intervals"
        )


def load_evaluation_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[PlanningModel, PlanningConfig, CheckpointState]:
    """Build and strictly restore a completed planning checkpoint.

    The saved configuration is authoritative.  This path imports no Qwen/VLM
    model and never enables downloads or unaudited remote model code.
    """

    resolved_device = _resolve_device(device)
    state = inspect_checkpoint(checkpoint_path, map_location="cpu")
    validate_evaluation_checkpoint(state)
    model = build_planning_model(state.config)
    load_checkpoint(
        checkpoint_path,
        model=model,
        config=state.config,
        compatibility="resume",
        restore_rng=False,
        map_location="cpu",
    )
    model.to(resolved_device)
    model.eval()
    return model, state.config, state


class JEPAPlanningTrajectoryProvider:
    """Benchmark callback backed by LeVJEPA + intent JEPA + DiT.

    Production construction takes a planning checkpoint.  ``model`` and
    ``config`` are explicit dependency-injection hooks for unit tests; both or
    neither must be supplied.  Inference preprocessing is always observation
    only, and :meth:`PlanningModel.predict` initializes the DiT from pure noise.
    """

    horizon_s = BENCHMARK_HORIZON_S
    dt_s = BENCHMARK_DT_S

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
        num_inference_steps: int | None = None,
        seed: int = 42,
        intent_mode: str = "predicted",
        model: nn.Module | None = None,
        config: PlanningConfig | None = None,
    ) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if num_inference_steps is not None and num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if intent_mode not in {"predicted", "no_intent"}:
            raise ValueError("benchmark intent_mode must be predicted or no_intent")
        injected = model is not None or config is not None
        if injected and (model is None or config is None):
            raise ValueError("model and config must be supplied together")
        if injected and checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be combined with an injected model")
        if not injected and checkpoint_path is None:
            raise ValueError("checkpoint_path is required when no model is injected")

        self.device = _resolve_device(device)
        if injected:
            assert model is not None and config is not None
            self.model = model.to(self.device)
            self.config = config.validate()
            if self.config.data.future_steps != MODEL_FUTURE_STEPS or not math.isclose(
                self.config.data.future_dt_s, MODEL_FUTURE_DT_S
            ):
                raise ValueError(
                    "benchmark provider requires ten future poses at 0.5 s intervals"
                )
            self.checkpoint_state: CheckpointState | None = None
        else:
            self.model, self.config, self.checkpoint_state = load_evaluation_model(
                checkpoint_path, device=self.device  # type: ignore[arg-type]
            )
        self.model.eval()
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.intent_mode = intent_mode
        self.last_latency_s: float | None = None
        self.latencies_s: list[float] = []

    @property
    def num_steps(self) -> int:
        return BENCHMARK_STEPS

    def _sample_seed(self, clip_path: str, key_frame_idx: int) -> int:
        identity = f"{Path(clip_path).resolve()}:{key_frame_idx}".encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
        return (self.seed + offset) % (2**63 - 1)

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> np.ndarray:
        sample = load_planning_sample(
            clip_path,
            self.config.data,
            anchor_frame_index=int(key_frame_idx),
            observation_only=True,
        )
        batch = planning_collate_fn([sample])
        if not isinstance(batch, ObservationBatch) or hasattr(batch, "future"):
            raise TypeError("provider preprocessing must produce an observation-only batch")
        batch = batch.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        predicted = self.model.predict(
            batch,
            num_steps=self.num_inference_steps,
            seed=self._sample_seed(clip_path, key_frame_idx),
            intent_mode=self.intent_mode,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_latency_s = time.perf_counter() - started
        self.latencies_s.append(self.last_latency_s)

        if not torch.is_tensor(predicted):
            raise TypeError("PlanningModel.predict must return a torch.Tensor")
        ego_waypoints = predicted.detach().float().cpu().numpy()
        if ego_waypoints.shape != (1, MODEL_FUTURE_STEPS, 3):
            raise ValueError(
                "PlanningModel.predict returned shape "
                f"{ego_waypoints.shape}; expected (1, {MODEL_FUTURE_STEPS}, 3)"
            )
        dense_ego = interpolate_ego_trajectory(ego_waypoints[0])
        global_trajectory = ego_to_global(dense_ego, ego_pose_from_state(ego_state))
        if global_trajectory.shape != (BENCHMARK_STEPS, 3):
            raise AssertionError("internal benchmark trajectory shape mismatch")
        return global_trajectory


__all__ = [
    "BENCHMARK_DT_S",
    "BENCHMARK_HORIZON_S",
    "BENCHMARK_STEPS",
    "JEPAPlanningTrajectoryProvider",
    "ego_pose_from_state",
    "ego_to_global",
    "interpolate_ego_trajectory",
    "load_evaluation_model",
    "validate_evaluation_checkpoint",
    "wrap_to_pi",
]

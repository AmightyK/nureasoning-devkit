"""
Baseline planners for the nuReasoning planning benchmark.

Every baseline implements the benchmark's ``trajectory_provider`` interface:

    provider(clip_path: str, key_frame_idx: int, ego_state) -> np.ndarray

returning a global-frame ``(N, 3)`` array of ``(x, y, yaw)`` waypoints sampled
at 0.1 s from t=0 to the planning horizon (5 s by default, 51 points).

Available baselines:
  * ``constant_velocity`` — simple kinematic extrapolation (fully implemented).
  * ``uniad``             — UniAD end-to-end planner (placeholder).
  * ``diffusion_drive``   — DiffusionDrive planner (placeholder).

The trained nuVLA model is exposed through
``nureasoning.nuvla.trajectory_provider.VLATrajectoryProvider``.
"""

from .base import BASELINE_REGISTRY, BaseTrajectoryProvider, get_baseline
from .constant_velocity import ConstantVelocityProvider

__all__ = [
    "BASELINE_REGISTRY",
    "BaseTrajectoryProvider",
    "ConstantVelocityProvider",
    "get_baseline",
]

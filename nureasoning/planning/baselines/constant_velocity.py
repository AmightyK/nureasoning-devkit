"""Constant-velocity kinematic baseline."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .base import BaseTrajectoryProvider, register_baseline


@register_baseline("constant_velocity")
class ConstantVelocityProvider(BaseTrajectoryProvider):
    """
    Extrapolates the ego pose forward assuming the current ego-frame velocity
    stays constant over the planning horizon. Serves as the sanity-check lower
    bound for the planning benchmark and as a reference implementation of the
    ``trajectory_provider`` interface.
    """

    def __init__(self, horizon_s: float = 5.0, dt_s: float = 0.1):
        self.horizon_s = horizon_s
        self.dt_s = dt_s

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        del clip_path, key_frame_idx

        pose = ego_state.pose if isinstance(ego_state.pose, dict) else {}
        velocity = ego_state.velocity if isinstance(ego_state.velocity, dict) else {}

        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        yaw = float(pose.get("yaw", 0.0))
        vx = float(velocity.get("vx", 0.0))  # ego frame
        vy = float(velocity.get("vy", 0.0))

        # Rotate the ego-frame velocity into the global frame.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        vx_g = vx * cos_y - vy * sin_y
        vy_g = vx * sin_y + vy * cos_y

        times = np.arange(self.num_steps, dtype=np.float64) * self.dt_s
        trajectory = np.empty((self.num_steps, 3), dtype=np.float64)
        trajectory[:, 0] = x + vx_g * times
        trajectory[:, 1] = y + vy_g * times
        trajectory[:, 2] = yaw
        return trajectory

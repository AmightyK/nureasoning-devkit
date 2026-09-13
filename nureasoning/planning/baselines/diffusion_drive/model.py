"""
DiffusionDrive baseline (placeholder).

DiffusionDrive is a diffusion-based end-to-end planner. This module is a
placeholder for integrating a DiffusionDrive-style model into the nuReasoning
planning benchmark.

To integrate:

  1. Install DiffusionDrive and its dependencies, and obtain checkpoints
     trained (or fine-tuned) on nuReasoning.
  2. Convert nuReasoning clip data to the model's expected input format:
     multi-view cameras / scene context from ``metadata.json``, plus the ego
     state and route command of the key frame.
  3. Implement ``DiffusionDriveTrajectoryProvider.__call__`` to run inference
     at the key frame and return a global-frame ``(N, 3)`` array of
     ``(x, y, yaw)`` waypoints at 0.1 s resolution over the 5 s horizon (see
     ``nureasoning.planning.baselines.base.BaseTrajectoryProvider``). Convert
     ego-frame waypoints with the ego pose as in
     ``nureasoning.nuvla.trajectory_provider.VLATrajectoryProvider._ego_to_global``.
  4. Evaluate with ``python -m nureasoning.planning.benchmark``, or produce a
     challenge submission with
     ``python -m nureasoning.submission.challenge --planning-provider diffusion_drive``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import BaseTrajectoryProvider, register_baseline


@register_baseline("diffusion_drive")
class DiffusionDriveTrajectoryProvider(BaseTrajectoryProvider):
    """Placeholder trajectory provider for the DiffusionDrive baseline."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        raise NotImplementedError(
            "DiffusionDrive is provided as a placeholder baseline. Follow the "
            "integration notes in "
            "nureasoning/planning/baselines/diffusion_drive/model.py."
        )

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError

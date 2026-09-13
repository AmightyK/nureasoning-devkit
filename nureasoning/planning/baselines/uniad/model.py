"""
UniAD baseline (placeholder).

UniAD ("Planning-oriented Autonomous Driving", CVPR 2023 best paper,
https://github.com/OpenDriveLab/UniAD) is a unified end-to-end framework that
jointly performs tracking, mapping, motion forecasting, occupancy prediction,
and planning from multi-view camera input.

This module is a placeholder that defines where a nuReasoning-adapted UniAD
planner plugs into the benchmark. To integrate UniAD:

  1. Install UniAD and its dependencies (mmcv/mmdet3d) following the official
     repository instructions, and obtain checkpoints trained (or fine-tuned)
     on nuReasoning multi-view images.
  2. Convert nuReasoning clip data to the model's expected input format:
     eight camera views + calibrations from ``metadata.json``, and the ego
     state / route command of the key frame.
  3. Implement ``UniADTrajectoryProvider.__call__`` to run inference at the
     key frame and return a global-frame ``(N, 3)`` array of ``(x, y, yaw)``
     waypoints at 0.1 s resolution over the 5 s horizon (see
     ``baselines.base.BaseTrajectoryProvider``). UniAD outputs ego-frame
     waypoints; convert them with the ego pose exactly as done in
     ``nureasoning.nuvla.trajectory_provider.VLATrajectoryProvider._ego_to_global``.
  4. Evaluate with ``python -m nureasoning.planning.benchmark`` on the validation split, or
     produce a challenge submission with ``python -m nureasoning.submission.challenge
     --planning-provider uniad``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import BaseTrajectoryProvider, register_baseline


@register_baseline("uniad")
class UniADTrajectoryProvider(BaseTrajectoryProvider):
    """Placeholder trajectory provider for the UniAD baseline."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        raise NotImplementedError(
            "UniAD is provided as a placeholder baseline. Follow the integration "
            "steps in nureasoning/planning/baselines/uniad/model.py to plug a trained UniAD model "
            "into the nuReasoning planning benchmark."
        )

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError

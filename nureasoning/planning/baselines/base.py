"""Common interface and registry for baseline planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

import numpy as np


class BaseTrajectoryProvider(ABC):
    """
    Interface shared by all baseline planners.

    A provider is a callable that, given a clip directory, the key-frame index,
    and the ego state at the key frame, returns the planned ego trajectory as a
    global-frame ``(N, 3)`` array of ``(x, y, yaw)`` waypoints sampled at
    ``dt_s`` from t=0 (inclusive) to ``horizon_s``.
    """

    #: planning horizon and sampling cadence expected by the benchmark
    horizon_s: float = 5.0
    dt_s: float = 0.1

    @abstractmethod
    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        """Return the planned global-frame trajectory for one clip."""

    @property
    def num_steps(self) -> int:
        return int(round(self.horizon_s / self.dt_s)) + 1


#: name -> provider class; extended by each baseline module at import time
BASELINE_REGISTRY: Dict[str, Callable[..., BaseTrajectoryProvider]] = {}


def register_baseline(name: str) -> Callable[[Type], Type]:
    def _decorator(cls: Type) -> Type:
        BASELINE_REGISTRY[name] = cls
        return cls

    return _decorator


def get_baseline(name: str, **kwargs: Any) -> BaseTrajectoryProvider:
    """Instantiate a registered baseline by name."""
    # Import placeholder modules lazily so that a missing optional dependency
    # in one baseline does not break the registry for the others.
    if name not in BASELINE_REGISTRY:
        if name == "uniad":
            from .uniad import model  # noqa: F401
        elif name == "diffusion_drive":
            from .diffusion_drive import model  # noqa: F401
    if name not in BASELINE_REGISTRY:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[name](**kwargs)

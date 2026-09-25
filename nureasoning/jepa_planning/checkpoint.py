"""Versioned checkpoints for the JEPA planning pipeline.

The visual backbone is identified by its immutable configuration/runtime
identity instead of being copied into every planning checkpoint.  All learned
planning modules, including the frozen trajectory target used after Stage A,
are saved explicitly and restored strictly.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn

from .config import STAGE_POLICIES, PlanningConfig, Stage, config_from_dict


CHECKPOINT_FORMAT = "nureasoning.jepa_planning"
CHECKPOINT_VERSION = 1
MODULE_NAMES = (
    "scene_adapter",
    "target_encoder",
    "trajectory_decoder",
    "intent_predictor",
    "action_expert",
    "command_embedding",
)


@dataclass(frozen=True)
class CheckpointState:
    """Training cursor and provenance restored from a checkpoint."""

    stage: Stage
    stage_complete: bool
    epoch: int
    next_batch_index: int
    global_step: int
    config: PlanningConfig
    path: str


def capture_rng_state() -> Dict[str, Any]:
    """Capture all RNGs used by the single-process training entrypoint."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "resume on the original device class"
            )
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the saved deterministic resume state"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _module(model: nn.Module, name: str) -> nn.Module:
    value = getattr(model, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"PlanningModel.{name} must be an nn.Module")
    return value


def _module_states(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        name: copy.deepcopy(_module(model, name).state_dict())
        for name in MODULE_NAMES
    }


def _backbone_identity(model: nn.Module, config: PlanningConfig) -> Dict[str, Any]:
    backbone = _module(model, "backbone")
    identity = getattr(backbone, "identity", None)
    if identity is None:
        runtime_identity: Any = {
            "model_class": f"{type(backbone).__module__}.{type(backbone).__qualname__}"
        }
    elif hasattr(identity, "to_dict"):
        runtime_identity = identity.to_dict()
    elif isinstance(identity, Mapping):
        runtime_identity = dict(identity)
    else:
        raise TypeError("backbone.identity must be a mapping or expose to_dict()")
    return {
        "config": copy.deepcopy(config.to_dict()["backbone"]),
        "runtime": runtime_identity,
    }


def _target_identity(model: nn.Module) -> Dict[str, Any]:
    encoder = _module(model, "target_encoder")
    return {
        "class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "num_waypoints": getattr(encoder, "num_waypoints", None),
        "latent_dim": getattr(encoder, "latent_dim", None),
        "hidden_dim": getattr(encoder, "hidden_dim", None),
        "is_pretrained": bool(getattr(encoder, "is_pretrained", False)),
    }


def _representation_metadata(config: PlanningConfig) -> Dict[str, Any]:
    return {
        "trajectory_representation": "current_ego_frame_cumulative_xy_heading",
        "trajectory_norm_scale": list(config.model.trajectory_norm_scale),
        "future_steps": config.data.future_steps,
        "future_dt_s": config.data.future_dt_s,
        "history_steps": config.data.history_steps,
        "history_dt_s": config.data.history_dt_s,
        "cameras": list(config.data.cameras),
        "camera_vocab": list(config.data.camera_vocab),
        "command_vocab": list(config.data.command_vocab),
        "num_video_frames": config.data.num_video_frames,
        "video_window_s": config.data.video_window_s,
        "frame_tolerance_s": config.data.frame_tolerance_s,
        "image_size": config.data.image_size,
        "loss_weights": copy.deepcopy(config.to_dict()["loss"]),
    }


def _module_metadata(model: nn.Module) -> Dict[str, Any]:
    action_expert = _module(model, "action_expert")
    action_config = getattr(action_expert, "config", None)
    if dataclasses.is_dataclass(action_config):
        serialized_action_config: Any = dataclasses.asdict(action_config)
    elif isinstance(action_config, Mapping):
        serialized_action_config = copy.deepcopy(dict(action_config))
    else:
        serialized_action_config = None
    return {
        "classes": {
            name: f"{type(_module(model, name)).__module__}."
            f"{type(_module(model, name)).__qualname__}"
            for name in MODULE_NAMES
        },
        "action_expert_config": serialized_action_config,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: PlanningConfig,
    stage: Stage | str,
    stage_complete: bool,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a complete, strict planning checkpoint."""

    stage = Stage(stage)
    if min(epoch, next_batch_index, global_step) < 0:
        raise ValueError("checkpoint training cursors must be non-negative")
    target_identity = _target_identity(model)
    if stage_complete and not target_identity["is_pretrained"]:
        raise RuntimeError("cannot complete a training stage with an unready target encoder")

    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": copy.deepcopy(config.to_dict()),
        "stage": stage.value,
        "stage_complete": bool(stage_complete),
        "training_state": {
            "epoch": int(epoch),
            "next_batch_index": int(next_batch_index),
            "global_step": int(global_step),
        },
        "modules": _module_states(model),
        "backbone_identity": _backbone_identity(model, config),
        "target_encoder_identity": target_identity,
        "module_metadata": _module_metadata(model),
        "representation": _representation_metadata(config),
        "freeze_policy": dataclasses.asdict(STAGE_POLICIES[stage]),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "extra": dict(extra or {}),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_payload(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("planning checkpoint must contain a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "not a JEPA-planning checkpoint; legacy Qwen/nuVLA checkpoints are "
            "not accepted by this loader"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version={payload.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    required = {
        "config",
        "stage",
        "stage_complete",
        "training_state",
        "modules",
        "backbone_identity",
        "target_encoder_identity",
        "module_metadata",
        "representation",
        "freeze_policy",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"planning checkpoint is incomplete: {sorted(missing)}")
    return payload


def _resume_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    training = dict(result.get("training", {}))
    # These do not alter optimization or tensor semantics and may be changed to
    # extend a run or relocate its artifacts.
    for name in ("epochs", "output_dir", "log_every"):
        training.pop(name, None)
    result["training"] = training
    return result


def _transition_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    # Stage transitions may choose a new optimizer/run length, but all data,
    # representation, backbone, model, and objective semantics stay identical.
    result.pop("training", None)
    result.pop("experiment_name", None)
    return result


def assert_config_compatible(
    saved: Mapping[str, Any],
    current: PlanningConfig,
    *,
    mode: str,
) -> None:
    if mode == "resume":
        saved_comparable = _resume_config(saved)
        current_comparable = _resume_config(current.to_dict())
    elif mode == "transition":
        saved_comparable = _transition_config(saved)
        current_comparable = _transition_config(current.to_dict())
    else:
        raise ValueError("compatibility mode must be 'resume' or 'transition'")
    if saved_comparable != current_comparable:
        raise ValueError(
            f"checkpoint configuration is incompatible with this {mode}; "
            "do not change preprocessing, vocabularies, normalization, model, "
            "backbone identity, loss definitions, or resume optimizer settings"
        )


def inspect_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> CheckpointState:
    payload = _load_payload(path, map_location=map_location)
    training = payload["training_state"]
    return CheckpointState(
        stage=Stage(payload["stage"]),
        stage_complete=bool(payload["stage_complete"]),
        epoch=int(training["epoch"]),
        next_batch_index=int(training["next_batch_index"]),
        global_step=int(training["global_step"]),
        config=config_from_dict(payload["config"]),
        path=str(path),
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: PlanningConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    compatibility: str = "resume",
    restore_rng: bool = True,
    map_location: Any = "cpu",
) -> CheckpointState:
    """Strictly restore model and optional lifecycle state.

    Passing an optimizer/scheduler/scaler requires that the corresponding state
    was saved.  Initializing a later stage normally omits those objects, while a
    same-stage resume restores all of them.
    """

    payload = _load_payload(path, map_location=map_location)
    assert_config_compatible(payload["config"], config, mode=compatibility)
    modules = payload["modules"]
    if set(modules) != set(MODULE_NAMES):
        raise ValueError(
            "checkpoint module inventory mismatch: "
            f"expected {list(MODULE_NAMES)}, got {sorted(modules)}"
        )
    current_backbone = _backbone_identity(model, config)
    if payload["backbone_identity"] != current_backbone:
        raise ValueError("checkpoint backbone identity does not match the loaded backbone")
    current_target = _target_identity(model)
    saved_target = payload["target_encoder_identity"]
    for name in ("class", "num_waypoints", "latent_dim", "hidden_dim"):
        if saved_target.get(name) != current_target.get(name):
            raise ValueError(f"checkpoint target encoder identity differs at {name}")
    if payload["module_metadata"] != _module_metadata(model):
        raise ValueError("checkpoint module identity/configuration does not match the model")
    saved_stage = Stage(payload["stage"])
    if payload["freeze_policy"] != dataclasses.asdict(STAGE_POLICIES[saved_stage]):
        raise ValueError("checkpoint frozen-module policy is incompatible")

    for name in MODULE_NAMES:
        _module(model, name).load_state_dict(modules[name], strict=True)

    lifecycle = (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    )
    for name, value in lifecycle:
        if value is not None:
            if payload.get(name) is None:
                raise ValueError(f"checkpoint does not contain required {name} state")
            value.load_state_dict(payload[name])
    if restore_rng:
        restore_rng_state(payload["rng_state"])

    training = payload["training_state"]
    return CheckpointState(
        stage=Stage(payload["stage"]),
        stage_complete=bool(payload["stage_complete"]),
        epoch=int(training["epoch"]),
        next_batch_index=int(training["next_batch_index"]),
        global_step=int(training["global_step"]),
        config=config_from_dict(payload["config"]),
        path=str(path),
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CheckpointState",
    "MODULE_NAMES",
    "assert_config_compatible",
    "capture_rng_state",
    "inspect_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]

"""Three-stage training entrypoint for LeVJEPA + Intent JEPA + DiT.

This module deliberately remains single-process for the M0--M2 smoke path.  It
does not download weights: real LeVJEPA construction is delegated to the local,
audited loader enforced by :mod:`nureasoning.jepa_planning.backbone`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .checkpoint import CheckpointState, inspect_checkpoint, load_checkpoint, save_checkpoint
from .config import PlanningConfig, Stage, load_config
from .contracts import TrainingBatch
from .data import PlanningDataset, assert_disjoint_planning_splits, planning_collate_fn


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochResult:
    metrics: Dict[str, float]
    global_step: int
    batches: int
    optimizer_steps: int
    gradient_norm: float


def seed_everything(seed: int) -> None:
    """Seed the single-process Python, NumPy, CPU, and CUDA RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_module(model: nn.Module, name: str) -> nn.Module:
    value = getattr(model, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"PlanningModel.{name} must be an nn.Module")
    return value


def _action_core_parameters(model: nn.Module) -> list[nn.Parameter]:
    action_expert = _require_module(model, "action_expert")
    state_encoder = _require_module(action_expert, "state_encoder")
    state_ids = {id(parameter) for parameter in state_encoder.parameters()}
    return [
        parameter
        for parameter in action_expert.parameters()
        if id(parameter) not in state_ids
    ]


def _stage_parameter_groups(
    model: nn.Module,
    stage: Stage,
) -> "OrderedDict[str, list[nn.Parameter]]":
    action_expert = _require_module(model, "action_expert")
    state_encoder = _require_module(action_expert, "state_encoder")
    if stage is Stage.TRAJECTORY_AE:
        modules = OrderedDict(
            (
                ("target_encoder", _require_module(model, "target_encoder")),
                ("trajectory_decoder", _require_module(model, "trajectory_decoder")),
            )
        )
    elif stage is Stage.INTENT:
        modules = OrderedDict(
            (
                ("scene_adapter", _require_module(model, "scene_adapter")),
                ("intent_predictor", _require_module(model, "intent_predictor")),
                ("action_expert.state_encoder", state_encoder),
                ("command_embedding", _require_module(model, "command_embedding")),
            )
        )
    else:
        modules = OrderedDict(
            (
                ("scene_adapter", _require_module(model, "scene_adapter")),
                ("intent_predictor", _require_module(model, "intent_predictor")),
                ("action_expert.state_encoder", state_encoder),
                ("command_embedding", _require_module(model, "command_embedding")),
            )
        )

    groups: "OrderedDict[str, list[nn.Parameter]]" = OrderedDict(
        (name, list(module.parameters())) for name, module in modules.items()
    )
    if stage is Stage.JOINT:
        groups["action_expert.core"] = _action_core_parameters(model)
    for name, parameters in groups.items():
        if not parameters:
            raise ValueError(f"trainable group {name!r} contains no parameters")
    return groups


def configure_stage(model: nn.Module, stage: Stage | str) -> Dict[str, list[nn.Parameter]]:
    """Apply and validate the exact frozen-module policy for one stage."""

    stage = Stage(stage)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    target = _require_module(model, "target_encoder")
    decoder = _require_module(model, "trajectory_decoder")
    backbone = _require_module(model, "backbone")
    action_expert = _require_module(model, "action_expert")
    state_encoder = _require_module(action_expert, "state_encoder")

    if stage is Stage.TRAJECTORY_AE:
        unfreeze = getattr(target, "unfreeze_for_pretraining", None)
        if callable(unfreeze):
            unfreeze()
        else:
            target.requires_grad_(True)
            target.train()
        decoder.requires_grad_(True)
        decoder.train()
    else:
        require_pretrained = getattr(target, "require_pretrained", None)
        if callable(require_pretrained):
            require_pretrained()
        elif not bool(getattr(target, "is_pretrained", False)):
            raise RuntimeError(
                "target encoder is not marked pretrained; load a completed Stage-A artifact"
            )
        freeze_target = getattr(model, "freeze_target_encoder", None)
        if callable(freeze_target):
            freeze_target()
        else:
            target.requires_grad_(False)
            target.eval()

    groups = _stage_parameter_groups(model, stage)
    for parameters in groups.values():
        for parameter in parameters:
            parameter.requires_grad_(True)

    if stage is Stage.INTENT:
        _require_module(model, "scene_adapter").train()
        _require_module(model, "intent_predictor").train()
        state_encoder.train()
        _require_module(model, "command_embedding").train()
        action_expert.eval()
        # action_expert.eval() recursively changed the shared state encoder.
        state_encoder.train()
    elif stage is Stage.JOINT:
        _require_module(model, "scene_adapter").train()
        _require_module(model, "intent_predictor").train()
        _require_module(model, "command_embedding").train()
        action_expert.train()

    backbone.requires_grad_(False)
    backbone.eval()
    if stage is not Stage.TRAJECTORY_AE:
        target.requires_grad_(False)
        target.eval()
        decoder.requires_grad_(False)
        decoder.eval()

    expected_ids: set[int] = set()
    for name, parameters in groups.items():
        for parameter in parameters:
            if id(parameter) in expected_ids:
                raise RuntimeError(f"parameter appears in duplicate optimizer group {name!r}")
            expected_ids.add(id(parameter))
    actual_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual_ids != expected_ids:
        raise RuntimeError(
            "stage freeze policy mismatch: optimizer groups do not exactly match "
            "requires_grad parameters"
        )
    return dict(groups)


def build_optimizer(
    model: nn.Module,
    config: PlanningConfig,
    stage: Stage | str,
) -> torch.optim.Optimizer:
    groups = configure_stage(model, stage)
    optimizer_groups = [
        {
            "name": name,
            "params": parameters,
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        }
        for name, parameters in groups.items()
    ]
    return torch.optim.AdamW(optimizer_groups)


def build_scheduler(optimizer: torch.optim.Optimizer) -> Any:
    """Build the explicit constant starter schedule saved in checkpoints."""

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def build_grad_scaler(device: torch.device, amp: bool) -> Any:
    enabled = bool(amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"AMP is unsupported on device type {device.type!r}")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _stage_losses(
    model: nn.Module,
    batch: TrainingBatch,
    stage: Stage,
) -> Mapping[str, torch.Tensor]:
    if stage is Stage.TRAJECTORY_AE:
        method = getattr(model, "compute_trajectory_autoencoder_losses", None)
        if not callable(method):
            raise TypeError("PlanningModel must expose compute_trajectory_autoencoder_losses")
        losses = method(batch.future)
    elif stage is Stage.INTENT:
        method = getattr(model, "compute_intent_losses", None)
        if not callable(method):
            raise TypeError("PlanningModel must expose compute_intent_losses")
        losses = method(batch)
    else:
        method = getattr(model, "compute_losses", None)
        if not callable(method):
            raise TypeError("PlanningModel must expose compute_losses")
        losses = method(batch, intent_mode="predicted")
    if not isinstance(losses, Mapping) or "loss" not in losses:
        raise TypeError("stage loss method must return a mapping containing 'loss'")
    return losses


def _finite_loss_mapping(losses: Mapping[str, Any]) -> None:
    loss = losses.get("loss")
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise TypeError("loss must be a scalar tensor")
    for name, value in losses.items():
        if torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite training value in {name!r}")


def _move_batch(batch: Any, device: torch.device) -> TrainingBatch:
    if not isinstance(batch, TrainingBatch):
        raise TypeError("training loader must yield TrainingBatch values")
    return batch.to(device)


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite gradient detected")
        squared += float(gradient.float().square().sum().item())
    return math.sqrt(squared)


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[TrainingBatch],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any,
    config: PlanningConfig,
    stage: Stage | str,
    device: torch.device | str,
    global_step: int = 0,
    start_batch_index: int = 0,
) -> EpochResult:
    """Train one epoch, correctly flushing a partial accumulation window."""

    stage = Stage(stage)
    device = torch.device(device)
    configure_stage(model, stage)
    accumulation = config.training.gradient_accumulation_steps
    trainable = _trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    batches = 0
    optimizer_steps = 0
    latest_gradient_norm = 0.0
    totals: Dict[str, float] = defaultdict(float)

    try:
        loader_length = len(loader)  # type: ignore[arg-type]
    except TypeError:
        loader_length = None
    if start_batch_index < 0 or (
        loader_length is not None and start_batch_index > loader_length
    ):
        raise ValueError("start_batch_index is outside the loader")

    for batch_index, batch_value in enumerate(loader):
        if batch_index < start_batch_index:
            continue
        batch = _move_batch(batch_value, device)
        with _autocast(device, config.training.amp):
            losses = _stage_losses(model, batch, stage)
            _finite_loss_mapping(losses)
            scaled_loss = losses["loss"] / accumulation
        scaler.scale(scaled_loss).backward()
        pending += 1
        batches += 1
        for name, value in losses.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[name] += float(value.detach().float().item())

        last_batch = loader_length is not None and batch_index + 1 == loader_length
        if pending == accumulation or last_batch:
            scaler.unscale_(optimizer)
            if pending < accumulation:
                correction = accumulation / pending
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if config.training.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    trainable, config.training.max_grad_norm, error_if_nonfinite=True
                )
            latest_gradient_norm = _gradient_norm(trainable)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            global_step += 1
            optimizer_steps += 1

    # Iterable loaders without __len__ cannot announce their last batch.
    if pending:
        scaler.unscale_(optimizer)
        correction = accumulation / pending
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if config.training.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                trainable, config.training.max_grad_norm, error_if_nonfinite=True
            )
        latest_gradient_norm = _gradient_norm(trainable)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        optimizer_steps += 1

    if batches == 0:
        raise ValueError("training loader yielded no batches")
    metrics = {name: total / batches for name, total in totals.items()}
    metrics["gradient_norm"] = latest_gradient_norm
    return EpochResult(
        metrics=metrics,
        global_step=global_step,
        batches=batches,
        optimizer_steps=optimizer_steps,
        gradient_norm=latest_gradient_norm,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[TrainingBatch],
    *,
    config: PlanningConfig,
    stage: Stage | str,
    device: torch.device | str,
) -> Dict[str, float]:
    stage = Stage(stage)
    device = torch.device(device)
    model.eval()
    totals: Dict[str, float] = defaultdict(float)
    batches = 0
    for batch_value in loader:
        batch = _move_batch(batch_value, device)
        with _autocast(device, config.training.amp):
            losses = _stage_losses(model, batch, stage)
        _finite_loss_mapping(losses)
        for name, value in losses.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[name] += float(value.detach().float().item())
        batches += 1
    configure_stage(model, stage)
    if batches == 0:
        raise ValueError("validation loader yielded no batches")
    return {name: total / batches for name, total in totals.items()}


def _make_loader(
    dataset: Dataset[Any],
    config: PlanningConfig,
    *,
    training: bool,
    epoch: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(config.training.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=(
            config.training.batch_size if training else config.training.val_batch_size
        ),
        shuffle=training,
        num_workers=config.training.num_workers,
        collate_fn=planning_collate_fn,
        generator=generator,
        drop_last=False,
    )


def _load_stage_source(
    model: nn.Module,
    config: PlanningConfig,
    stage: Stage,
    *,
    init_checkpoint: str | Path | None,
    resume: str | Path | None,
    device: torch.device,
) -> CheckpointState | None:
    if init_checkpoint is not None and resume is not None:
        raise ValueError("use either init_checkpoint or resume, not both")
    if resume is not None:
        state = inspect_checkpoint(resume, map_location="cpu")
        if state.stage is not stage:
            raise ValueError(
                f"resume checkpoint stage is {state.stage.value!r}, expected {stage.value!r}"
            )
        if state.stage_complete:
            raise ValueError("cannot resume a checkpoint whose stage is already complete")
        return load_checkpoint(
            resume,
            model=model,
            config=config,
            compatibility="resume",
            restore_rng=False,
            map_location=device,
        )

    prerequisites = {
        Stage.INTENT: Stage.TRAJECTORY_AE,
        Stage.JOINT: Stage.INTENT,
    }
    if stage is Stage.TRAJECTORY_AE:
        if init_checkpoint is not None:
            raise ValueError("trajectory_ae starts fresh or resumes the same stage")
        return None
    if init_checkpoint is None:
        raise ValueError(
            f"stage {stage.value!r} requires --init-checkpoint from a completed "
            f"{prerequisites[stage].value!r} stage"
        )
    state = inspect_checkpoint(init_checkpoint, map_location="cpu")
    if state.stage is not prerequisites[stage] or not state.stage_complete:
        raise ValueError(
            f"stage {stage.value!r} requires a completed "
            f"{prerequisites[stage].value!r} checkpoint"
        )
    load_checkpoint(
        init_checkpoint,
        model=model,
        config=config,
        compatibility="transition",
        restore_rng=False,
        map_location=device,
    )
    return None


def run_training(
    model: nn.Module,
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any] | None,
    *,
    config: PlanningConfig,
    stage: Stage | str,
    device: torch.device | str,
    init_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> CheckpointState:
    """Run one explicitly requested stage and return its completed artifact."""

    stage = Stage(stage)
    config.validate(require_real_backbone=not config.backbone.use_stub)
    device = torch.device(device)
    seed_everything(config.training.seed)
    model.to(device)
    resume_state = _load_stage_source(
        model,
        config,
        stage,
        init_checkpoint=init_checkpoint,
        resume=resume,
        device=device,
    )
    optimizer = build_optimizer(model, config, stage)
    scheduler = build_scheduler(optimizer)
    scaler = build_grad_scaler(device, config.training.amp)

    start_epoch = 0
    start_batch_index = 0
    global_step = 0
    if resume is not None:
        restored = load_checkpoint(
            resume,
            model=model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            compatibility="resume",
            restore_rng=True,
            map_location=device,
        )
        start_epoch = restored.epoch
        start_batch_index = restored.next_batch_index
        global_step = restored.global_step
    elif resume_state is not None:
        raise AssertionError("resume state was not restored")

    if start_epoch >= config.training.epochs:
        raise ValueError(
            f"checkpoint is already at epoch {start_epoch}, but configured epochs="
            f"{config.training.epochs}"
        )
    destination = Path(
        checkpoint_path
        or Path(config.training.output_dir) / f"{stage.value}_last.pt"
    )

    for epoch in range(start_epoch, config.training.epochs):
        train_loader = _make_loader(train_dataset, config, training=True, epoch=epoch)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            stage=stage,
            device=device,
            global_step=global_step,
            start_batch_index=start_batch_index if epoch == start_epoch else 0,
        )
        global_step = train_result.global_step
        validation_metrics: Dict[str, float] = {}
        if val_dataset is not None:
            validation_metrics = validate_one_epoch(
                model,
                _make_loader(val_dataset, config, training=False, epoch=epoch),
                config=config,
                stage=stage,
                device=device,
            )

        complete = epoch + 1 == config.training.epochs
        if complete and stage is Stage.TRAJECTORY_AE:
            mark_pretrained = getattr(
                _require_module(model, "target_encoder"), "mark_pretrained", None
            )
            if not callable(mark_pretrained):
                raise TypeError("target encoder must expose mark_pretrained()")
            mark_pretrained()
        save_checkpoint(
            destination,
            model=model,
            config=config,
            stage=stage,
            stage_complete=complete,
            epoch=epoch + 1,
            next_batch_index=0,
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            extra={
                "train_metrics": train_result.metrics,
                "validation_metrics": validation_metrics,
            },
        )
        logger.info(
            "stage=%s epoch=%d train=%s validation=%s checkpoint=%s",
            stage.value,
            epoch + 1,
            train_result.metrics,
            validation_metrics,
            destination,
        )
        start_batch_index = 0

    return inspect_checkpoint(destination, map_location="cpu")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="planning YAML configuration")
    parser.add_argument("--stage", required=True, choices=[stage.value for stage in Stage])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-checkpoint",
        help="completed prerequisite-stage checkpoint (required for intent/joint)",
    )
    parser.add_argument("--resume", help="incomplete same-stage checkpoint")
    parser.add_argument("--checkpoint-path", help="override the stage artifact path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    seed_everything(config.training.seed)

    # Imports remain local so ``--help`` and checkpoint inspection never load a
    # model implementation or optional real-backbone dependencies.
    from .model import build_planning_model

    model = build_planning_model(config)
    train_dataset = PlanningDataset(config.data, split="train", observation_only=False)
    val_dataset = PlanningDataset(config.data, split="val", observation_only=False)
    assert_disjoint_planning_splits(train_dataset, val_dataset)
    if len(train_dataset) == 0:
        raise RuntimeError(f"no eligible training samples; coverage={train_dataset.coverage}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"no eligible validation samples; coverage={val_dataset.coverage}")

    state = run_training(
        model,
        train_dataset,
        val_dataset,
        config=config,
        stage=Stage(args.stage),
        device=args.device,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
    )
    print(
        json.dumps(
            {
                "checkpoint": state.path,
                "stage": state.stage.value,
                "stage_complete": state.stage_complete,
                "epoch": state.epoch,
                "global_step": state.global_step,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "EpochResult",
    "build_grad_scaler",
    "build_optimizer",
    "build_scheduler",
    "configure_stage",
    "main",
    "run_training",
    "seed_everything",
    "train_one_epoch",
    "validate_one_epoch",
]

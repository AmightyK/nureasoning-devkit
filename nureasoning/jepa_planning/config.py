"""Serializable, validated configuration for the JEPA planning pipeline."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Type, TypeVar

import yaml


SCHEMA_VERSION = 1
CAMERA_VOCAB: Tuple[str, ...] = (
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "back",
    "back_left",
    "back_right",
)
COMMAND_VOCAB: Tuple[str, ...] = (
    "UNKNOWN",
    "LANE_FOLLOW",
    "STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LEFT_LANE_CHANGE",
    "RIGHT_LANE_CHANGE",
    "STOP",
)


class Stage(str, Enum):
    TRAJECTORY_AE = "trajectory_ae"
    INTENT = "intent"
    JOINT = "joint"


@dataclass(frozen=True)
class FrozenModulePolicy:
    """Exact module policy used by the three supported training stages."""

    backbone: bool
    target_encoder: bool
    trajectory_decoder: bool
    scene_adapter: bool
    intent_predictor: bool
    action_expert_core: bool
    state_encoder: bool
    command_embedding: bool


STAGE_POLICIES: Dict[Stage, FrozenModulePolicy] = {
    Stage.TRAJECTORY_AE: FrozenModulePolicy(
        backbone=True,
        target_encoder=False,
        trajectory_decoder=False,
        scene_adapter=True,
        intent_predictor=True,
        action_expert_core=True,
        state_encoder=True,
        command_embedding=True,
    ),
    Stage.INTENT: FrozenModulePolicy(
        backbone=True,
        target_encoder=True,
        trajectory_decoder=True,
        scene_adapter=False,
        intent_predictor=False,
        action_expert_core=True,
        state_encoder=False,  # supplies the predictor state token
        command_embedding=False,
    ),
    Stage.JOINT: FrozenModulePolicy(
        backbone=True,
        target_encoder=True,
        trajectory_decoder=True,
        scene_adapter=False,
        intent_predictor=False,
        action_expert_core=False,
        state_encoder=False,
        command_embedding=False,
    ),
}


@dataclass
class DataConfig:
    train_root: str = "./dataset/data/train"
    val_root: str = "./dataset/data/validation"
    cameras: Tuple[str, ...] = ("front",)
    camera_vocab: Tuple[str, ...] = CAMERA_VOCAB
    command_vocab: Tuple[str, ...] = COMMAND_VOCAB
    image_size: int = 224
    num_video_frames: int = 16
    video_window_s: float = 2.0
    frame_tolerance_s: float = 0.075
    history_steps: int = 6
    history_dt_s: float = 0.5
    future_steps: int = 10
    future_dt_s: float = 0.5
    train_clip_fraction: float = 1.0
    clip_seed: int = 42
    require_complete_inputs: bool = True


@dataclass
class BackboneConfig:
    model_id: str = "galilai-group/LeVJEPA-VideoMix-Large"
    revision: str | None = None
    local_path: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False
    allow_audited_local_code: bool = False
    code_revision: str | None = None
    expected_feature_dim: int = 1024
    expected_has_cls_token: bool = True
    camera_chunk_size: int = 1
    frozen: bool = True
    use_stub: bool = False
    stub_patch_tokens: int = 17


@dataclass
class ModelConfig:
    context_dim: int = 512
    scene_tokens: int = 64
    intent_layers: int = 4
    intent_heads: int = 8
    intent_dropout: float = 0.1
    trajectory_layers: int = 3
    trajectory_hidden_dim: int = 512
    action_layers: int = 12
    action_heads: int = 8
    action_dropout: float = 0.1
    action_mlp_ratio: float = 4.0
    action_interleave_self_attention: bool = True
    num_inference_steps: int = 10
    trajectory_norm_scale: Tuple[float, float, float] = (50.0, 20.0, math.pi)


@dataclass
class LossConfig:
    reconstruction_position: float = 1.0
    reconstruction_heading: float = 1.0
    reconstruction_motion: float = 0.25
    feature: float = 1.0
    token_cosine: float = 1.0
    info_nce: float = 0.1
    info_nce_temperature: float = 0.1
    lambda_jepa: float = 1.0


@dataclass
class TrainingConfig:
    batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    amp: bool = False
    seed: int = 42
    log_every: int = 10
    output_dir: str = "./outputs/jepa_planning"


@dataclass
class PlanningConfig:
    schema_version: int = SCHEMA_VERSION
    experiment_name: str = "levjepa_intent_dit"
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self, *, require_real_backbone: bool = False) -> "PlanningConfig":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.data.cameras or len(set(self.data.cameras)) != len(self.data.cameras):
            raise ValueError("data.cameras must be a non-empty ordered set")
        unknown_cameras = set(self.data.cameras) - set(self.data.camera_vocab)
        if unknown_cameras:
            raise ValueError(f"cameras missing from camera_vocab: {sorted(unknown_cameras)}")
        if not self.data.command_vocab or self.data.command_vocab[0] != "UNKNOWN":
            raise ValueError("command_vocab[0] must be the explicit UNKNOWN command")
        if len(set(self.data.command_vocab)) != len(self.data.command_vocab):
            raise ValueError("command_vocab values must be unique")
        if self.data.image_size != 224:
            raise ValueError("LeVJEPA MVP requires 224x224 video frames")
        if self.data.num_video_frames != 16:
            raise ValueError("LeVJEPA MVP requires exactly 16 chronological frames")
        if self.data.history_steps != 6 or self.data.future_steps != 10:
            raise ValueError("contract requires 6 history and 10 future poses")
        if self.data.history_dt_s != 0.5 or self.data.future_dt_s != 0.5:
            raise ValueError("history/future poses must use the fixed 0.5 s grid")
        if not 0.0 < self.data.train_clip_fraction <= 1.0:
            raise ValueError("train_clip_fraction must be in (0, 1]")
        if self.data.frame_tolerance_s < 0.0:
            raise ValueError("frame_tolerance_s must be non-negative")
        if self.model.context_dim <= 0 or self.model.scene_tokens <= 0:
            raise ValueError("context_dim and scene_tokens must be positive")
        if self.model.context_dim % self.model.intent_heads:
            raise ValueError("context_dim must be divisible by intent_heads")
        if self.model.context_dim % self.model.action_heads:
            raise ValueError("context_dim must be divisible by action_heads")
        if len(self.model.trajectory_norm_scale) != 3 or any(
            value <= 0.0 for value in self.model.trajectory_norm_scale
        ):
            raise ValueError("trajectory_norm_scale must contain three positive values")
        if self.backbone.trust_remote_code:
            raise ValueError(
                "automatic remote model code is prohibited; audit and install it locally"
            )
        if self.backbone.allow_audited_local_code and (
            not self.backbone.local_path
            or not self.backbone.revision
            or not self.backbone.code_revision
        ):
            raise ValueError(
                "audited local code requires local_path plus immutable model and code revisions"
            )
        if not self.backbone.frozen:
            raise ValueError("M0-M2 policy requires the LeVJEPA backbone to remain frozen")
        if require_real_backbone and not self.backbone.use_stub:
            if not self.backbone.local_path or not self.backbone.revision:
                raise ValueError(
                    "real LeVJEPA runs require a local snapshot and immutable audited revision"
                )
        for name, value in dataclasses.asdict(self.loss).items():
            if value < 0.0:
                raise ValueError(f"loss.{name} must be non-negative")
        if self.loss.info_nce and self.loss.info_nce_temperature <= 0.0:
            raise ValueError("info_nce_temperature must be positive")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least one")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


ConfigType = TypeVar("ConfigType")


def _construct(config_type: Type[ConfigType], values: Mapping[str, Any]) -> ConfigType:
    allowed = {item.name for item in dataclasses.fields(config_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {config_type.__name__} fields: {sorted(unknown)}")
    converted = dict(values)
    for item in dataclasses.fields(config_type):
        if item.name in converted and item.name in {
            "cameras", "camera_vocab", "command_vocab", "trajectory_norm_scale"
        }:
            converted[item.name] = tuple(converted[item.name])
    return config_type(**converted)


def config_from_dict(values: Mapping[str, Any]) -> PlanningConfig:
    allowed = {item.name for item in dataclasses.fields(PlanningConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown PlanningConfig fields: {sorted(unknown)}")
    config = PlanningConfig(
        schema_version=int(values.get("schema_version", SCHEMA_VERSION)),
        experiment_name=str(values.get("experiment_name", "levjepa_intent_dit")),
        data=_construct(DataConfig, values.get("data", {})),
        backbone=_construct(BackboneConfig, values.get("backbone", {})),
        model=_construct(ModelConfig, values.get("model", {})),
        loss=_construct(LossConfig, values.get("loss", {})),
        training=_construct(TrainingConfig, values.get("training", {})),
    )
    return config.validate()


def load_config(path: str | Path) -> PlanningConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise ValueError("planning config must contain a YAML mapping")
    return config_from_dict(values)


def stage_policy(stage: Stage | str) -> FrozenModulePolicy:
    return STAGE_POLICIES[Stage(stage)]

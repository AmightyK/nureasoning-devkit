"""Composition of LeVJEPA, future-intent JEPA, and the nuVLA action expert.

The model keeps the three trajectory uses deliberately separate:

* the trajectory autoencoder consumes one explicitly normalized future;
* the frozen target encoder consumes one explicitly normalized future;
* the unchanged action expert consumes the raw future and performs its own
  normalization internally.

Only observation-derived predicted intent can enter the DiT condition.  Target
intent is constructed in a private, no-gradient supervision path and cannot be
supplied to :meth:`predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import torch
from torch import nn

from nureasoning.jepa_planning.backbone import (
    LeVJEPABackbone,
    SceneAdapter,
    build_backbone,
)
from nureasoning.jepa_planning.config import PlanningConfig
from nureasoning.jepa_planning.contracts import ObservationBatch, TrainingBatch
from nureasoning.jepa_planning.intent import IntentPredictor
from nureasoning.jepa_planning.losses import (
    jepa_loss,
    trajectory_reconstruction_loss,
)
from nureasoning.jepa_planning.trajectory import (
    TargetTrajectoryEncoder,
    TrajectoryDecoder,
    normalize_trajectory,
)
from nureasoning.nuvla.models.action_expert import (
    ActionExpertConfig,
    FlowMatchingDiTActionExpert,
)


IntentMode = Literal["predicted", "no_intent", "shuffled"]


@dataclass(frozen=True)
class PlanningConditioning:
    """Observation-derived intermediate tensors reused by the planner.

    ``predicted_intent`` always contains the predictor output, including for an
    ablation.  ``dit_context`` contains exactly the memory seen by the action
    expert after applying ``intent_mode``.
    """

    scene_tokens: torch.Tensor
    state_token: torch.Tensor
    command_token: torch.Tensor
    predicted_intent: torch.Tensor
    dit_context: torch.Tensor
    intent_mode: IntentMode


def _prefixed(
    values: Dict[str, torch.Tensor],
    prefix: str,
    *,
    skip: tuple[str, ...] = ("loss",),
) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}{name}": value
        for name, value in values.items()
        if name not in skip
    }


class PlanningModel(nn.Module):
    """LeVJEPA + predicted-intent JEPA + flow-matching DiT planner."""

    def __init__(
        self,
        config: PlanningConfig,
        *,
        backbone: LeVJEPABackbone | None = None,
        scene_adapter: SceneAdapter | None = None,
        target_encoder: TargetTrajectoryEncoder | None = None,
        trajectory_decoder: TrajectoryDecoder | None = None,
        intent_predictor: IntentPredictor | None = None,
        action_expert: FlowMatchingDiTActionExpert | None = None,
    ) -> None:
        super().__init__()
        self.config = config.validate()
        data = self.config.data
        model = self.config.model

        self.backbone = backbone or build_backbone(self.config.backbone)
        self.scene_adapter = scene_adapter or SceneAdapter(
            feature_dim=self.config.backbone.expected_feature_dim,
            context_dim=model.context_dim,
            num_scene_tokens=model.scene_tokens,
            # IDs are stable vocabulary IDs, rather than positions in the
            # selected-camera subset.
            num_cameras=len(data.camera_vocab),
            num_frames=data.num_video_frames,
            num_heads=model.intent_heads,
            dropout=model.intent_dropout,
        )

        trajectory_kwargs = dict(
            num_waypoints=data.future_steps,
            trajectory_dim=3,
            latent_dim=model.context_dim,
            hidden_dim=model.trajectory_hidden_dim,
            num_layers=model.trajectory_layers,
            num_heads=model.intent_heads,
            dropout=model.intent_dropout,
        )
        self.target_encoder = target_encoder or TargetTrajectoryEncoder(
            **trajectory_kwargs
        )
        self.trajectory_decoder = trajectory_decoder or TrajectoryDecoder(
            **trajectory_kwargs
        )
        self.intent_predictor = intent_predictor or IntentPredictor(
            context_dim=model.context_dim,
            num_waypoints=data.future_steps,
            num_layers=model.intent_layers,
            num_heads=model.intent_heads,
            dropout=model.intent_dropout,
        )
        self.command_embedding = nn.Embedding(
            len(data.command_vocab), model.context_dim
        )

        action_config = ActionExpertConfig(
            vlm_feature_dim=model.context_dim,
            ego_state_dim=4,
            max_history_traj_points=data.history_steps,
            history_traj_dim=3,
            num_waypoints=data.future_steps,
            trajectory_dim=3,
            hidden_dim=model.context_dim,
            num_heads=model.action_heads,
            num_dit_layers=model.action_layers,
            dropout=model.action_dropout,
            mlp_ratio=model.action_mlp_ratio,
            interleave_self_attention=model.action_interleave_self_attention,
            num_inference_steps=model.num_inference_steps,
            trajectory_norm_scale=tuple(model.trajectory_norm_scale),
        )
        self.action_expert = action_expert or FlowMatchingDiTActionExpert(
            action_config
        )
        self._validate_composition()

    def _validate_composition(self) -> None:
        model = self.config.model
        data = self.config.data
        if self.target_encoder.num_waypoints != data.future_steps:
            raise ValueError("target encoder waypoint count disagrees with config")
        if self.target_encoder.latent_dim != model.context_dim:
            raise ValueError("target encoder latent width disagrees with context_dim")
        if self.trajectory_decoder.num_waypoints != data.future_steps:
            raise ValueError("trajectory decoder waypoint count disagrees with config")
        if self.trajectory_decoder.latent_dim != model.context_dim:
            raise ValueError("trajectory decoder latent width disagrees with context_dim")
        if self.intent_predictor.num_waypoints != data.future_steps:
            raise ValueError("intent predictor waypoint count disagrees with config")
        if self.intent_predictor.context_dim != model.context_dim:
            raise ValueError("intent predictor width disagrees with context_dim")

        expert_config = self.action_expert.config
        expected = {
            "vlm_feature_dim": model.context_dim,
            "hidden_dim": model.context_dim,
            "num_waypoints": data.future_steps,
            "trajectory_dim": 3,
            "max_history_traj_points": data.history_steps,
        }
        for name, value in expected.items():
            if getattr(expert_config, name) != value:
                raise ValueError(
                    f"action expert {name}={getattr(expert_config, name)!r}; "
                    f"expected {value!r}"
                )
        if tuple(expert_config.trajectory_norm_scale) != tuple(
            model.trajectory_norm_scale
        ):
            raise ValueError(
                "action expert and target-encoder trajectory normalization "
                "must use the same scale"
            )

    def _validate_observations(self, batch: ObservationBatch) -> None:
        batch.validate(
            num_cameras=len(self.config.data.cameras),
            num_frames=self.config.data.num_video_frames,
            image_size=self.config.data.image_size,
            num_history=self.config.data.history_steps,
            camera_vocab_size=len(self.config.data.camera_vocab),
            command_vocab_size=len(self.config.data.command_vocab),
        )

    def _validate_training_batch(self, batch: TrainingBatch) -> None:
        batch.validate(
            num_cameras=len(self.config.data.cameras),
            num_frames=self.config.data.num_video_frames,
            image_size=self.config.data.image_size,
            num_history=self.config.data.history_steps,
            num_waypoints=self.config.data.future_steps,
            camera_vocab_size=len(self.config.data.camera_vocab),
            command_vocab_size=len(self.config.data.command_vocab),
        )

    def _require_frozen_pretrained_target(self) -> None:
        self.target_encoder.require_pretrained()
        if not self.target_encoder.is_frozen:
            raise RuntimeError(
                "trajectory target encoder must be frozen before intent or "
                "joint training"
            )
        if self.target_encoder.training:
            raise RuntimeError("frozen trajectory target encoder must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.target_encoder.parameters()):
            raise RuntimeError("frozen trajectory target encoder has trainable parameters")

    def freeze_target_encoder(self) -> TargetTrajectoryEncoder:
        """Freeze an explicitly pretrained Stage-A target representation."""

        return self.target_encoder.freeze()

    @staticmethod
    def _validate_intent_mode(intent_mode: str) -> IntentMode:
        if intent_mode not in ("predicted", "no_intent", "shuffled"):
            raise ValueError(
                "intent_mode must be one of 'predicted', 'no_intent', or 'shuffled'"
            )
        return intent_mode  # type: ignore[return-value]

    def encode_observations(
        self,
        batch: ObservationBatch,
        *,
        intent_mode: IntentMode = "predicted",
    ) -> PlanningConditioning:
        """Compute observation context and predicted intent exactly once.

        No future tensor is accepted by this interface.  ``shuffled`` is a
        validation diagnostic and is rejected in training mode.
        """

        self._validate_observations(batch)
        mode = self._validate_intent_mode(intent_mode)
        if mode == "shuffled" and self.training:
            raise RuntimeError("shuffled intent is a validation-only ablation")

        patch_tokens = self.backbone(batch.video)
        scene_tokens = self.scene_adapter(
            patch_tokens,
            batch.camera_ids,
            batch.frame_times_s,
        )
        # The state encoder is registered only under action_expert.  Reusing it
        # here preserves one parameter set and one checkpoint key namespace.
        state_token = self.action_expert.state_encoder(
            batch.ego_state,
            batch.history,
        )
        command_token = self.command_embedding(batch.command_id).unsqueeze(1)
        predicted_intent = self.intent_predictor(
            scene_tokens,
            state_token,
            command_token,
        )

        context_parts = [scene_tokens, command_token]
        if mode == "predicted":
            context_parts.append(predicted_intent)
        elif mode == "shuffled":
            if batch.batch_size < 2:
                raise ValueError("shuffled intent requires a batch of at least two")
            context_parts.append(predicted_intent.roll(shifts=1, dims=0))
        # no_intent deliberately removes the predicted tokens rather than
        # replacing them with an unmarked learned or zero-valued condition.
        dit_context = torch.cat(context_parts, dim=1)
        return PlanningConditioning(
            scene_tokens=scene_tokens,
            state_token=state_token,
            command_token=command_token,
            predicted_intent=predicted_intent,
            dit_context=dit_context,
            intent_mode=mode,
        )

    def _target_intent(self, raw_future: torch.Tensor) -> torch.Tensor:
        normalized_future = normalize_trajectory(
            raw_future,
            self.config.model.trajectory_norm_scale,
        )
        with torch.no_grad():
            return self.target_encoder(normalized_future)

    def compute_trajectory_autoencoder_losses(
        self,
        raw_future: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage-A reconstruction objective from raw ego-frame poses."""

        normalized_future = normalize_trajectory(
            raw_future,
            self.config.model.trajectory_norm_scale,
        )
        reconstructed = self.trajectory_decoder(
            self.target_encoder(normalized_future)
        )
        values = trajectory_reconstruction_loss(
            reconstructed,
            normalized_future,
            scale=self.config.model.trajectory_norm_scale,
            dt_s=self.config.data.future_dt_s,
            position_weight=self.config.loss.reconstruction_position,
            heading_weight=self.config.loss.reconstruction_heading,
            motion_weight=self.config.loss.reconstruction_motion,
        )
        return {"loss": values["loss"], "total_loss": values["loss"], **values}

    def _jepa_losses(
        self,
        predicted_intent: torch.Tensor,
        raw_future: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._require_frozen_pretrained_target()
        target_intent = self._target_intent(raw_future)
        return jepa_loss(
            predicted_intent,
            target_intent,
            feature_weight=self.config.loss.feature,
            token_cosine_weight=self.config.loss.token_cosine,
            info_nce_weight=self.config.loss.info_nce,
            temperature=self.config.loss.info_nce_temperature,
        )

    def compute_intent_losses(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, torch.Tensor]:
        """Stage-B JEPA objective without executing the DiT."""

        self._validate_training_batch(batch)
        self._require_frozen_pretrained_target()
        conditioning = self.encode_observations(batch.observations())
        values = self._jepa_losses(conditioning.predicted_intent, batch.future)
        return {"loss": values["loss"], "total_loss": values["loss"], **values}

    def compute_losses(
        self,
        batch: TrainingBatch,
        *,
        intent_mode: IntentMode = "predicted",
    ) -> Dict[str, torch.Tensor]:
        """Stage-C joint flow and JEPA losses.

        ``batch.future`` is passed raw to the unchanged action expert.  A
        separately normalized tensor is created only for target supervision.
        """

        self._validate_training_batch(batch)
        self._require_frozen_pretrained_target()
        conditioning = self.encode_observations(
            batch.observations(),
            intent_mode=intent_mode,
        )
        flow = self.action_expert(
            x_1=batch.future,
            vlm_features=conditioning.dit_context,
            ego_state=batch.ego_state,
            history_trajectory=batch.history,
        )
        jepa = self._jepa_losses(conditioning.predicted_intent, batch.future)
        flow_loss = flow["loss"]
        latent_loss = jepa["loss"]
        total = flow_loss + self.config.loss.lambda_jepa * latent_loss
        return {
            "loss": total,
            "total_loss": total,
            "flow_loss": flow_loss,
            "jepa_loss": latent_loss,
            **_prefixed(flow, "flow_"),
            **_prefixed(jepa, "jepa_", skip=("loss", "jepa_loss")),
        }

    @torch.no_grad()
    def predict(
        self,
        batch: ObservationBatch,
        num_steps: int | None = None,
        *,
        seed: int | None = None,
        intent_mode: IntentMode = "predicted",
    ) -> torch.Tensor:
        """Generate raw ego-frame poses using observations and pure noise only."""

        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        conditioning = self.encode_observations(batch, intent_mode=intent_mode)

        def sample() -> torch.Tensor:
            return self.action_expert.sample(
                vlm_features=conditioning.dit_context,
                ego_state=batch.ego_state,
                history_trajectory=batch.history,
                num_steps=num_steps,
            )

        if seed is None:
            return sample()

        cuda_devices: list[int] = []
        if batch.video.device.type == "cuda":
            device_index = batch.video.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices.append(device_index)
        # The unchanged action expert does not accept a Generator.  fork_rng
        # makes seeded sampling deterministic without mutating caller RNG state.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return sample()


def build_planning_model(
    config: PlanningConfig,
    **injected_modules: nn.Module,
) -> PlanningModel:
    """Build the planner; keyword injections are intended for local tests."""

    return PlanningModel(config, **injected_modules)


__all__ = [
    "IntentMode",
    "PlanningConditioning",
    "PlanningModel",
    "build_planning_model",
]

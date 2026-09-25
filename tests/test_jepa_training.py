"""Synthetic CPU tests for JEPA planning training/checkpoint lifecycle."""

from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nureasoning.jepa_planning.checkpoint import (
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from nureasoning.jepa_planning.config import PlanningConfig, Stage
from nureasoning.jepa_planning.contracts import ObservationBatch, TrainingBatch
from nureasoning.jepa_planning.data import PlanningSample
from nureasoning.jepa_planning.train import (
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    configure_stage,
    run_training,
    seed_everything,
    train_one_epoch,
    validate_one_epoch,
)


class TinyTargetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1)
        self.num_waypoints = 10
        self.latent_dim = 1
        self.hidden_dim = 1
        self.register_buffer("_pretrained_ready", torch.tensor(False))
        self._frozen = False

    @property
    def is_pretrained(self) -> bool:
        return bool(self._pretrained_ready.item())

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def mark_pretrained(self):
        self._pretrained_ready.fill_(True)
        return self

    def require_pretrained(self) -> None:
        if not self.is_pretrained:
            raise RuntimeError("target is not pretrained")

    def freeze(self):
        self.require_pretrained()
        self.requires_grad_(False)
        self._frozen = True
        super().train(False)
        return self

    def unfreeze_for_pretraining(self):
        self.requires_grad_(True)
        self._frozen = False
        super().train(True)
        return self

    def train(self, mode: bool = True):
        return super().train(False if self._frozen else mode)

    def forward(self, future: torch.Tensor) -> torch.Tensor:
        return self.projection(future).mean(dim=1)


class TinyActionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(22, 1)
        self.core = nn.Linear(1, 1)
        self.config = {"kind": "tiny-test-expert", "normalization": [50.0, 20.0, math.pi]}


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("constant", torch.tensor(1.0))
        self.identity = {"model_id": "synthetic", "revision": "fixture-v1"}


class TinyPlanningModel(nn.Module):
    """Trainer-facing PlanningModel mock with every real public attribute."""

    def __init__(self, config: PlanningConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyBackbone()
        self.scene_adapter = nn.Linear(1, 1)
        self.target_encoder = TinyTargetEncoder()
        self.trajectory_decoder = nn.Linear(1, 1)
        self.intent_predictor = nn.Linear(1, 1)
        self.action_expert = TinyActionExpert()
        self.command_embedding = nn.Embedding(len(config.data.command_vocab), 1)

    def freeze_target_encoder(self):
        return self.target_encoder.freeze()

    def compute_trajectory_autoencoder_losses(self, raw_future: torch.Tensor):
        latent = self.target_encoder(raw_future)
        reconstruction = self.trajectory_decoder(latent)
        target = raw_future.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        loss = (reconstruction - target).square().mean()
        return {"loss": loss, "total_loss": loss, "reconstruction_loss": loss}

    def _predicted_intent(self, batch: TrainingBatch) -> torch.Tensor:
        scene = self.scene_adapter(batch.video.mean(dim=(1, 2, 3, 4, 5)).unsqueeze(-1))
        state_input = torch.cat((batch.ego_state, batch.history.flatten(1)), dim=-1)
        state = self.action_expert.state_encoder(state_input)
        command = self.command_embedding(batch.command_id)
        return self.intent_predictor(scene + state + command)

    def compute_intent_losses(self, batch: TrainingBatch):
        self.target_encoder.require_pretrained()
        predicted = self._predicted_intent(batch)
        with torch.no_grad():
            target = self.target_encoder(batch.future)
        loss = (predicted - target).square().mean()
        return {"loss": loss, "total_loss": loss, "jepa_loss": loss}

    def compute_losses(self, batch: TrainingBatch, *, intent_mode: str = "predicted"):
        if intent_mode != "predicted":
            raise AssertionError("training may only use predicted intent")
        intent = self.compute_intent_losses(batch)["loss"]
        predicted = self._predicted_intent(batch)
        flow = self.action_expert.core(predicted).square().mean()
        loss = flow + self.config.loss.lambda_jepa * intent
        return {
            "loss": loss,
            "total_loss": loss,
            "flow_loss": flow,
            "jepa_loss": intent,
        }

    @torch.no_grad()
    def predict(self, batch: ObservationBatch, num_steps=None, *, seed=None):
        command = self.command_embedding(batch.command_id)
        value = self.action_expert.core(command).reshape(-1, 1, 1)
        return value.expand(-1, 10, 3).clone()


def make_config(**training_updates) -> PlanningConfig:
    config = PlanningConfig()
    config.backbone.use_stub = True
    config.backbone.expected_feature_dim = 1
    config.model.context_dim = 8
    config.model.intent_heads = 1
    config.model.action_heads = 1
    config.model.trajectory_hidden_dim = 8
    config.model.trajectory_layers = 1
    config.model.intent_layers = 1
    config.model.action_layers = 1
    config.model.scene_tokens = 1
    config.training.epochs = 1
    config.training.batch_size = 1
    config.training.val_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.amp = False
    config.training.max_grad_norm = 1.0
    for name, value in training_updates.items():
        setattr(config.training, name, value)
    return config.validate()


def make_batch(seed: int, batch_size: int = 1) -> TrainingBatch:
    generator = torch.Generator().manual_seed(seed)
    return TrainingBatch(
        video=torch.randn(batch_size, 1, 3, 1, 1, 1, generator=generator),
        camera_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        frame_times_s=torch.zeros(batch_size, 1, 1),
        ego_state=torch.randn(batch_size, 4, generator=generator),
        history=torch.randn(batch_size, 6, 3, generator=generator),
        command_id=torch.zeros(batch_size, dtype=torch.long),
        sample_id=[f"synthetic-{seed}-{index}" for index in range(batch_size)],
        future=torch.randn(batch_size, 10, 3, generator=generator),
    )


class TinyDataset(Dataset[TrainingBatch]):
    def __init__(self, size: int) -> None:
        self.values = [make_batch(index) for index in range(size)]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> TrainingBatch:
        return self.values[index]


class TinyPlanningDataset(Dataset[PlanningSample]):
    """Unbatched samples for the public run_training data-loader path."""

    def __init__(self, size: int) -> None:
        self.values = []
        for index in range(size):
            batch = make_batch(index)
            self.values.append(
                PlanningSample(
                    video=torch.zeros(1, 3, 16, 224, 224),
                    camera_ids=batch.camera_ids[0],
                    frame_times_s=torch.linspace(-2.0, 0.0, 16).reshape(1, 16),
                    ego_state=batch.ego_state[0],
                    history=batch.history[0],
                    command_id=batch.command_id[0],
                    sample_id=batch.sample_id[0],
                    future=batch.future[0],
                )
            )

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> PlanningSample:
        return self.values[index]


def direct_collate(values):
    if len(values) != 1:
        raise AssertionError("tiny fixture expects batch_size=1")
    return values[0]


def parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


class TrainingStageTests(unittest.TestCase):
    def test_exact_trainable_groups_for_all_stages(self) -> None:
        model = TinyPlanningModel(make_config())
        configure_stage(model, Stage.TRAJECTORY_AE)
        expected = parameter_ids(model.target_encoder) | parameter_ids(
            model.trajectory_decoder
        )
        self.assertEqual(
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
            expected,
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))

        model.target_encoder.mark_pretrained()
        groups = configure_stage(model, Stage.INTENT)
        self.assertEqual(
            set(groups),
            {
                "scene_adapter",
                "intent_predictor",
                "action_expert.state_encoder",
                "command_embedding",
            },
        )
        self.assertTrue(all(p.requires_grad for p in model.action_expert.state_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.action_expert.core.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.target_encoder.parameters()))

        groups = configure_stage(model, Stage.JOINT)
        self.assertIn("action_expert.core", groups)
        self.assertTrue(all(p.requires_grad for p in model.action_expert.core.parameters()))
        all_ids = [id(p) for values in groups.values() for p in values]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_intent_rejects_unready_target(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "pretrained"):
            configure_stage(TinyPlanningModel(make_config()), Stage.INTENT)

    def test_intent_updates_state_encoder_but_not_action_core(self) -> None:
        config = make_config()
        model = TinyPlanningModel(config)
        model.target_encoder.mark_pretrained()
        optimizer = build_optimizer(model, config, Stage.INTENT)
        state_before = copy.deepcopy(model.action_expert.state_encoder.state_dict())
        core_before = copy.deepcopy(model.action_expert.core.state_dict())
        train_one_epoch(
            model,
            [make_batch(5)],
            optimizer=optimizer,
            scheduler=None,
            scaler=build_grad_scaler(torch.device("cpu"), False),
            config=config,
            stage=Stage.INTENT,
            device="cpu",
        )
        self.assertTrue(
            any(
                not torch.equal(state_before[name], value)
                for name, value in model.action_expert.state_encoder.state_dict().items()
            )
        )
        for name, value in model.action_expert.core.state_dict().items():
            torch.testing.assert_close(value, core_before[name], rtol=0, atol=0)

    def test_cpu_amp_accumulation_flush_validation_and_nonfinite(self) -> None:
        config = make_config(gradient_accumulation_steps=2, amp=True)
        model = TinyPlanningModel(config)
        optimizer = build_optimizer(model, config, Stage.TRAJECTORY_AE)
        scheduler = build_scheduler(optimizer)
        scaler = build_grad_scaler(torch.device("cpu"), config.training.amp)
        loader = DataLoader(TinyDataset(3), batch_size=1, collate_fn=direct_collate)
        before = copy.deepcopy(model.target_encoder.state_dict())
        result = train_one_epoch(
            model,
            loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            stage=Stage.TRAJECTORY_AE,
            device="cpu",
        )
        self.assertEqual(result.batches, 3)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertTrue(math.isfinite(result.metrics["loss"]))
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.target_encoder.state_dict().items()
            )
        )
        metrics = validate_one_epoch(
            model,
            loader,
            config=config,
            stage=Stage.TRAJECTORY_AE,
            device="cpu",
        )
        self.assertTrue(math.isfinite(metrics["loss"]))

        bad_batch = make_batch(99)
        bad_batch.future.fill_(float("nan"))
        bad_loader = [bad_batch]
        with self.assertRaises(FloatingPointError):
            train_one_epoch(
                model,
                bad_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                device="cpu",
            )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_prediction_rng_and_incompatible_config(self) -> None:
        config = make_config()
        model = TinyPlanningModel(config)
        model.target_encoder.mark_pretrained()
        optimizer = build_optimizer(model, config, Stage.INTENT)
        scheduler = build_scheduler(optimizer)
        scaler = build_grad_scaler(torch.device("cpu"), False)
        observation = make_batch(4).observations()
        expected = model.predict(observation)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "intent.pt"
            seed_everything(123)
            save_checkpoint(
                path,
                model=model,
                config=config,
                stage=Stage.INTENT,
                stage_complete=True,
                epoch=1,
                next_batch_index=0,
                global_step=7,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(1)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            self.assertFalse(torch.equal(expected, model.predict(observation)))
            state = load_checkpoint(
                path,
                model=model,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            self.assertEqual(state.global_step, 7)
            self.assertTrue(state.stage_complete)
            torch.testing.assert_close(expected, model.predict(observation))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            torch.testing.assert_close(torch.rand(1), expected_torch)

            incompatible = copy.deepcopy(config)
            incompatible.data.command_vocab = tuple(config.data.command_vocab) + ("NEW",)
            incompatible_model = TinyPlanningModel(incompatible)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint(path, model=incompatible_model, config=incompatible)

    def test_legacy_and_incomplete_stage_artifacts_are_rejected(self) -> None:
        config = make_config()
        model = TinyPlanningModel(config)
        with tempfile.TemporaryDirectory() as temporary:
            legacy = Path(temporary) / "legacy.pt"
            torch.save({"model": model.state_dict()}, legacy)
            with self.assertRaisesRegex(ValueError, "legacy"):
                inspect_checkpoint(legacy)

            incomplete = Path(temporary) / "stage_a_incomplete.pt"
            save_checkpoint(
                incomplete,
                model=model,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                stage_complete=False,
                epoch=0,
                next_batch_index=0,
                global_step=0,
            )
            state = inspect_checkpoint(incomplete)
            self.assertFalse(state.stage_complete)
            with self.assertRaisesRegex(RuntimeError, "unready"):
                save_checkpoint(
                    Path(temporary) / "invalid_complete.pt",
                    model=model,
                    config=config,
                    stage=Stage.TRAJECTORY_AE,
                    stage_complete=True,
                    epoch=1,
                    next_batch_index=0,
                    global_step=1,
                )

    def test_resume_equivalence_from_epoch_boundary(self) -> None:
        config = make_config(epochs=2, seed=18)
        data = TinyDataset(2)

        seed_everything(config.training.seed)
        uninterrupted = TinyPlanningModel(config)
        optimizer_a = build_optimizer(uninterrupted, config, Stage.TRAJECTORY_AE)
        scheduler_a = build_scheduler(optimizer_a)
        scaler_a = build_grad_scaler(torch.device("cpu"), False)
        first = train_one_epoch(
            uninterrupted,
            DataLoader(data, batch_size=1, collate_fn=direct_collate),
            optimizer=optimizer_a,
            scheduler=scheduler_a,
            scaler=scaler_a,
            config=config,
            stage=Stage.TRAJECTORY_AE,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "resume.pt"
            save_checkpoint(
                checkpoint,
                model=uninterrupted,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                stage_complete=False,
                epoch=1,
                next_batch_index=0,
                global_step=first.global_step,
                optimizer=optimizer_a,
                scheduler=scheduler_a,
                scaler=scaler_a,
            )
            second = train_one_epoch(
                uninterrupted,
                DataLoader(data, batch_size=1, collate_fn=direct_collate),
                optimizer=optimizer_a,
                scheduler=scheduler_a,
                scaler=scaler_a,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                device="cpu",
                global_step=first.global_step,
            )

            resumed = TinyPlanningModel(config)
            optimizer_b = build_optimizer(resumed, config, Stage.TRAJECTORY_AE)
            scheduler_b = build_scheduler(optimizer_b)
            scaler_b = build_grad_scaler(torch.device("cpu"), False)
            restored = load_checkpoint(
                checkpoint,
                model=resumed,
                config=config,
                optimizer=optimizer_b,
                scheduler=scheduler_b,
                scaler=scaler_b,
            )
            resumed_result = train_one_epoch(
                resumed,
                DataLoader(data, batch_size=1, collate_fn=direct_collate),
                optimizer=optimizer_b,
                scheduler=scheduler_b,
                scaler=scaler_b,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                device="cpu",
                global_step=restored.global_step,
            )
            self.assertEqual(second.global_step, resumed_result.global_step)
            for name, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


class EndToEndStageTests(unittest.TestCase):
    def test_three_stage_synthetic_cpu_lifecycle(self) -> None:
        config = make_config(epochs=1, seed=7)
        train_data = TinyPlanningDataset(2)
        validation_data = TinyPlanningDataset(1)
        with tempfile.TemporaryDirectory() as temporary:
            stage_a_path = Path(temporary) / "trajectory_ae.pt"
            stage_b_path = Path(temporary) / "intent.pt"
            stage_c_path = Path(temporary) / "joint.pt"

            stage_a_model = TinyPlanningModel(config)
            stage_a = run_training(
                stage_a_model,
                train_data,
                validation_data,
                config=config,
                stage=Stage.TRAJECTORY_AE,
                device="cpu",
                checkpoint_path=stage_a_path,
            )
            self.assertTrue(stage_a.stage_complete)
            self.assertTrue(stage_a_model.target_encoder.is_pretrained)

            stage_b_model = TinyPlanningModel(config)
            stage_b = run_training(
                stage_b_model,
                train_data,
                validation_data,
                config=config,
                stage=Stage.INTENT,
                device="cpu",
                init_checkpoint=stage_a_path,
                checkpoint_path=stage_b_path,
            )
            self.assertTrue(stage_b.stage_complete)

            stage_c_model = TinyPlanningModel(config)
            stage_c = run_training(
                stage_c_model,
                train_data,
                validation_data,
                config=config,
                stage=Stage.JOINT,
                device="cpu",
                init_checkpoint=stage_b_path,
                checkpoint_path=stage_c_path,
            )
            self.assertTrue(stage_c.stage_complete)
            self.assertEqual(stage_c.stage, Stage.JOINT)
            prediction = stage_c_model.predict(make_batch(10).observations())
            self.assertEqual(tuple(prediction.shape), (1, 10, 3))
            self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()

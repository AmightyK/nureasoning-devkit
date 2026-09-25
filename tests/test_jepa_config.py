import tempfile
import unittest
from pathlib import Path

import torch

from nureasoning.jepa_planning.config import (
    BackboneConfig,
    PlanningConfig,
    Stage,
    load_config,
    stage_policy,
)
from nureasoning.jepa_planning.contracts import ObservationBatch, TrainingBatch


ROOT = Path(__file__).resolve().parents[1]


def observation(frame_times=None):
    times = frame_times
    if times is None:
        times = torch.linspace(-2.0, 0.0, 16).reshape(1, 1, 16)
    return ObservationBatch(
        video=torch.zeros(1, 1, 3, 16, 224, 224),
        camera_ids=torch.zeros(1, 1, dtype=torch.long),
        frame_times_s=times,
        ego_state=torch.zeros(1, 4),
        history=torch.zeros(1, 6, 3),
        command_id=torch.zeros(1, dtype=torch.long),
        sample_id=["clip:anchor"],
    )


class PlanningConfigTests(unittest.TestCase):
    def test_bundled_configs_are_explicit_and_valid(self):
        smoke = load_config(ROOT / "nureasoning/jepa_planning/configs/smoke.yaml")
        train = load_config(ROOT / "nureasoning/jepa_planning/configs/train.yaml")
        self.assertTrue(smoke.backbone.use_stub)
        self.assertEqual(train.model.context_dim, 512)
        self.assertEqual(train.model.trajectory_norm_scale[:2], (50.0, 20.0))
        self.assertEqual(train.data.command_vocab[0], "UNKNOWN")

    def test_config_round_trip(self):
        config = PlanningConfig(backbone=BackboneConfig(use_stub=True)).validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config.save(path)
            restored = load_config(path)
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_real_backbone_requires_audited_identity(self):
        config = PlanningConfig()
        with self.assertRaisesRegex(ValueError, "local snapshot"):
            config.validate(require_real_backbone=True)

    def test_audited_code_requires_local_identity(self):
        with self.assertRaisesRegex(ValueError, "immutable model and code revisions"):
            PlanningConfig(
                backbone=BackboneConfig(allow_audited_local_code=True),
            ).validate()

    def test_remote_code_and_unknown_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "remote model code"):
            PlanningConfig(
                backbone=BackboneConfig(trust_remote_code=True),
            ).validate()

    def test_stage_freeze_policy(self):
        self.assertFalse(stage_policy(Stage.TRAJECTORY_AE).target_encoder)
        self.assertTrue(stage_policy(Stage.INTENT).target_encoder)
        self.assertTrue(stage_policy(Stage.INTENT).action_expert_core)
        self.assertFalse(stage_policy(Stage.INTENT).state_encoder)
        self.assertTrue(stage_policy(Stage.JOINT).backbone)
        self.assertFalse(stage_policy(Stage.JOINT).intent_predictor)


class ContractTests(unittest.TestCase):
    def test_observation_only_contract(self):
        batch = observation().validate(
            num_cameras=1,
            camera_vocab_size=8,
            command_vocab_size=8,
        )
        self.assertEqual(batch.batch_size, 1)
        self.assertFalse(hasattr(batch, "future"))

    def test_future_frame_is_rejected(self):
        times = torch.linspace(-1.5, 0.1, 16).reshape(1, 1, 16)
        with self.assertRaisesRegex(ValueError, "future-data leakage"):
            observation(times).validate(num_cameras=1)

    def test_nonchronological_frames_are_rejected(self):
        times = torch.linspace(-2.0, 0.0, 16).reshape(1, 1, 16)
        times[0, 0, 5] = -1.8
        with self.assertRaisesRegex(ValueError, "chronological"):
            observation(times).validate(num_cameras=1)

    def test_training_batch_extracts_observations(self):
        values = observation().__dict__
        batch = TrainingBatch(**values, future=torch.zeros(1, 10, 3))
        batch.validate(num_cameras=1)
        inference = batch.observations()
        self.assertIsInstance(inference, ObservationBatch)
        self.assertNotIsInstance(inference, TrainingBatch)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import torch

from nureasoning.jepa_planning.config import (
    BackboneConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    PlanningConfig,
)
from nureasoning.jepa_planning.contracts import ObservationBatch, TrainingBatch
from nureasoning.jepa_planning.model import PlanningModel, build_planning_model
from nureasoning.jepa_planning.trajectory import TargetTrajectoryEncoder


def tiny_config(*, lambda_jepa=1.0):
    return PlanningConfig(
        data=DataConfig(cameras=("front",)),
        backbone=BackboneConfig(
            use_stub=True,
            expected_feature_dim=8,
            # CLS plus one spatial patch for each of sixteen frames.
            stub_patch_tokens=17,
        ),
        model=ModelConfig(
            context_dim=16,
            scene_tokens=3,
            intent_layers=1,
            intent_heads=4,
            intent_dropout=0.0,
            trajectory_layers=1,
            trajectory_hidden_dim=16,
            action_layers=1,
            action_heads=4,
            action_dropout=0.0,
            action_mlp_ratio=2.0,
            num_inference_steps=2,
        ),
        loss=LossConfig(lambda_jepa=lambda_jepa),
    ).validate()


def observations(batch_size=2):
    return ObservationBatch(
        video=torch.zeros(batch_size, 1, 3, 16, 224, 224),
        camera_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        frame_times_s=torch.linspace(-2.0, 0.0, 16)
        .reshape(1, 1, 16)
        .expand(batch_size, -1, -1)
        .clone(),
        ego_state=torch.randn(batch_size, 4) * 0.1,
        history=torch.randn(batch_size, 6, 3) * 0.1,
        command_id=torch.arange(batch_size, dtype=torch.long) % 2,
        sample_id=[f"sample:{index}" for index in range(batch_size)],
    )


def training_batch(batch_size=2):
    observation = observations(batch_size)
    future = torch.randn(batch_size, 10, 3)
    future[..., 0] *= 10.0
    future[..., 1] *= 4.0
    future[..., 2] *= math.pi / 2.0
    return TrainingBatch(**vars(observation), future=future)


def ready_model(*, lambda_jepa=1.0):
    model = build_planning_model(tiny_config(lambda_jepa=lambda_jepa))
    model.target_encoder.mark_pretrained()
    model.freeze_target_encoder()
    return model


class RecordingTargetEncoder(TargetTrajectoryEncoder):
    def __init__(self):
        super().__init__(
            num_waypoints=10,
            latent_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        self.last_input = None

    def forward(self, normalized_future):
        self.last_input = normalized_future.detach().clone()
        return super().forward(normalized_future)


class PlanningModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_random_or_unfrozen_target_is_rejected_for_intent_and_joint_losses(self):
        model = PlanningModel(tiny_config())
        batch = training_batch()
        with self.assertRaisesRegex(RuntimeError, "not pretrained"):
            model.compute_intent_losses(batch)
        with self.assertRaisesRegex(RuntimeError, "not pretrained"):
            model.compute_losses(batch)

        model.target_encoder.mark_pretrained()
        with self.assertRaisesRegex(RuntimeError, "must be frozen"):
            model.compute_losses(batch)

    def test_raw_future_reaches_flow_and_one_normalized_copy_reaches_target(self):
        target = RecordingTargetEncoder().mark_pretrained().freeze()
        model = PlanningModel(tiny_config(), target_encoder=target).train()
        batch = training_batch()
        seen = {}
        original_forward = model.action_expert.forward

        def record_action(*args, **kwargs):
            seen["raw_future"] = kwargs["x_1"].detach().clone()
            seen["dit_context"] = kwargs["vlm_features"]
            return original_forward(*args, **kwargs)

        model.action_expert.forward = record_action
        values = model.compute_losses(batch)

        self.assertTrue(torch.isfinite(values["loss"]))
        self.assertEqual(values["loss"].ndim, 0)
        torch.testing.assert_close(seen["raw_future"], batch.future)
        scale = batch.future.new_tensor((50.0, 20.0, math.pi))
        torch.testing.assert_close(target.last_input, batch.future / scale)
        self.assertEqual(seen["dit_context"].shape, (2, 14, 16))
        # The target encoder is supervision-only and receives no gradient.
        values["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_stage_specific_loss_entry_points_are_finite(self):
        model = ready_model()
        batch = training_batch()

        # Stage A explicitly makes the representation trainable again.
        model.target_encoder.unfreeze_for_pretraining()
        autoencoder = model.compute_trajectory_autoencoder_losses(batch.future)
        self.assertTrue(torch.isfinite(autoencoder["loss"]))
        self.assertIn("reconstruction_loss", autoencoder)

        model.target_encoder.mark_pretrained().freeze()
        intent = model.compute_intent_losses(batch)
        self.assertTrue(torch.isfinite(intent["loss"]))
        self.assertIn("feature_loss", intent)

        joint = model.compute_losses(batch)
        self.assertTrue(torch.isfinite(joint["loss"]))
        torch.testing.assert_close(
            joint["loss"],
            joint["flow_loss"] + model.config.loss.lambda_jepa * joint["jepa_loss"],
        )

    def test_condition_is_predicted_intent_and_ablation_modes_are_explicit(self):
        model = ready_model().eval()
        batch = observations()
        normal = model.encode_observations(batch)
        without = model.encode_observations(batch, intent_mode="no_intent")
        shuffled = model.encode_observations(batch, intent_mode="shuffled")

        self.assertEqual(normal.dit_context.shape, (2, 14, 16))
        self.assertEqual(without.dit_context.shape, (2, 4, 16))
        torch.testing.assert_close(
            normal.dit_context[:, -10:], normal.predicted_intent
        )
        torch.testing.assert_close(
            shuffled.dit_context[:, -10:],
            shuffled.predicted_intent.roll(1, dims=0),
        )
        with self.assertRaisesRegex(ValueError, "intent_mode"):
            model.encode_observations(batch, intent_mode="target")

        model.train()
        with self.assertRaisesRegex(RuntimeError, "validation-only"):
            model.encode_observations(batch, intent_mode="shuffled")

    def test_observation_only_seeded_prediction_reuses_context_once(self):
        model = ready_model().eval()
        batch = observations()
        counts = {"backbone": 0, "adapter": 0, "predictor": 0}

        def count(name):
            def hook(_module, _inputs, _output):
                counts[name] += 1

            return hook

        handles = [
            model.backbone.register_forward_hook(count("backbone")),
            model.scene_adapter.register_forward_hook(count("adapter")),
            model.intent_predictor.register_forward_hook(count("predictor")),
        ]
        try:
            first = model.predict(batch, num_steps=3, seed=101)
        finally:
            for handle in handles:
                handle.remove()
        second = model.predict(batch, num_steps=3, seed=101)
        different = model.predict(batch, num_steps=3, seed=102)

        self.assertEqual(counts, {"backbone": 1, "adapter": 1, "predictor": 1})
        self.assertEqual(first.shape, (2, 10, 3))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.all(first[..., 2].abs() <= math.pi + 1e-6))
        torch.testing.assert_close(first, second)
        self.assertGreater((first - different).abs().max().item(), 1e-6)

    def test_seeded_prediction_does_not_mutate_caller_rng(self):
        model = ready_model().eval()
        batch = observations(batch_size=1)
        torch.manual_seed(717)
        expected = torch.randn(5)
        torch.manual_seed(717)
        model.predict(batch, num_steps=1, seed=99)
        actual = torch.randn(5)
        torch.testing.assert_close(actual, expected)

    def test_flow_gradient_reaches_predictor_after_zero_decoder_warmup(self):
        # The inherited action decoder starts at exact zero, so the first flow
        # backward can only train its final projection.  One optimizer update
        # opens a gradient path into the predicted-intent condition.
        model = ready_model(lambda_jepa=0.0).train()
        batch = training_batch()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        optimizer.zero_grad(set_to_none=True)
        warmup = model.compute_losses(batch)["loss"]
        warmup.backward()
        optimizer.step()
        self.assertGreater(
            model.action_expert.action_decoder[-1].weight.abs().sum().item(),
            0.0,
        )

        optimizer.zero_grad(set_to_none=True)
        flow_only = model.compute_losses(batch)["loss"]
        flow_only.backward()
        predictor_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.intent_predictor.parameters()
            if parameter.grad is not None
        )
        adapter_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.scene_adapter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(predictor_gradient, 0.0)
        self.assertGreater(adapter_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()

import inspect
import math
import unittest

import torch

from nureasoning.jepa_planning.intent import IntentPredictor
from nureasoning.jepa_planning.losses import (
    jepa_loss,
    trajectory_reconstruction_loss,
)
from nureasoning.jepa_planning.trajectory import (
    TargetTrajectoryEncoder,
    TrajectoryAutoencoder,
    denormalize_trajectory,
    normalize_trajectory,
)


class TrajectoryRepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def _autoencoder(self):
        return TrajectoryAutoencoder(
            num_waypoints=10,
            trajectory_dim=3,
            latent_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )

    def test_normalization_round_trip_and_temporal_shapes(self):
        raw = torch.randn(3, 10, 3)
        raw[..., 0] *= 20.0
        raw[..., 1] *= 8.0
        raw[..., 2] *= math.pi
        normalized = normalize_trajectory(raw)
        torch.testing.assert_close(denormalize_trajectory(normalized), raw)

        autoencoder = self._autoencoder()
        latent = autoencoder.encoder(normalized)
        reconstructed = autoencoder.decoder(latent)
        self.assertEqual(tuple(latent.shape), (3, 10, 16))
        self.assertEqual(tuple(reconstructed.shape), (3, 10, 3))

    def test_reconstruction_heading_and_motion_are_wrap_aware(self):
        target_raw = torch.zeros(1, 10, 3)
        predicted_raw = target_raw.clone()
        target_raw[..., 2] = math.pi - 0.01
        predicted_raw[..., 2] = -math.pi + 0.01
        losses = trajectory_reconstruction_loss(
            normalize_trajectory(predicted_raw),
            normalize_trajectory(target_raw),
        )
        self.assertLess(losses["heading_loss"].item(), 0.001)
        self.assertLess(losses["motion_loss"].item(), 1e-10)

        moving = target_raw.clone()
        moving[:, 1:, 0] = torch.arange(1, 10)
        moving_losses = trajectory_reconstruction_loss(
            normalize_trajectory(moving),
            normalize_trajectory(target_raw),
            dt_s=0.5,
            position_weight=0.0,
            heading_weight=0.0,
            motion_weight=1.0,
        )
        # x velocity error is 2 m/s; y and angular rates are zero, and the
        # physical motion loss averages all three channels.
        self.assertAlmostEqual(moving_losses["motion_loss"].item(), 4.0 / 3.0, places=5)

    def test_tiny_trajectory_autoencoder_overfits(self):
        autoencoder = self._autoencoder()
        autoencoder.train()
        target = torch.randn(2, 10, 3) * 0.2
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=2e-2)
        with torch.no_grad():
            initial = trajectory_reconstruction_loss(
                autoencoder(target), target, motion_weight=0.05
            )["loss"].item()
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = trajectory_reconstruction_loss(
                autoencoder(target), target, motion_weight=0.05
            )["loss"]
            loss.backward()
            optimizer.step()
        final = trajectory_reconstruction_loss(
            autoencoder(target), target, motion_weight=0.05
        )["loss"].item()
        self.assertLess(final, initial * 0.25)

    def test_random_target_cannot_be_silently_frozen(self):
        encoder = TargetTrajectoryEncoder(
            latent_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        self.assertFalse(encoder.is_pretrained)
        with self.assertRaisesRegex(RuntimeError, "not pretrained"):
            encoder.freeze()
        encoder.mark_pretrained().freeze()
        encoder.train()
        self.assertTrue(encoder.is_pretrained)
        self.assertTrue(encoder.is_frozen)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))


class IntentAndJEPATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.predictor = IntentPredictor(
            context_dim=16,
            num_waypoints=10,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        ).eval()

    def _inputs(self, batch_size=3):
        return (
            torch.randn(batch_size, 5, 16),
            torch.randn(batch_size, 1, 16),
            torch.randn(batch_size, 1, 16),
        )

    def test_predictor_uses_only_observation_tokens_and_has_expected_shape(self):
        parameters = tuple(inspect.signature(self.predictor.forward).parameters)
        self.assertEqual(parameters, ("scene_tokens", "state_token", "command_token"))
        prediction = self.predictor(*self._inputs())
        self.assertEqual(tuple(prediction.shape), (3, 10, 16))

    def test_predictor_is_sensitive_to_state_and_command(self):
        scene, state, command = self._inputs(batch_size=2)
        baseline = self.predictor(scene, state, command)
        perturbation = torch.linspace(-2.0, 2.0, 16).reshape(1, 1, 16)
        changed_state = self.predictor(scene, state + perturbation, command)
        changed_command = self.predictor(scene, state, command - perturbation)
        self.assertGreater((baseline - changed_state).abs().max().item(), 1e-5)
        self.assertGreater((baseline - changed_command).abs().max().item(), 1e-5)

    def test_jepa_losses_are_finite_and_target_is_stop_gradient(self):
        predicted = torch.randn(4, 10, 16, requires_grad=True)
        target = torch.randn(4, 10, 16, requires_grad=True)
        losses = jepa_loss(predicted, target, temperature=0.2)
        for value in losses.values():
            self.assertTrue(torch.isfinite(value).item())
            self.assertEqual(value.ndim, 0)
        self.assertEqual(losses["info_nce_active"].item(), 1.0)
        self.assertEqual(losses["num_local_negatives"].item(), 3.0)
        losses["loss"].backward()
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(predicted.grad.abs().sum().item(), 0.0)
        self.assertIsNone(target.grad)

    def test_batch_one_disables_info_nce_without_disabling_alignment(self):
        predicted = torch.randn(1, 10, 16, requires_grad=True)
        target = torch.randn(1, 10, 16)
        losses = jepa_loss(predicted, target)
        self.assertEqual(losses["info_nce_loss"].item(), 0.0)
        self.assertEqual(losses["info_nce_active"].item(), 0.0)
        self.assertEqual(losses["num_local_negatives"].item(), 0.0)
        losses["loss"].backward()
        self.assertGreater(predicted.grad.abs().sum().item(), 0.0)

    def test_predictor_and_jepa_loss_have_finite_gradients(self):
        scene, state, command = self._inputs(batch_size=2)
        target = torch.randn(2, 10, 16)
        losses = jepa_loss(self.predictor(scene, state, command), target)
        losses["loss"].backward()
        gradients = [
            parameter.grad
            for parameter in self.predictor.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()

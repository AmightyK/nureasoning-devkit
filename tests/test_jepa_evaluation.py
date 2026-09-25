import math
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from nureasoning.jepa_planning.checkpoint import CheckpointState
from nureasoning.jepa_planning.config import BackboneConfig, PlanningConfig, Stage
from nureasoning.jepa_planning.contracts import ObservationBatch
from nureasoning.jepa_planning.data import PlanningSample
from nureasoning.jepa_planning.evaluate import (
    build_argument_parser,
    evaluate_dataset,
    main,
    summarize_seeds,
    trajectory_error_sums,
)
from nureasoning.jepa_planning.trajectory_provider import (
    BENCHMARK_STEPS,
    JEPAPlanningTrajectoryProvider,
    ego_to_global,
    interpolate_ego_trajectory,
    validate_evaluation_checkpoint,
)


def tiny_config():
    return PlanningConfig(
        backbone=BackboneConfig(
            use_stub=True,
            expected_feature_dim=8,
            stub_patch_tokens=17,
        )
    ).validate()


_UNSET = object()


def sample(sample_id="clip:100", *, future=_UNSET):
    if future is _UNSET:
        future = torch.zeros(10, 3)
    return PlanningSample(
        video=torch.zeros(1, 3, 16, 224, 224),
        camera_ids=torch.zeros(1, dtype=torch.long),
        frame_times_s=torch.linspace(-2.0, 0.0, 16).reshape(1, 16),
        ego_state=torch.zeros(4),
        history=torch.zeros(6, 3),
        command_id=torch.tensor(0, dtype=torch.long),
        sample_id=sample_id,
        future=future,
    )


class MockPlanner(nn.Module):
    def __init__(self, prediction=None, fail_sample=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.prediction = (
            torch.zeros(1, 10, 3) if prediction is None else prediction.clone()
        )
        self.fail_sample = fail_sample
        self.calls = []

    def predict(self, batch, num_steps=None, *, seed=None, intent_mode="predicted"):
        self.calls.append((batch, num_steps, seed, intent_mode))
        if batch.sample_id[0] == self.fail_sample:
            raise RuntimeError("synthetic inference failure")
        return self.prediction.to(batch.video.device).expand(batch.batch_size, -1, -1)


class MockDataset:
    def __init__(self, samples):
        self.samples = samples
        self.coverage = {
            "candidates": len(samples) + 1,
            "included": len(samples),
            "excluded": 1,
            "coverage": len(samples) / (len(samples) + 1),
            "excluded_by_reason": {"missing_video_frame": 1},
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TrajectoryConversionTests(unittest.TestCase):
    def test_interpolation_adds_exact_zero_anchor_and_51_pose_grid(self):
        sparse = np.zeros((10, 3), dtype=np.float64)
        sparse[:, 0] = np.arange(1, 11) * 0.5
        dense = interpolate_ego_trajectory(sparse)

        self.assertEqual(dense.shape, (BENCHMARK_STEPS, 3))
        np.testing.assert_array_equal(dense[0], np.zeros(3))
        np.testing.assert_allclose(dense[:, 0], np.arange(51) * 0.1)
        np.testing.assert_allclose(dense[::5], np.vstack((np.zeros(3), sparse)))

    def test_heading_interpolation_takes_short_path_across_wrap(self):
        sparse = np.zeros((10, 3), dtype=np.float64)
        sparse[:, 2] = -3.0
        sparse[0, 2] = 3.0
        dense = interpolate_ego_trajectory(sparse)

        # Between +3 and -3 radians the correct interpolation stays near pi,
        # rather than swinging through heading zero.
        self.assertGreater(abs(dense[7, 2]), 3.0)
        self.assertGreater(abs(dense[8, 2]), 3.0)
        self.assertTrue(np.all(dense[:, 2] >= -math.pi))
        self.assertTrue(np.all(dense[:, 2] < math.pi))

    def test_ego_to_global_preserves_keyframe_anchor(self):
        sparse = np.zeros((10, 3), dtype=np.float64)
        sparse[:, 0] = np.arange(1, 11) * 0.5
        dense = interpolate_ego_trajectory(sparse)
        global_trajectory = ego_to_global(dense, (10.0, 20.0, math.pi / 2.0))

        np.testing.assert_allclose(
            global_trajectory[0], [10.0, 20.0, math.pi / 2.0], atol=1e-7
        )
        np.testing.assert_allclose(global_trajectory[-1, :2], [10.0, 25.0], atol=1e-7)


class ProviderTests(unittest.TestCase):
    def test_provider_uses_observation_only_preprocessing_and_predict(self):
        prediction = torch.zeros(1, 10, 3)
        prediction[0, :, 0] = torch.arange(1, 11) * 0.5
        model = MockPlanner(prediction)
        provider = JEPAPlanningTrajectoryProvider(
            model=model,
            config=tiny_config(),
            device="cpu",
            num_inference_steps=7,
            seed=19,
        )
        observation_sample = sample(future=None)

        with mock.patch(
            "nureasoning.jepa_planning.trajectory_provider.load_planning_sample",
            return_value=observation_sample,
        ) as loader:
            first = provider(
                "/clips/clip-a",
                100,
                {"pose": {"x": 10.0, "y": 20.0, "yaw": math.pi / 2.0}},
            )
            second = provider(
                "/clips/clip-a",
                100,
                {"pose": {"x": 10.0, "y": 20.0, "yaw": math.pi / 2.0}},
            )

        self.assertEqual(first.shape, (51, 3))
        np.testing.assert_allclose(first, second)
        loader.assert_called_with(
            "/clips/clip-a",
            provider.config.data,
            anchor_frame_index=100,
            observation_only=True,
        )
        inference_batch, steps, first_seed, intent_mode = model.calls[0]
        self.assertIsInstance(inference_batch, ObservationBatch)
        self.assertFalse(hasattr(inference_batch, "future"))
        self.assertEqual(steps, 7)
        self.assertEqual(first_seed, model.calls[1][2])
        self.assertEqual(intent_mode, "predicted")
        self.assertIsNotNone(provider.last_latency_s)
        self.assertEqual(len(provider.latencies_s), 2)

    def test_provider_rejects_incompatible_shape(self):
        model = MockPlanner(torch.zeros(1, 9, 3))
        provider = JEPAPlanningTrajectoryProvider(
            model=model, config=tiny_config(), device="cpu"
        )
        with mock.patch(
            "nureasoning.jepa_planning.trajectory_provider.load_planning_sample",
            return_value=sample(future=None),
        ):
            with self.assertRaisesRegex(ValueError, "expected"):
                provider("/clips/clip-a", 100, {"pose": {"x": 0, "y": 0, "yaw": 0}})

    def test_only_completed_joint_checkpoint_is_evaluable(self):
        config = tiny_config()
        for stage, complete in ((Stage.INTENT, True), (Stage.JOINT, False)):
            state = CheckpointState(stage, complete, 0, 0, 0, config, "mock.pt")
            with self.assertRaisesRegex(ValueError, "completed joint"):
                validate_evaluation_checkpoint(state)
        validate_evaluation_checkpoint(
            CheckpointState(Stage.JOINT, True, 0, 0, 0, config, "mock.pt")
        )


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_wrapped_and_model_never_receives_future(self):
        target = torch.zeros(10, 3)
        target[:, 0] = torch.arange(1, 11, dtype=torch.float32)
        target[:, 2] = -math.pi + 0.05
        prediction = target.unsqueeze(0).clone()
        prediction[..., 2] += 2.0 * math.pi
        model = MockPlanner(prediction, fail_sample="failure")
        dataset = MockDataset(
            [sample("success", future=target), sample("failure", future=target)]
        )

        report = evaluate_dataset(model, dataset, device="cpu", seed=31)

        self.assertAlmostEqual(report["metrics"]["ADE_m"], 0.0)
        self.assertAlmostEqual(report["metrics"]["FDE_m"], 0.0)
        self.assertLess(report["metrics"]["heading_error_rad"], 1e-5)
        self.assertEqual(report["coverage"]["source_candidates"], 3)
        self.assertEqual(report["coverage"]["preprocessing_excluded"], 1)
        self.assertEqual(report["coverage"]["evaluated"], 1)
        self.assertEqual(report["coverage"]["inference_failures"], 1)
        self.assertEqual(report["failures"][0]["sample_id"], "failure")
        self.assertEqual(report["seed"], 31)
        self.assertEqual(report["latency_s"]["count"], 1)
        for batch, _steps, _seed, intent_mode in model.calls:
            self.assertIsInstance(batch, ObservationBatch)
            self.assertFalse(hasattr(batch, "future"))
            self.assertEqual(intent_mode, "predicted")

    def test_intent_ablation_modes_are_runnable(self):
        dataset = MockDataset([sample("first"), sample("second")])
        model = MockPlanner()
        removed = evaluate_dataset(
            model, dataset, intent_mode="no_intent", batch_size=2
        )
        shuffled = evaluate_dataset(
            model, dataset, intent_mode="shuffled", batch_size=2
        )
        self.assertEqual(removed["coverage"]["evaluated"], 2)
        self.assertEqual(shuffled["coverage"]["evaluated"], 2)
        self.assertEqual(model.calls[-2][-1], "no_intent")
        self.assertEqual(model.calls[-1][-1], "shuffled")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            evaluate_dataset(model, dataset, intent_mode="shuffled", batch_size=1)

    def test_metric_shape_checks_and_seed_summary(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            trajectory_error_sums(torch.zeros(1, 10, 3), torch.zeros(1, 9, 3))
        reports = [
            {"metrics": {"ADE_m": 1.0, "FDE_m": 2.0, "heading_error_rad": 0.1, "heading_error_deg": 5.0}},
            {"metrics": {"ADE_m": 3.0, "FDE_m": 4.0, "heading_error_rad": 0.3, "heading_error_deg": 15.0}},
        ]
        summary = summarize_seeds(reports)
        self.assertEqual(summary["num_seeds"], 2)
        self.assertEqual(summary["metrics"]["ADE_m"]["mean"], 2.0)
        self.assertEqual(summary["metrics"]["ADE_m"]["std_population"], 1.0)

    def test_cli_parses_exact_metrics_and_benchmark_commands(self):
        parser = build_argument_parser()
        metrics = parser.parse_args(
            [
                "metrics",
                "--checkpoint",
                "joint.pt",
                "--data-root",
                "validation",
                "--seeds",
                "1,2",
                "--output",
                "metrics.json",
            ]
        )
        self.assertEqual(metrics.seeds, [1, 2])
        self.assertEqual(metrics.intent_mode, "predicted")
        benchmark = parser.parse_args(
            [
                "benchmark",
                "--checkpoint",
                "joint.pt",
                "--data-root",
                "validation",
                "--output",
                "benchmark.json",
            ]
        )
        self.assertEqual(benchmark.key_frame_index, 100)

    def test_cli_fails_explicitly_when_evaluator_data_is_unavailable(self):
        with self.assertRaisesRegex(FileNotFoundError, "data root is unavailable"):
            main(
                [
                    "metrics",
                    "--checkpoint",
                    "joint.pt",
                    "--data-root",
                    "/definitely/not/a/nureasoning/dataset",
                    "--output",
                    "metrics.json",
                ]
            )


if __name__ == "__main__":
    unittest.main()

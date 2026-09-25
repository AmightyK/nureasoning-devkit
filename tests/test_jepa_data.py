import json
import math
import pickle
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nureasoning.jepa_planning.config import CAMERA_VOCAB, DataConfig
from nureasoning.jepa_planning.contracts import ObservationBatch, TrainingBatch
from nureasoning.jepa_planning.data import (
    PlanningDataset,
    assert_disjoint_planning_splits,
    ego_pose_to_global,
    ego_vector_to_global,
    global_pose_to_ego,
    global_vector_to_ego,
    load_planning_sample,
    planning_collate_fn,
)


class SyntheticPlanningData:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def anchor_pose():
        return np.asarray([100.0, -20.0, math.pi - 0.05], dtype=np.float64)

    def make_clip(
        self,
        name="clip_a",
        *,
        cameras=("front",),
        start_tick=-30,
        end_tick=50,
        reorder=False,
        log_name=None,
        missing_anchor_camera=None,
    ) -> Path:
        clip = self.root / name
        (clip / "cameras").mkdir(parents=True)
        (clip / "ego_state").mkdir()

        for camera_index, camera in enumerate(cameras):
            for period, color in (
                ("past", (10 + camera_index, 20, 200)),
                ("anchor", (220, 30 + camera_index, 10)),
                ("future", (20, 220, 10 + camera_index)),
            ):
                Image.new("RGB", (18, 12), color).save(
                    clip / "cameras" / f"{camera}_{period}.png"
                )

        anchor = self.anchor_pose()
        frames = []
        for tick in range(start_tick, end_tick + 1):
            relative_time = tick / 10.0
            timestamp_us = 20_000_000 + tick * 100_000
            local_pose = np.asarray(
                [2.0 * relative_time, 0.25 * relative_time, 0.2 * relative_time]
            )
            global_pose = ego_pose_to_global(local_pose, anchor)
            velocity_global = ego_vector_to_global((2.0, 0.25), float(global_pose[2]))
            acceleration_global = ego_vector_to_global((0.1, -0.2), float(global_pose[2]))
            state = {
                "pose": {
                    "x": float(global_pose[0]),
                    "y": float(global_pose[1]),
                    "yaw": float(global_pose[2]),
                },
                "velocity": {
                    "vx": float(velocity_global[0]),
                    "vy": float(velocity_global[1]),
                    "frame": "global",
                },
                "acceleration": {
                    "ax": float(acceleration_global[0]),
                    "ay": float(acceleration_global[1]),
                    "frame": "global",
                },
            }
            ego_path = clip / "ego_state" / f"{timestamp_us}.pkl"
            with ego_path.open("wb") as handle:
                pickle.dump(state, handle)

            period = "past" if tick < 0 else "anchor" if tick == 0 else "future"
            camera_paths = {
                camera: f"cameras/{camera}_{period}.png" for camera in cameras
            }
            if tick == 0 and missing_anchor_camera:
                camera_paths.pop(missing_anchor_camera, None)
            frames.append(
                {
                    # Deliberately not useful as a list position or time stride.
                    "frame_index": 10_000 - 7 * tick,
                    "timestamp_us": timestamp_us,
                    "relative_time_s": relative_time,
                    "ego_state": f"ego_state/{timestamp_us}.pkl",
                    "sensors": {"cameras": camera_paths},
                    "mission_goal": {"command": "TURN_LEFT"},
                }
            )

        if reorder:
            random.Random(81).shuffle(frames)
        metadata = {
            "clip_token": name,
            "log_name": log_name or f"log_{name}",
            "frame_rate_hz": 10.0,
            "frames": frames,
        }
        (clip / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return clip


class PlanningDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_root = self.root / "train"
        self.val_root = self.root / "val"
        self.train_root.mkdir()
        self.val_root.mkdir()

    def config(self, **values):
        base = DataConfig(
            train_root=str(self.train_root),
            val_root=str(self.val_root),
            cameras=("front",),
        )
        return replace(base, **values)

    def test_coordinate_and_vector_round_trips_wrap_heading(self):
        anchor = (13.0, -7.0, math.pi - 0.01)
        pose = (-2.0, 4.0, -math.pi + 0.02)
        local = global_pose_to_ego(pose, anchor)
        restored = ego_pose_to_global(local, anchor)
        np.testing.assert_allclose(restored[:2], pose[:2], atol=1e-5)
        self.assertAlmostEqual(float(global_pose_to_ego(restored, pose)[2]), 0.0, places=5)

        vector = (3.5, -1.25)
        np.testing.assert_allclose(
            ego_vector_to_global(global_vector_to_ego(vector, anchor[2]), anchor[2]),
            vector,
            atol=1e-6,
        )

    def test_training_sample_uses_exact_pose_grids_and_raw_units(self):
        SyntheticPlanningData(self.train_root).make_clip(reorder=True)
        dataset = PlanningDataset(self.config(), split="train")
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        self.assertEqual(tuple(sample.video.shape), (1, 3, 16, 224, 224))
        self.assertEqual(tuple(sample.frame_times_s.shape), (1, 16))
        self.assertTrue(torch.all(sample.frame_times_s <= 0.0))
        self.assertAlmostEqual(float(sample.frame_times_s[0, -1]), 0.0)
        self.assertTrue(torch.all(sample.frame_times_s[0, 1:] > sample.frame_times_s[0, :-1]))
        np.testing.assert_allclose(sample.history[:, 0].numpy(), [-6, -5, -4, -3, -2, -1], atol=1e-4)
        np.testing.assert_allclose(sample.future[:, 0].numpy(), np.arange(1, 11), atol=1e-4)
        np.testing.assert_allclose(sample.history[:, 2].numpy(), np.arange(-6, 0) * 0.1, atol=1e-4)
        np.testing.assert_allclose(sample.future[:, 2].numpy(), np.arange(1, 11) * 0.1, atol=1e-4)
        # A 10 metre endpoint demonstrates that this layer did not divide by 50.
        self.assertAlmostEqual(float(sample.future[-1, 0]), 10.0, places=4)
        np.testing.assert_allclose(sample.ego_state.numpy(), [2.0, 0.25, 0.1, -0.2], atol=1e-5)
        self.assertEqual(int(sample.command_id), self.config().command_vocab.index("TURN_LEFT"))
        self.assertFalse(any("reasoning" in key for key in dataset.samples[0].anchor.frame))

        batch = planning_collate_fn([sample])
        self.assertIsInstance(batch, TrainingBatch)
        self.assertEqual(tuple(batch.future.shape), (1, 10, 3))

    def test_observation_only_requires_no_future_metadata(self):
        clip = SyntheticPlanningData(self.val_root).make_clip(end_tick=0)
        config = self.config()
        dataset = PlanningDataset(config, split="val", observation_only=True)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertIsNone(sample.future)
        batch = planning_collate_fn([sample])
        self.assertIsInstance(batch, ObservationBatch)
        self.assertNotIsInstance(batch, TrainingBatch)

        metadata = json.loads((clip / "metadata.json").read_text(encoding="utf-8"))
        anchor_index = next(
            index for index, frame in enumerate(metadata["frames"])
            if frame["relative_time_s"] == 0.0
        )
        direct = load_planning_sample(
            clip, config, anchor_frame_index=anchor_index, observation_only=True
        )
        self.assertIsNone(direct.future)
        self.assertEqual(direct.sample_id, sample.sample_id)

    def test_missing_complete_input_is_reported_not_zero_filled(self):
        SyntheticPlanningData(self.train_root).make_clip(missing_anchor_camera="front")
        dataset = PlanningDataset(self.config(), split="train")
        self.assertEqual(len(dataset), 0)
        report = dataset.coverage
        self.assertGreater(report["excluded_by_reason"].get("missing_anchor_camera", 0), 0)
        self.assertIn("missing_anchor_camera", report["examples"])

    def test_train_subsets_whole_clips_reproducibly_but_validation_does_not(self):
        builder = SyntheticPlanningData(self.train_root)
        for index in range(5):
            builder.make_clip(f"clip_{index}")
        config = self.config(train_clip_fraction=0.4, clip_seed=19)
        first = PlanningDataset(config, split="train")
        second = PlanningDataset(config, split="train")
        other = PlanningDataset(replace(config, clip_seed=1), split="train")
        first_names = [sample.clip_name for sample in first.samples]
        self.assertEqual(len(first_names), 2)
        self.assertEqual(first_names, [sample.clip_name for sample in second.samples])
        self.assertNotEqual(first_names, [sample.clip_name for sample in other.samples])
        self.assertEqual(first.coverage["clips_selected"], 2)

        val_builder = SyntheticPlanningData(self.val_root)
        for index in range(3):
            val_builder.make_clip(f"val_{index}")
        validation = PlanningDataset(config, split="val")
        self.assertEqual(len(validation), 3)
        self.assertEqual(validation.coverage["clips_selected"], 3)

    def test_split_overlap_fails_even_when_train_subset_might_hide_it(self):
        SyntheticPlanningData(self.train_root).make_clip("duplicated", end_tick=0, log_name="same_log")
        SyntheticPlanningData(self.val_root).make_clip("duplicated", end_tick=0, log_name="same_log")
        config = self.config(train_clip_fraction=0.2)
        train = PlanningDataset(config, split="train", observation_only=True)
        validation = PlanningDataset(config, split="val", observation_only=True)
        with self.assertRaisesRegex(ValueError, "split overlap"):
            assert_disjoint_planning_splits(train, validation)

    def test_unknown_command_has_explicit_id_zero(self):
        clip = SyntheticPlanningData(self.train_root).make_clip()
        metadata_path = clip / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for frame in metadata["frames"]:
            if frame["relative_time_s"] == 0.0:
                frame["mission_goal"] = {"command": "FLY"}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(int(PlanningDataset(self.config())[0].command_id), 0)

    def test_eight_camera_contract_and_stable_ids(self):
        cameras = tuple(CAMERA_VOCAB)
        clip = SyntheticPlanningData(self.train_root).make_clip(cameras=cameras)
        config = self.config(cameras=cameras)
        metadata = json.loads((clip / "metadata.json").read_text(encoding="utf-8"))
        anchor_index = next(
            index for index, frame in enumerate(metadata["frames"])
            if frame["relative_time_s"] == 0.0
        )
        sample = load_planning_sample(
            clip, config, anchor_frame_index=anchor_index, observation_only=False
        )
        self.assertEqual(tuple(sample.video.shape), (8, 3, 16, 224, 224))
        self.assertEqual(sample.camera_ids.tolist(), list(range(8)))
        self.assertEqual(tuple(sample.frame_times_s.shape), (8, 16))


if __name__ == "__main__":
    unittest.main()

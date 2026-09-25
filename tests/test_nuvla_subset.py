import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nureasoning.nuvla.models.data_loader import NuReasoningVLADataset, VLADataConfig
from nureasoning.nuvla.train import VLATrainer, parse_args


class TrainingSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index in range(40):
            clip = self.root / "part_1" / f"clip_{index:03d}"
            clip.mkdir(parents=True)
            (clip / "reasoning.json").write_text("{}")
            frames = [
                {"frame_index": frame, "reasoning": "reasoning.json", "ego_state": "ego.pkl"}
                for frame in (10, 20)
            ]
            (clip / "metadata.json").write_text(json.dumps({"frames": frames}))

    def dataset(self, fraction=1.0, seed=42):
        return NuReasoningVLADataset(VLADataConfig(
            data_root=str(self.root), clip_fraction=fraction, clip_seed=seed,
        ))

    def test_default_keeps_all_samples(self):
        self.assertEqual(len(self.dataset()), 80)

    def test_five_percent_keeps_whole_clips_reproducibly(self):
        random_state = random.getstate()
        first = self.dataset(0.05)
        names = [sample["clip_name"] for sample in first.samples]
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(names, [s["clip_name"] for s in self.dataset(0.05).samples])
        self.assertEqual(random.getstate(), random_state)
        self.assertNotEqual(names, [s["clip_name"] for s in self.dataset(0.05, 7).samples])

    def test_small_fraction_keeps_one_clip(self):
        self.assertEqual(len(self.dataset(0.001)), 2)

    def test_invalid_fractions(self):
        for fraction in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.dataset(fraction)

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(len(NuReasoningVLADataset(VLADataConfig(
                data_root=empty, clip_fraction=0.05,
            ))), 0)

    def test_trainer_applies_fraction_only_to_training(self):
        with patch("sys.argv", ["train", "--train_fraction", "0.05", "--data_seed", "7"]):
            args = parse_args()
        args.data_root = args.test_data_root = str(self.root)
        args.num_workers = 0
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.args = args
        trainer.is_distributed = False
        trainer.is_main = False
        trainer._build_data()
        self.assertEqual(len(trainer.train_loader.dataset), 4)
        self.assertEqual(len(trainer.test_loader.dataset), 80)
        self.assertEqual(trainer.train_loader.dataset.config.clip_seed, 7)

    def test_cli_rejects_invalid_fraction(self):
        with patch("sys.argv", ["train", "--train_fraction", "0"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

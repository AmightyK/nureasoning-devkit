import unittest
from types import SimpleNamespace

import torch
from torch import nn

from nureasoning.jepa_planning.backbone import (
    LeVJEPABackbone,
    SceneAdapter,
    build_backbone,
)
from nureasoning.jepa_planning.config import BackboneConfig


def stub_config(**overrides):
    values = {
        "use_stub": True,
        "expected_feature_dim": 8,
        # Seventeen raw tokens means CLS + one spatial patch for each frame.
        "stub_patch_tokens": 17,
        "camera_chunk_size": 2,
    }
    values.update(overrides)
    return BackboneConfig(**values)


def tiny_video(batch=2, cameras=3):
    return torch.randn(batch, cameras, 3, 16, 4, 4)


class TensorOutputModel(nn.Module):
    def __init__(self, tokens=17, width=8, prefix_tokens=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.tokens = tokens
        self.width = width
        if prefix_tokens is not None:
            self.config = SimpleNamespace(num_prefix_tokens=prefix_tokens)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        offsets = torch.arange(
            self.tokens * self.width,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        ).reshape(1, self.tokens, self.width)
        return offsets.expand(batch, -1, -1) * self.scale


class LeVJEPABackboneTests(unittest.TestCase):
    def test_stub_output_removes_cls_and_preserves_camera_order(self):
        wrapper = build_backbone(stub_config())
        video = torch.stack(
            [torch.full((3, 16, 4, 4), float(value)) for value in range(6)]
        ).reshape(2, 3, 3, 16, 4, 4)
        output = wrapper(video)
        self.assertEqual(output.shape, (2, 3, 16, 8))
        self.assertTrue(torch.all(output[0, 0] < output[0, 1]))
        self.assertTrue(torch.all(output[0, 2] < output[1, 0]))
        self.assertEqual(wrapper.last_layout.num_frames, 16)
        self.assertEqual(wrapper.last_layout.patches_per_frame, 1)
        self.assertTrue(wrapper.identity.is_stub)
        self.assertIn("code_revision", wrapper.identity.to_dict())

    def test_advertised_cls_contract_is_checked(self):
        model = TensorOutputModel(prefix_tokens=0)
        with self.assertRaisesRegex(ValueError, "prefix-token metadata"):
            LeVJEPABackbone(stub_config(use_stub=False), model=model)

    def test_cls_is_not_returned_as_a_patch(self):
        model = TensorOutputModel(tokens=17, prefix_tokens=1)
        wrapper = LeVJEPABackbone(stub_config(use_stub=False), model=model)
        output = wrapper(tiny_video(batch=1, cameras=1))
        raw = model(tiny_video(batch=1, cameras=1).reshape(1, 3, 16, 4, 4))
        self.assertEqual(output.shape[2], 16)
        self.assertTrue(torch.equal(output[0, 0, 0], raw[0, 1]))

    def test_incompatible_spatiotemporal_layout_fails(self):
        wrapper = LeVJEPABackbone(
            stub_config(use_stub=False),
            model=TensorOutputModel(tokens=16, prefix_tokens=1),
        )
        with self.assertRaisesRegex(ValueError, "frame-major layout"):
            wrapper(tiny_video(batch=1, cameras=1))

    def test_parent_train_keeps_feature_model_in_eval_mode(self):
        wrapper = build_backbone(stub_config())
        wrapper.train()
        self.assertTrue(wrapper.training)
        self.assertFalse(wrapper.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.parameters()))

    def test_nonlocal_real_loading_is_rejected_before_import(self):
        config = BackboneConfig(
            use_stub=False,
            local_path=None,
            revision="immutable-model-commit",
            code_revision="immutable-code-commit",
            allow_audited_local_code=True,
        )
        with self.assertRaisesRegex(ValueError, "local_path"):
            build_backbone(config)


class SceneAdapterTests(unittest.TestCase):
    def make_adapter(self):
        return SceneAdapter(
            feature_dim=8,
            context_dim=16,
            num_scene_tokens=5,
            num_cameras=4,
            num_frames=16,
            num_heads=4,
        )

    def test_fixed_budget_and_adapter_gradients_with_frozen_backbone(self):
        backbone = build_backbone(stub_config(camera_chunk_size=1))
        adapter = self.make_adapter()
        video = tiny_video(batch=2, cameras=2)
        camera_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        frame_times = torch.linspace(-2.0, 0.0, 16).reshape(1, 1, 16).expand(2, 2, -1)

        patch_tokens = backbone(video)
        scene_tokens = adapter(patch_tokens, camera_ids, frame_times)
        self.assertEqual(scene_tokens.shape, (2, 5, 16))
        scene_tokens.square().mean().backward()

        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
        gradients = [parameter.grad for parameter in adapter.parameters()]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in gradients))

    def test_camera_identity_changes_scene_features(self):
        torch.manual_seed(7)
        adapter = self.make_adapter().eval()
        patch_tokens = torch.ones(1, 2, 16, 8)
        frame_times = torch.linspace(-2.0, 0.0, 16).reshape(1, 1, 16).expand(1, 2, -1)
        first_ids = torch.tensor([[0, 1]], dtype=torch.long)
        second_ids = torch.tensor([[2, 3]], dtype=torch.long)
        first = adapter(patch_tokens, first_ids, frame_times)
        second = adapter(patch_tokens, second_ids, frame_times)
        self.assertFalse(torch.allclose(first, second))

    def test_frame_layout_and_camera_vocabulary_are_validated(self):
        adapter = self.make_adapter()
        times = torch.zeros(1, 1, 16)
        with self.assertRaisesRegex(ValueError, "frame-major layout"):
            adapter(torch.zeros(1, 1, 15, 8), torch.zeros(1, 1, dtype=torch.long), times)
        with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
            adapter(torch.zeros(1, 1, 16, 8), torch.full((1, 1), 4), times)

    def test_adapter_rejects_future_or_nonchronological_frame_times(self):
        adapter = self.make_adapter()
        patches = torch.zeros(1, 1, 16, 8)
        camera_ids = torch.zeros(1, 1, dtype=torch.long)
        future = torch.linspace(-1.0, 0.1, 16).reshape(1, 1, 16)
        with self.assertRaisesRegex(ValueError, "future-data leakage"):
            adapter(patches, camera_ids, future)
        nonchronological = torch.linspace(-1.0, 0.0, 16).reshape(1, 1, 16)
        nonchronological[..., 5] = -0.9
        with self.assertRaisesRegex(ValueError, "chronological"):
            adapter(patches, camera_ids, nonchronological)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training dependency")
class BoundaryRoleModelTests(unittest.TestCase):
    def test_rng_restore_normalizes_checkpoint_mapped_tensors(self) -> None:
        import random

        import numpy as np
        import torch

        from paper_reconstruction_stage2.training import (
            _restore_rng_state,
            _rng_state,
        )

        saved = _rng_state()
        expected_cpu = saved["torch_cpu"].clone()
        # A non-byte tensor exercises the same normalization path as a CPU RNG
        # tensor that torch.load(map_location="cuda") moved onto the GPU.
        saved["torch_cpu"] = saved["torch_cpu"].to(dtype=torch.int64)
        if torch.cuda.is_available():
            saved["torch_cpu"] = saved["torch_cpu"].to(device="cuda")
            saved["torch_cuda"] = [
                value.to(device="cuda") for value in saved["torch_cuda"]
            ]

        random.random()
        np.random.random()
        torch.rand(1)
        _restore_rng_state(saved)

        self.assertEqual(torch.get_rng_state().device.type, "cpu")
        self.assertEqual(torch.get_rng_state().dtype, torch.uint8)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_cpu))

    def test_model_and_multitask_loss_shapes(self) -> None:
        import torch

        from paper_reconstruction_stage2.losses import BoundaryRoleLoss
        from paper_reconstruction_stage2.model import (
            BoundaryRoleModelConfig,
            BoundaryRoleNetV1,
        )

        model = BoundaryRoleNetV1(
            BoundaryRoleModelConfig(branch_channels=8, fusion_channels=8)
        )
        geometry = torch.randn(2, 17, 7)
        raster = torch.randn(2, 4, 17, 9)
        outputs = model(geometry, raster)
        self.assertEqual(tuple(outputs["role_logits"].shape), (2, 17, 3))
        self.assertEqual(tuple(outputs["transition_logits"].shape), (2, 17))
        targets = torch.softmax(torch.randn(2, 17, 3), dim=-1)
        confidence = torch.rand(2, 17)
        losses = BoundaryRoleLoss()(outputs, targets, confidence)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_hybrid_attention_and_single_branch_candidates(self) -> None:
        import torch

        from paper_reconstruction_stage2.model import BoundaryRoleNetV1
        from paper_reconstruction_stage2.model_candidates import candidate_model_config

        geometry = torch.randn(2, 17, 7)
        raster = torch.randn(2, 4, 17, 9)
        hybrid = candidate_model_config("hybrid_attention")
        hybrid.branch_channels = 8
        hybrid.fusion_channels = 16
        hybrid.attention_layers = 1
        model = BoundaryRoleNetV1(hybrid)
        self.assertEqual(
            tuple(model(geometry, raster)["role_logits"].shape), (2, 17, 3)
        )
        geometry_model = BoundaryRoleNetV1(
            candidate_model_config("geometry_only")
        )
        self.assertEqual(
            tuple(geometry_model(geometry)["transition_logits"].shape), (2, 17)
        )
        raster_model = BoundaryRoleNetV1(candidate_model_config("raster_only"))
        self.assertEqual(
            tuple(raster_model(geometry, raster)["role_logits"].shape), (2, 17, 3)
        )

        global_config = candidate_model_config("global_context_hybrid")
        global_config.branch_channels = 8
        global_config.fusion_channels = 16
        global_config.attention_layers = 1
        global_config.global_attention_layers = 1
        global_config.global_context_points = 32
        global_config.global_mask_size = 32
        global_model = BoundaryRoleNetV1(global_config)
        global_outputs = global_model(
            geometry,
            raster,
            torch.randn(2, 32, 9),
            torch.rand(2, 1, 32, 32),
        )
        self.assertEqual(tuple(global_outputs["role_logits"].shape), (2, 17, 3))


if __name__ == "__main__":
    unittest.main()

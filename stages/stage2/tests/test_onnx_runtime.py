from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from paper_reconstruction_stage2.fragments import FragmentProposalConfig
from paper_reconstruction_stage2.inference import ContourInferenceConfig
from paper_reconstruction_stage2.model import BoundaryRoleModelConfig
from paper_reconstruction_stage2.onnx_runtime import (
    OnnxContourPredictor,
    OnnxModelRelease,
)
from paper_reconstruction_stage2.training_data import BoundaryWindowConfig
from paper_reconstruction_stage2.release import (
    STAGE2_RELEASE_CHECKPOINT_SHA256,
    STAGE2_RELEASE_ONNX_SHA256,
)


def _release_document(model_name: str, digest: str) -> dict:
    return {
        "schema_version": "stage2.model-release.v2",
        "release_version": "1.0.0",
        "model_name": "global_context_hybrid",
        "source_checkpoint": {"sha256": "a" * 64},
        "production_model": {
            "asset": model_name,
            "format": "onnx",
            "sha256": digest,
            "opset": 18,
            "execution_provider": "CPUExecutionProvider",
        },
        "model_config": BoundaryRoleModelConfig(
            architecture="global_context_hybrid",
            attention_layers=1,
            use_global_context=True,
        ).public_dict(),
        "window_config": {
            "sequence_length": 33,
            "strip_width": 17,
            "outside_extent_px": 8.0,
            "inside_extent_px": 24.0,
            "minimum_label_confidence": 0.25,
            "puzzle_cache_size": 2,
            "image_cache_size": 8,
            "global_context_cache_size": 64,
        },
        "default_inference_config": {
            "sequence_length": 33,
            "strip_width": 17,
            "stride": 8,
            "batch_size": 2,
            "role_temperature": 1.0,
            "transition_temperature": 1.0,
        },
        "fragment_config": FragmentProposalConfig().public_dict(),
    }


class _Node:
    def __init__(self, name: str, shape: list, rank: int):
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"
        assert len(shape) == rank


class _FakeSession:
    def get_inputs(self):
        return [
            _Node("geometry", ["batch", 33, 7], 3),
            _Node("raster", ["batch", 4, 33, 17], 4),
            _Node("global_geometry", ["batch", 256, 9], 3),
            _Node("global_mask", ["batch", 1, 64, 64], 4),
        ]

    def get_outputs(self):
        return [
            _Node("role_logits", ["batch", 33, 3], 3),
            _Node("transition_logits", ["batch", 33], 2),
        ]

    def run(self, output_names, feeds):
        self.output_names = output_names
        self.feeds = feeds
        batch = feeds["geometry"].shape[0]
        return (
            np.zeros((batch, 33, 3), dtype=np.float32),
            np.zeros((batch, 33), dtype=np.float32),
        )


class OnnxReleaseTests(unittest.TestCase):
    def test_repository_release_matches_frozen_constants(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        release = OnnxModelRelease.load(project_root / "models" / "release.json")
        self.assertEqual(release.model_sha256, STAGE2_RELEASE_ONNX_SHA256)
        self.assertEqual(
            release.source_checkpoint_sha256,
            STAGE2_RELEASE_CHECKPOINT_SHA256,
        )

    def test_release_loader_verifies_model_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"model-bytes")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            release_path = root / "release.json"
            release_path.write_text(
                json.dumps(_release_document(model.name, digest)), encoding="utf-8"
            )
            release = OnnxModelRelease.load(release_path)
            self.assertEqual(release.model_sha256, digest)
            model.write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                OnnxModelRelease.load(release_path)

    def test_release_model_path_cannot_escape_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.onnx"
            outside.write_bytes(b"model")
            release_dir = root / "release"
            release_dir.mkdir()
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            document = _release_document("../outside.onnx", digest)
            release_path = release_dir / "release.json"
            release_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                OnnxModelRelease.load(release_path)

    def test_predictor_builds_float32_onnx_feed(self) -> None:
        model_config = BoundaryRoleModelConfig(
            architecture="global_context_hybrid",
            attention_layers=1,
            use_global_context=True,
        )
        sampler = SimpleNamespace(
            config=BoundaryWindowConfig(sequence_length=33, strip_width=17)
        )
        session = _FakeSession()
        predictor = OnnxContourPredictor(
            session,
            sampler,
            model_config,
            ContourInferenceConfig(
                sequence_length=33, strip_width=17, stride=8, batch_size=2
            ),
            FragmentProposalConfig(),
        )
        sample = {
            "geometry": np.ones((33, 7), dtype=np.float64),
            "raster": np.ones((4, 33, 17), dtype=np.float64),
            "global_geometry": np.ones((256, 9), dtype=np.float64),
            "global_mask": np.ones((1, 64, 64), dtype=np.float64),
        }
        roles, transitions = predictor._predict_batch([sample, sample])
        self.assertEqual(roles.shape, (2, 33, 3))
        self.assertEqual(transitions.shape, (2, 33))
        self.assertTrue(
            all(value.dtype == np.float32 for value in session.feeds.values())
        )
        self.assertTrue(
            all(value.flags.c_contiguous for value in session.feeds.values())
        )


if __name__ == "__main__":
    unittest.main()

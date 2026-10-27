from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from paper_reconstruction_stage2.fragments import FragmentProposalConfig
from paper_reconstruction_stage2.inference import ContourInferenceConfig
from paper_reconstruction_stage2.production import (
    EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION,
    EDGE_HYPOTHESES_SCHEMA_VERSION,
    BoundaryGeometryConfig,
    ProductionBoundarySampler,
    ProductionInferenceConfig,
    Stage2ProductionRunner,
    load_stage1_output,
)
from paper_reconstruction_stage2.training_data import BoundaryWindowConfig
from paper_reconstruction_stage2.verification import verify_stage2_output


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stage1_capture(
    root: Path,
    capture_id: str,
    *,
    accepted: bool = True,
    disconnected: bool = False,
    bbox_origin: tuple[int, int] = (20, 30),
) -> dict:
    (root / "pieces").mkdir(parents=True, exist_ok=True)
    (root / "piece_masks").mkdir(parents=True, exist_ok=True)
    height, width = 72, 80
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[5:68, 7:75] = 255
    if disconnected:
        mask[:] = 0
        mask[4:20, 4:20] = 255
        mask[45:65, 55:75] = 255
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = (238, 232, 214)
    rgba[..., 3] = mask
    piece_path = root / "pieces" / f"{capture_id}.png"
    mask_path = root / "piece_masks" / f"{capture_id}.png"
    Image.fromarray(rgba, mode="RGBA").save(piece_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    x, y = bbox_origin
    return {
        "schema_version": "stage1.segmentation-manifest.v1",
        "capture_id": capture_id,
        "source_path": f"captures/{capture_id}.jpg",
        "source_sha256": "a" * 64,
        "capture_type_hint": "phone",
        "original_size_hw": [300, 400],
        "bbox_xywh": [x, y, width, height],
        "piece_path": piece_path.relative_to(root).as_posix(),
        "piece_mask_path": mask_path.relative_to(root).as_posix(),
        "accepted": accepted,
        "decision_source": "automatic",
        "review_status": "none",
        "model_version": "1.0.0",
    }


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "segmentation_manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class _FakePredictor:
    config = ContourInferenceConfig(
        sequence_length=33,
        strip_width=17,
        stride=8,
        batch_size=2,
    )
    fragment_config = FragmentProposalConfig(minimum_fragment_points=4)

    def __init__(self, sampler: ProductionBoundarySampler):
        self.sampler = sampler

    def predict(self, entry):
        contour = self.sampler.full_observation(entry)
        count = len(contour["points_xy"])
        probabilities = np.zeros((count, 3), dtype=np.float32)
        probabilities[:, 0] = 0.90
        probabilities[:, 1] = 0.08
        probabilities[:, 2] = 0.02
        transitions = np.full(count, 0.1, dtype=np.float32)
        length = max(4, count // 3)
        return {
            "role_logits": np.log(probabilities),
            "role_probabilities": probabilities,
            "transition_logits": np.full(count, -2.0, dtype=np.float32),
            "transition_probabilities": transitions,
            "candidate_fragments": [
                {
                    "rank": 1,
                    "start_index": 0,
                    "end_index": length - 1,
                    "length_points": length,
                    "length_fraction": length / count,
                    "wraps": False,
                    "score": 0.9,
                    "mean_seam_probability": 0.9,
                    "mean_uncertainty_probability": 0.02,
                    "endpoint_transition_probability": 0.1,
                    "source": "test",
                    "hypothesis": "test",
                    "split_threshold": None,
                }
            ],
        }


class Stage1InputTests(unittest.TestCase):
    def test_loader_validates_and_skips_unaccepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = _write_stage1_capture(root, "capture_b")
            rejected = _write_stage1_capture(
                root, "capture_a", accepted=False, bbox_origin=(100, 120)
            )
            manifest = _write_manifest(root, [rejected, accepted])
            before = _file_sha(manifest)

            inputs = load_stage1_output(root)

            self.assertEqual([item.capture_id for item in inputs.captures], ["capture_b"])
            self.assertEqual(inputs.total_row_count, 2)
            self.assertEqual(inputs.skipped_unaccepted_count, 1)
            self.assertEqual(before, _file_sha(manifest))

    def test_disconnected_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _write_stage1_capture(root, "capture_bad", disconnected=True)
            _write_manifest(root, [row])
            with self.assertRaisesRegex(ValueError, "connected component"):
                load_stage1_output(root)

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _write_stage1_capture(root, "capture_bad")
            row["piece_path"] = "../outside.png"
            _write_manifest(root, [row])
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_stage1_output(root)


class ProductionInferenceTests(unittest.TestCase):
    def test_truth_free_export_coordinates_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "stage1"
            stage1.mkdir()
            row = _write_stage1_capture(
                stage1, "capture_001", bbox_origin=(20, 30)
            )
            _write_manifest(stage1, [row])
            model = root / "model.onnx"
            model.write_bytes(b"synthetic-onnx-model")
            output = root / "stage2"
            inputs = load_stage1_output(stage1)
            stage1_hash_before = _file_sha(stage1 / "segmentation_manifest.jsonl")

            window = BoundaryWindowConfig(sequence_length=33, strip_width=17)
            sampler = ProductionBoundarySampler(
                window,
                BoundaryGeometryConfig(
                    boundary_spacing_px=2.0,
                    minimum_boundary_points=32,
                    maximum_boundary_points=512,
                    max_inward_thickness_px=32,
                ),
            )
            first = Stage2ProductionRunner(
                config=ProductionInferenceConfig(
                    input_path=stage1,
                    model_path=model,
                    output_dir=output,
                    release_version="1.0.0",
                    release_manifest_sha256="b" * 64,
                    source_checkpoint_sha256="a" * 64,
                ),
                inputs=inputs,
                sampler=sampler,
                predictor=_FakePredictor(sampler),
            ).run()

            self.assertEqual(
                first["schema_version"], EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION
            )
            self.assertTrue(first["truth_free"])
            self.assertEqual(first["piece_count"], 1)
            record_path = output / first["pieces"][0]["record_path"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], EDGE_HYPOTHESES_SCHEMA_VERSION)
            self.assertEqual(record["candidate_count"], 1)
            self.assertIsNotNone(record["overlay_sha256"])
            self.assertEqual(
                record["coordinate_frames"]["crop_to_source_homography"],
                [[1.0, 0.0, 20.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]],
            )
            candidate = record["candidate_fragments"][0]
            self.assertAlmostEqual(
                candidate["start_xy_source"][0],
                candidate["start_xy_crop"][0] + 20.0,
            )
            self.assertAlmostEqual(
                candidate["start_xy_source"][1],
                candidate["start_xy_crop"][1] + 30.0,
            )
            arrays_path = output / record["arrays_path"]
            with np.load(arrays_path, allow_pickle=False) as arrays:
                self.assertNotIn("seam_target", arrays.files)
                self.assertNotIn("nearest_seam_index", arrays.files)
                self.assertIn("points_xy_source", arrays.files)
                self.assertEqual(len(arrays["role_probabilities"]), record["point_count"])
            verification = verify_stage2_output(output, model_path=model)
            self.assertEqual(verification["status"], "ok")
            self.assertEqual(verification["piece_count"], 1)
            self.assertEqual(
                stage1_hash_before, _file_sha(stage1 / "segmentation_manifest.jsonl")
            )

            resume_sampler = ProductionBoundarySampler(
                window,
                BoundaryGeometryConfig(
                    boundary_spacing_px=2.0,
                    minimum_boundary_points=32,
                    maximum_boundary_points=512,
                    max_inward_thickness_px=32,
                ),
            )
            resumed = Stage2ProductionRunner(
                config=ProductionInferenceConfig(
                    input_path=stage1,
                    model_path=model,
                    output_dir=output,
                    release_version="1.0.0",
                    release_manifest_sha256="b" * 64,
                    source_checkpoint_sha256="a" * 64,
                    resume=True,
                ),
                inputs=inputs,
                sampler=resume_sampler,
                predictor=_FakePredictor(resume_sampler),
            ).run()
            self.assertEqual(resumed["reused_piece_count"], 1)

            arrays_path.write_bytes(arrays_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "arrays SHA-256 mismatch"):
                verify_stage2_output(output)

    def test_output_cannot_be_inside_stage1_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "stage1"
            stage1.mkdir()
            model = root / "model.onnx"
            model.write_bytes(b"model")
            config = ProductionInferenceConfig(
                input_path=stage1,
                model_path=model,
                output_dir=stage1 / "stage2_output",
                release_version="1.0.0",
                release_manifest_sha256="b" * 64,
                source_checkpoint_sha256="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "immutable Stage 1"):
                config.validate()


if __name__ == "__main__":
    unittest.main()


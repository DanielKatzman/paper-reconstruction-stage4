from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.generator import DatasetGenerator
from paper_reconstruction_stage2.observation import (
    ObservationCompiler,
    ObservationConfig,
)


class ObservationCompilerTests(unittest.TestCase):
    def test_observations_are_connected_mapped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            DatasetGenerator(
                GeneratorConfig(
                    output_dir=canonical,
                    puzzle_count=1,
                    seed=17,
                    min_pieces=2,
                    max_pieces=2,
                    page_width=240,
                    page_height=320,
                    min_piece_area_fraction=0.01,
                    min_piece_dimension_px=4,
                    preview_max_side=400,
                )
            ).generate()

            manifests = []
            for run in range(2):
                output = root / f"observations_{run}"
                manifests.append(
                    ObservationCompiler(
                        ObservationConfig(
                            canonical_dataset_dir=canonical,
                            output_dir=output,
                            seed=19,
                            views_per_piece=2,
                            preview_max_side=400,
                        )
                    ).compile()
                )

            self.assertEqual(
                manifests[0]["observation_dataset_fingerprint"],
                manifests[1]["observation_dataset_fingerprint"],
            )
            self.assertEqual(manifests[0]["observation_count"], 4)
            puzzle_summary = manifests[0]["puzzles"][0]
            puzzle_manifest_path = (
                root / "observations_0" / puzzle_summary["observation_manifest_path"]
            )
            puzzle = json.loads(puzzle_manifest_path.read_text(encoding="utf-8"))
            for observation in puzzle["observations"]:
                mask_path = (
                    puzzle_manifest_path.parent / observation["observed_mask_path"]
                )
                with Image.open(mask_path) as opened:
                    mask = np.asarray(opened.convert("L")) > 0
                _, components = ndimage.label(mask)
                self.assertEqual(components, 1)
                area_ratio = (
                    observation["validation"]["observed_area_px"]
                    / observation["validation"]["geometric_area_px"]
                )
                self.assertGreaterEqual(area_ratio, 0.85)
                crop_to_observed = np.asarray(
                    observation["canonical_crop_to_observed_homography"]
                )
                observed_to_crop = np.asarray(
                    observation["observed_to_canonical_crop_homography"]
                )
                roundtrip = observed_to_crop @ crop_to_observed
                roundtrip /= roundtrip[2, 2]
                self.assertTrue(np.allclose(roundtrip, np.eye(3), atol=1e-8))

    def test_nonempty_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            canonical.mkdir()
            (canonical / "dataset_manifest.json").write_text(
                json.dumps({"dataset_fingerprint": "0" * 64, "puzzles": []}),
                encoding="utf-8",
            )
            output = root / "observations"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ObservationCompiler(
                    ObservationConfig(
                        canonical_dataset_dir=canonical, output_dir=output
                    )
                ).compile()
            self.assertEqual(
                (output / "keep.txt").read_text(encoding="utf-8"), "user data"
            )


if __name__ == "__main__":
    unittest.main()

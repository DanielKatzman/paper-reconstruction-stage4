from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_reconstruction_stage2.boundary import (
    BoundaryConfig,
    BoundaryDatasetCompiler,
    _resample_closed_polyline,
    boundary_features,
    extract_closed_contour,
)
from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.generator import DatasetGenerator
from paper_reconstruction_stage2.observation import (
    ObservationCompiler,
    ObservationConfig,
)


class BoundaryGeometryTests(unittest.TestCase):
    def test_rectangle_contour_and_inward_normals(self) -> None:
        mask = np.zeros((50, 70), dtype=bool)
        mask[10:30, 20:50] = True
        contour = extract_closed_contour(mask)
        points, perimeter, spacing = _resample_closed_polyline(
            contour, 2.0, 16, 4096
        )
        self.assertAlmostEqual(perimeter, 100.0, places=5)
        features = boundary_features(
            mask,
            points,
            spacing,
            smoothing_sigma_samples=1.0,
            max_inward_thickness_px=64,
        )
        inward_samples = points + features["inward_normal_xy"] * 2.0
        xs = np.rint(inward_samples[:, 0]).astype(int)
        ys = np.rint(inward_samples[:, 1]).astype(int)
        valid = (
            (xs >= 0)
            & (xs < mask.shape[1])
            & (ys >= 0)
            & (ys < mask.shape[0])
        )
        inside_fraction = float(mask[ys[valid], xs[valid]].mean())
        self.assertGreater(inside_fraction, 0.90)
        self.assertTrue(np.all(features["inward_thickness_px"] >= 0))


class BoundaryCompilerTests(unittest.TestCase):
    def _create_inputs(self, root: Path) -> tuple[Path, Path]:
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
        observations = root / "observations"
        ObservationCompiler(
            ObservationConfig(
                canonical_dataset_dir=canonical,
                output_dir=observations,
                seed=19,
                views_per_piece=2,
                preview_max_side=400,
            )
        ).compile()
        return canonical, observations

    def test_boundary_dataset_is_mapped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical, observations = self._create_inputs(root)
            manifests = []
            for run in range(2):
                output = root / f"boundaries_{run}"
                manifests.append(
                    BoundaryDatasetCompiler(
                        BoundaryConfig(
                            canonical_dataset_dir=canonical,
                            observation_dataset_dir=observations,
                            output_dir=output,
                            preview_max_side=500,
                        )
                    ).compile()
                )
            self.assertEqual(
                manifests[0]["boundary_dataset_fingerprint"],
                manifests[1]["boundary_dataset_fingerprint"],
            )
            self.assertEqual(manifests[0]["observation_count"], 4)
            summary = manifests[0]["puzzles"][0]
            puzzle_dir = (
                root / "boundaries_0" / Path(summary["boundary_manifest_path"]).parent
            )
            puzzle = json.loads(
                (puzzle_dir / "boundary_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(puzzle["validation"]["valid"])
            self.assertGreaterEqual(
                puzzle["validation"]["minimum_aligned_fraction"], 0.80
            )
            self.assertEqual(puzzle["correspondence_track_count"], 4)
            self.assertEqual(puzzle["usable_correspondence_track_count"], 4)
            self.assertTrue(
                puzzle["validation"]["matchable_seams_with_both_owners_covered"]
            )
            self.assertEqual(
                puzzle["validation"]["missing_matchable_owner_track_count"], 0
            )
            self.assertEqual(
                {track["owner_direction"] for track in puzzle["correspondence_tracks"]},
                {"forward", "reverse"},
            )
            with np.load(puzzle_dir / "boundaries.npz", allow_pickle=False) as arrays:
                targets = (
                    arrays["seam_target"]
                    + arrays["page_border_target"]
                    + arrays["uncertainty_target"]
                )
                self.assertTrue(np.allclose(targets, 1.0, atol=1e-5))
                self.assertEqual(len(arrays["observation_offsets"]), 5)
                self.assertTrue(np.all(np.isfinite(arrays["points_xy"])))
            with np.load(
                puzzle_dir / "correspondences.npz", allow_pickle=False
            ) as arrays:
                self.assertEqual(len(arrays["track_offsets"]), 5)
                self.assertGreater(float(arrays["valid"].mean()), 0.90)
            self.assertTrue((puzzle_dir / "previews" / "boundary_roles.png").is_file())
            self.assertTrue((puzzle_dir / "previews" / "seam_ids.png").is_file())

            resumed = BoundaryDatasetCompiler(
                BoundaryConfig(
                    canonical_dataset_dir=canonical,
                    observation_dataset_dir=observations,
                    output_dir=root / "boundaries_0",
                    preview_max_side=500,
                    resume=True,
                )
            ).compile()
            self.assertEqual(
                resumed["boundary_dataset_fingerprint"],
                manifests[0]["boundary_dataset_fingerprint"],
            )

    def test_nonempty_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical, observations = self._create_inputs(root)
            output = root / "boundaries"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                BoundaryDatasetCompiler(
                    BoundaryConfig(
                        canonical_dataset_dir=canonical,
                        observation_dataset_dir=observations,
                        output_dir=output,
                    )
                ).compile()
            self.assertEqual(
                (output / "keep.txt").read_text(encoding="utf-8"), "user data"
            )


if __name__ == "__main__":
    unittest.main()

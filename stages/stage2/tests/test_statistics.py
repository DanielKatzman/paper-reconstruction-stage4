from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_reconstruction_stage2.statistics import (
    StatisticsConfig,
    TrainingStatisticsCompiler,
)
from paper_reconstruction_stage2.role_pipeline import (
    BoundaryRolePipeline,
    BoundaryRolePipelineConfig,
)
from paper_reconstruction_stage2.training_data import GEOMETRY_FEATURE_NAMES


class StatisticsTests(unittest.TestCase):
    def test_statistics_are_resumable_and_exclude_truth_only_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations"
            boundaries = root / "boundaries"
            puzzle_dir = boundaries / "puzzles" / "puzzle_000"
            observations.mkdir()
            puzzle_dir.mkdir(parents=True)
            (observations / "observation_dataset_manifest.json").write_text(
                json.dumps({"observation_dataset_fingerprint": "observed-1"}),
                encoding="utf-8",
            )
            (boundaries / "boundary_dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "boundary_dataset_fingerprint": "boundary-1",
                        "observation_dataset_fingerprint": "observed-1",
                        "puzzles": [
                            {
                                "boundary_manifest_path": (
                                    "puzzles/puzzle_000/boundary_manifest.json"
                                )
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            point_count = 6
            observation = {
                "observation_index": 0,
                "observation_id": "piece_001__view_00",
                "piece_id": "piece_001",
                "view_index": 0,
                "observed_piece_dataset_path": "unused.png",
                "observed_mask_dataset_path": "unused_mask.png",
                "point_count": point_count,
                "boundary_offset_start": 0,
                "boundary_offset_end": point_count,
                "hard_role_counts": {
                    "seam": 3,
                    "page_border": 3,
                    "uncertain": 0,
                },
            }
            (puzzle_dir / "boundary_manifest.json").write_text(
                json.dumps(
                    {
                        "puzzle_id": "puzzle_000",
                        "split": "train",
                        "difficulty_tier": "simple",
                        "boundaries_path": "boundaries.npz",
                        "observations": [observation],
                    }
                ),
                encoding="utf-8",
            )
            seam = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float32)
            border = 1.0 - seam
            np.savez_compressed(
                puzzle_dir / "boundaries.npz",
                points_xy=np.zeros((point_count, 2), dtype=np.float32),
                tangent_xy=np.tile((1.0, 0.0), (point_count, 1)).astype(np.float32),
                inward_normal_xy=np.tile((0.0, 1.0), (point_count, 1)).astype(np.float32),
                curvature_per_px=np.arange(point_count, dtype=np.float32),
                corner_strength=np.zeros(point_count, dtype=np.float32),
                inward_thickness_px=np.full(point_count, 10, dtype=np.float32),
                seam_target=seam,
                page_border_target=border,
                uncertainty_target=np.zeros(point_count, dtype=np.float32),
                label_confidence=np.ones(point_count, dtype=np.float32),
                nearest_truth_distance_px=np.full(point_count, 999, dtype=np.float32),
                observation_offsets=np.asarray([0, point_count], dtype=np.int64),
            )
            output = root / "statistics.json"
            report = TrainingStatisticsCompiler(
                StatisticsConfig(boundaries, observations, output)
            ).compile()
            self.assertEqual(
                report["geometry"]["feature_names"], list(GEOMETRY_FEATURE_NAMES)
            )
            self.assertEqual(len(report["geometry"]["mean"]), 7)
            self.assertNotIn("nearest_truth_distance_px", report["geometry"]["feature_names"])
            resumed = TrainingStatisticsCompiler(
                StatisticsConfig(boundaries, observations, output, resume=True)
            ).compile()
            self.assertEqual(resumed["status"], "complete")

            pipeline_output = root / "role_pipeline"
            state = BoundaryRolePipeline(
                BoundaryRolePipelineConfig(
                    boundary_dataset_dir=boundaries,
                    observation_dataset_dir=observations,
                    output_dir=pipeline_output,
                    stop_after="statistics",
                )
            ).run()
            self.assertEqual(state["stages"]["statistics"], "complete")
            self.assertEqual(state["status"], "stopped_after_stage")
            resumed_state = BoundaryRolePipeline(
                BoundaryRolePipelineConfig(
                    boundary_dataset_dir=boundaries,
                    observation_dataset_dir=observations,
                    output_dir=pipeline_output,
                    stop_after="statistics",
                    resume=True,
                )
            ).run()
            self.assertEqual(resumed_state["stages"]["statistics"], "complete")


if __name__ == "__main__":
    unittest.main()

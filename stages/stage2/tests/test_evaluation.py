from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from paper_reconstruction_stage2.evaluation import evaluate_full_contours
from paper_reconstruction_stage2.fragments import FragmentProposalConfig


class EvaluationDiagnosticTests(unittest.TestCase):
    def test_strict_fragment_failure_writes_manifest_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (96, 96), "white").save(root / "piece.png")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint")

            point_count = 40
            angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
            points = np.column_stack(
                (48 + 30 * np.cos(angles), 48 + 30 * np.sin(angles))
            ).astype(np.float32)
            targets = np.zeros((point_count, 3), dtype=np.float32)
            targets[:, 1] = 1.0
            targets[10:25] = (1.0, 0.0, 0.0)
            probabilities = np.full((point_count, 3), 0.025, dtype=np.float32)
            probabilities[:, 1] = 0.95
            probabilities[12:23] = (0.95, 0.025, 0.025)

            entry = SimpleNamespace(
                puzzle_id="puzzle_000",
                observation_id="piece_000_view_00",
                piece_id="piece_000",
                view_index=0,
                difficulty_tier="hard",
                puzzle_piece_count=8,
                point_count=point_count,
                image_path="piece.png",
            )
            index = SimpleNamespace(
                by_split={"validation": (0,)},
                entries=(entry,),
                observation_dataset_dir=root,
                boundary_dataset={"boundary_dataset_fingerprint": "boundary"},
                observation_dataset={"observation_dataset_fingerprint": "observation"},
            )

            class Sampler:
                @staticmethod
                def full_observation(_entry):
                    return {"points_xy": points, "targets": targets}

            class Predictor:
                fragment_config = FragmentProposalConfig(minimum_fragment_points=1)

                @staticmethod
                def predict(_entry):
                    return {
                        "role_probabilities": probabilities,
                        "transition_probabilities": np.full(
                            point_count, 0.1, dtype=np.float32
                        ),
                        "candidate_fragments": [
                            {
                                "start_index": 12,
                                "end_index": 22,
                                "length_points": 11,
                                "score": 0.9,
                                "rank": 1,
                            }
                        ],
                    }

            diagnostics = root / "failures"
            report = evaluate_full_contours(
                checkpoint_path=checkpoint,
                split="validation",
                index=index,
                sampler=Sampler(),
                predictor=Predictor(),
                output_path=root / "evaluation.json",
                failure_diagnostics_dir=diagnostics,
            )
            self.assertEqual(report["schema_version"], "stage2.boundary-role-evaluation.v3")
            self.assertEqual(report["failure_diagnostics"]["failure_count"], 1)
            self.assertEqual(report["fragments_by_difficulty"]["hard"]["truth_count"], 1)
            self.assertIn("recall_at_1_iou_50", report["fragments"])
            self.assertIn("recall_at_3_iou_80", report["fragments"])
            self.assertIn("recall_at_10_iou_80", report["fragments"])
            self.assertEqual(report["candidate_statistics"]["observation_count"], 1)
            self.assertTrue((diagnostics / "failure_diagnostics.json").is_file())
            self.assertEqual(len(list(diagnostics.glob("*.png"))), 1)

    def test_rank_six_candidate_is_reported_as_top_ten_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (96, 96), "white").save(root / "piece.png")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint")

            point_count = 60
            angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
            points = np.column_stack(
                (48 + 30 * np.cos(angles), 48 + 30 * np.sin(angles))
            ).astype(np.float32)
            targets = np.zeros((point_count, 3), dtype=np.float32)
            targets[:, 1] = 1.0
            targets[20:30] = (1.0, 0.0, 0.0)
            probabilities = np.full((point_count, 3), 0.025, dtype=np.float32)
            probabilities[:, 1] = 0.95
            probabilities[20:30] = (0.95, 0.025, 0.025)

            entry = SimpleNamespace(
                puzzle_id="puzzle_006",
                observation_id="piece_006_view_00",
                piece_id="piece_006",
                view_index=0,
                difficulty_tier="extreme",
                puzzle_piece_count=30,
                point_count=point_count,
                image_path="piece.png",
            )
            index = SimpleNamespace(
                by_split={"validation": (0,)},
                entries=(entry,),
                observation_dataset_dir=root,
                boundary_dataset={"boundary_dataset_fingerprint": "boundary"},
                observation_dataset={"observation_dataset_fingerprint": "observation"},
            )

            class Sampler:
                @staticmethod
                def full_observation(_entry):
                    return {"points_xy": points, "targets": targets}

            false_candidates = [
                {
                    "start_index": start,
                    "end_index": start + 1,
                    "length_points": 2,
                    "score": 1.0 - rank * 0.05,
                    "rank": rank,
                }
                for rank, start in enumerate((0, 2, 4, 6, 8), start=1)
            ]
            recovered_candidate = {
                "start_index": 20,
                "end_index": 29,
                "length_points": 10,
                "score": 0.5,
                "rank": 6,
            }

            class Predictor:
                fragment_config = FragmentProposalConfig(minimum_fragment_points=1)

                @staticmethod
                def predict(_entry):
                    return {
                        "role_probabilities": probabilities,
                        "transition_probabilities": np.full(
                            point_count, 0.1, dtype=np.float32
                        ),
                        "candidate_fragments": [
                            *false_candidates,
                            recovered_candidate,
                        ],
                    }

            diagnostics = root / "failures"
            report = evaluate_full_contours(
                checkpoint_path=checkpoint,
                split="validation",
                index=index,
                sampler=Sampler(),
                predictor=Predictor(),
                output_path=root / "evaluation.json",
                failure_diagnostics_dir=diagnostics,
            )

            self.assertEqual(report["fragments"]["recall_at_5_iou_50"], 0.0)
            self.assertEqual(report["fragments"]["recall_at_10_iou_50"], 1.0)
            self.assertEqual(report["fragments"]["recall_at_10_iou_80"], 1.0)
            self.assertEqual(
                report["failure_diagnostics"]["recovered_at_10_iou_80"], 1
            )
            self.assertEqual(
                report["failure_diagnostics"]["remaining_at_10_iou_80"], 0
            )
            self.assertEqual(report["candidate_statistics"]["mean"], 6.0)
            self.assertEqual(report["candidate_statistics"]["maximum"], 6)
            self.assertEqual(report["candidate_statistics"]["observations_over_5"], 1)

            diagnostics_payload = json.loads(
                (diagnostics / "failure_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            failure = diagnostics_payload["failures"][0]
            self.assertEqual(failure["top_10_candidate_rank"], 6)
            self.assertEqual(failure["top_10_best_iou"], 1.0)
            self.assertTrue(failure["recovered_at_10_iou_50"])
            self.assertTrue(failure["recovered_at_10_iou_80"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

from paper_reconstruction_stage2.experiments import (
    experiment_selection_score,
    rank_experiment_reports,
)
from paper_reconstruction_stage2.model_candidates import (
    candidate_model_config,
    candidate_names,
)
from paper_reconstruction_stage2.profiling import select_profile_recommendations
from paper_reconstruction_stage2.training import BoundaryTrainingConfig


def _report(
    point: float, recall_50: float, recall_80: float, transition_f1: float = 0.8
) -> dict:
    return {
        "selection_score": point,
        "metrics": {"transition": {"f1": transition_f1}},
        "fragments": {
            "recall_at_5_iou_50": recall_50,
            "recall_at_5_iou_80": recall_80,
        },
    }


class ExperimentPlanningTests(unittest.TestCase):
    def test_candidate_registry_contains_controlled_ablations(self) -> None:
        self.assertEqual(len(candidate_names()), 8)
        self.assertFalse(candidate_model_config("geometry_only").use_raster)
        self.assertFalse(candidate_model_config("raster_only").use_geometry)
        self.assertEqual(
            candidate_model_config("hybrid_attention").architecture,
            "hybrid_attention",
        )
        self.assertTrue(
            candidate_model_config("global_context_hybrid").use_global_context
        )
        self.assertGreater(
            candidate_model_config("xlarge_hybrid_attention").fusion_channels,
            candidate_model_config("hybrid_attention").fusion_channels,
        )

    def test_experiment_ranking_uses_fragment_quality(self) -> None:
        point_only = _report(0.95, 0.20, 0.10)
        useful_edges = _report(0.85, 0.90, 0.80)
        self.assertGreater(
            experiment_selection_score(useful_edges),
            experiment_selection_score(point_only),
        )
        ranking = rank_experiment_reports(
            {"point_only": point_only, "useful_edges": useful_edges}
        )
        self.assertEqual(ranking[0]["model"], "useful_edges")

    def test_profiler_selects_common_safe_recipe(self) -> None:
        records = []
        for model, speed in (("a", 100.0), ("b", 80.0)):
            records.extend(
                [
                    {
                        "phase": "model_matrix",
                        "status": "ok",
                        "model": model,
                        "batch_size": 32,
                        "workers": 2,
                        "precision": "fp32",
                        "windows_per_second": speed,
                        "peak_allocated_mb": 2000.0,
                    },
                    {
                        "phase": "model_matrix",
                        "status": "ok",
                        "model": model,
                        "batch_size": 64,
                        "workers": 2,
                        "precision": "fp32",
                        "windows_per_second": speed * 1.5,
                        "peak_allocated_mb": 7500.0,
                    },
                ]
            )
        common, per_model = select_profile_recommendations(
            records,
            ["a", "b"],
            memory_headroom=0.82,
            total_device_mb=8192.0,
        )
        self.assertEqual(common["batch_size"], 32)
        self.assertEqual(set(per_model), {"a", "b"})

    def test_loader_tuning_does_not_change_resume_contract(self) -> None:
        common = dict(
            boundary_dataset_dir=Path("boundaries"),
            observation_dataset_dir=Path("observations"),
            statistics_path=Path("statistics.json"),
            output_dir=Path("training"),
        )
        first = BoundaryTrainingConfig(
            **common, num_workers=0, prefetch_factor=1
        ).public_dict()
        second = BoundaryTrainingConfig(
            **common, num_workers=4, prefetch_factor=4
        ).public_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

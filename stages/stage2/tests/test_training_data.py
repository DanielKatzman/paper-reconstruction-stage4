from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from paper_reconstruction_stage2.boundary import BoundaryConfig, BoundaryDatasetCompiler
from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.generator import DatasetGenerator
from paper_reconstruction_stage2.observation import ObservationCompiler, ObservationConfig
from paper_reconstruction_stage2.parallel_data import DeterministicBoundaryWindowDataset
from paper_reconstruction_stage2.training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


class BoundaryTrainingDataTests(unittest.TestCase):
    def test_soft_transition_below_half_uses_strongest_points(self) -> None:
        sampler = BoundaryWindowSampler(SimpleNamespace())
        entry = SimpleNamespace(
            observation_id="soft_transition",
            boundaries_path="puzzles/puzzle_000/boundaries.npz",
            observation_index=0,
            boundary_offset_start=0,
            boundary_offset_end=12,
        )
        arrays = {
            "seam_target": np.full(12, 0.30, dtype=np.float32),
            "page_border_target": np.full(12, 0.30, dtype=np.float32),
            "uncertainty_target": np.full(12, 0.40, dtype=np.float32),
            "label_confidence": np.ones(12, dtype=np.float32),
        }
        indices = sampler._transition_indices(arrays, entry)
        self.assertGreater(len(indices), 0)
        self.assertTrue(np.all((indices >= 0) & (indices < 12)))

    def test_transition_cache_is_unique_across_repeated_observation_ids(self) -> None:
        sampler = BoundaryWindowSampler(SimpleNamespace())

        def values(count: int, peak: int) -> dict[str, np.ndarray]:
            seam = np.full(count, 0.50, dtype=np.float32)
            border = np.full(count, 0.50, dtype=np.float32)
            uncertain = np.zeros(count, dtype=np.float32)
            seam[peak] = 0.0
            border[peak] = 0.0
            uncertain[peak] = 1.0
            return {
                "seam_target": seam,
                "page_border_target": border,
                "uncertainty_target": uncertain,
                "label_confidence": np.ones(count, dtype=np.float32),
            }

        first = SimpleNamespace(
            observation_id="piece_004__view_01",
            boundaries_path="puzzles/puzzle_001/boundaries.npz",
            observation_index=9,
            boundary_offset_start=0,
            boundary_offset_end=20,
        )
        second = SimpleNamespace(
            observation_id="piece_004__view_01",
            boundaries_path="puzzles/puzzle_002/boundaries.npz",
            observation_index=9,
            boundary_offset_start=0,
            boundary_offset_end=10,
        )
        first_indices = sampler._transition_indices(values(20, 18), first)
        second_indices = sampler._transition_indices(values(10, 4), second)
        self.assertTrue(np.any(first_indices >= 18))
        self.assertTrue(np.all(second_indices < 10))
        self.assertTrue(np.any(second_indices == 4))

    def test_streaming_window_sampler_shapes_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            DatasetGenerator(
                GeneratorConfig(
                    output_dir=canonical,
                    puzzle_count=1,
                    seed=71,
                    min_pieces=2,
                    max_pieces=2,
                    page_width=220,
                    page_height=300,
                    min_piece_area_fraction=0.01,
                    min_piece_dimension_px=4,
                    preview_max_side=320,
                )
            ).generate()
            observations = root / "observations"
            ObservationCompiler(
                ObservationConfig(
                    canonical_dataset_dir=canonical,
                    output_dir=observations,
                    seed=73,
                    views_per_piece=2,
                    preview_max_side=320,
                )
            ).compile()
            boundaries = root / "boundaries"
            BoundaryDatasetCompiler(
                BoundaryConfig(
                    canonical_dataset_dir=canonical,
                    observation_dataset_dir=observations,
                    output_dir=boundaries,
                    preview_max_side=400,
                )
            ).compile()

            index = BoundaryTrainingIndex(boundaries, observations)
            self.assertEqual(index.summary()["observation_count"], 4)
            split = index.entries[0].split
            sampler = BoundaryWindowSampler(
                index,
                BoundaryWindowConfig(sequence_length=17, strip_width=9),
            )
            sample = sampler.sample(
                np.random.default_rng(79), split=split, role="seam"
            )
            self.assertEqual(sample["geometry"].shape, (17, 7))
            self.assertEqual(sample["targets"].shape, (17, 3))
            self.assertEqual(sample["raster"].shape, (4, 17, 9))
            self.assertTrue(np.all(np.isfinite(sample["geometry"])))
            self.assertTrue(np.allclose(sample["targets"].sum(axis=1), 1.0))
            self.assertEqual(sample["sampled_role"], "seam")
            self.assertEqual(sample["boundary_indices_local"].shape, (17,))
            transition = sampler.sample(
                np.random.default_rng(80),
                split=split,
                role="transition",
                include_global_context=True,
                global_context_points=32,
                global_mask_size=24,
            )
            self.assertEqual(transition["sampled_role"], "transition")
            self.assertEqual(transition["global_geometry"].shape, (32, 9))
            self.assertEqual(transition["global_mask"].shape, (1, 24, 24))
            self.assertTrue(np.all(np.isfinite(transition["global_geometry"])))
            natural = sampler.sample(
                np.random.default_rng(81),
                split=split,
                natural=True,
                include_raster=False,
            )
            self.assertEqual(natural["sampled_role"], "natural")
            self.assertNotIn("raster", natural)
            full = sampler.full_observation(index.entries[0])
            self.assertEqual(full["geometry"].shape[1], 7)
            self.assertEqual(len(full["points_xy"]), index.entries[0].point_count)

            deterministic = DeterministicBoundaryWindowDataset(
                boundaries,
                observations,
                window_config=BoundaryWindowConfig(
                    sequence_length=17, strip_width=9
                ),
                split=split,
                sample_count=4,
                seed=83,
                epoch=2,
                include_raster=False,
                role_sampling_weights={
                    "seam": 1.0,
                    "page_border": 0.0,
                    "uncertain": 0.0,
                },
                stream_id=17,
            )
            later = deterministic[3]
            earlier = deterministic[1]
            repeated = deterministic[3]
            self.assertTrue(np.array_equal(later["geometry"], repeated["geometry"]))
            self.assertTrue(np.array_equal(later["targets"], repeated["targets"]))
            self.assertEqual(earlier["geometry"].shape, (17, 7))


if __name__ == "__main__":
    unittest.main()

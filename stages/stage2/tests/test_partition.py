from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.partition import generate_partition
from paper_reconstruction_stage2.truth import extract_truth
from paper_reconstruction_stage2.validation import validate_partition


def test_config(directory: str, max_pieces: int = 30) -> GeneratorConfig:
    return GeneratorConfig(
        output_dir=Path(directory) / "out",
        puzzle_count=1,
        min_pieces=2,
        max_pieces=max_pieces,
        page_width=320,
        page_height=452,
        min_piece_area_fraction=0.0015,
        min_piece_dimension_px=4,
        max_split_attempts=500,
        preview_max_side=500,
    )


class PartitionTests(unittest.TestCase):
    def test_two_pieces_means_one_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(directory, max_pieces=2)
            result = generate_partition(
                config, np.random.default_rng(11), target_piece_count=2
            )
            self.assertEqual(result.target_piece_count, 2)
            self.assertEqual(result.successful_cut_count, 1)
            self.assertEqual(len(np.unique(result.labels)), 2)

    def test_deterministic_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(directory)
            first = generate_partition(
                config, np.random.default_rng(991), target_piece_count=12
            )
            second = generate_partition(
                config, np.random.default_rng(991), target_piece_count=12
            )
            np.testing.assert_array_equal(first.labels, second.labels)
            self.assertEqual(
                [event.to_dict() for event in first.cut_events],
                [event.to_dict() for event in second.cut_events],
            )

    def test_thirty_piece_stress_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(directory)
            result = generate_partition(
                config, np.random.default_rng(12345), target_piece_count=30
            )
            pieces, seams, _ = extract_truth(result.labels)
            report = validate_partition(result, pieces, seams, config)
            self.assertTrue(report["valid"])
            self.assertEqual(report["piece_count"], 30)
            self.assertEqual(result.successful_cut_count, 29)
            self.assertGreaterEqual(len(seams), 29)


if __name__ == "__main__":
    unittest.main()


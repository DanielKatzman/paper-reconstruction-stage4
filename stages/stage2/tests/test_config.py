from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_reconstruction_stage2.config import GeneratorConfig


class GeneratorConfigTests(unittest.TestCase):
    def test_two_pieces_is_minimum_valid_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = GeneratorConfig(
                output_dir=Path(directory) / "out",
                min_pieces=2,
                max_pieces=2,
            )
            config.validate()

    def test_one_piece_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = GeneratorConfig(
                output_dir=Path(directory) / "out",
                min_pieces=1,
                max_pieces=2,
            )
            with self.assertRaisesRegex(ValueError, "at least 2"):
                config.validate()

    def test_impossible_minimum_area_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = GeneratorConfig(
                output_dir=Path(directory) / "out",
                min_pieces=2,
                max_pieces=30,
                min_piece_area_fraction=0.03,
            )
            with self.assertRaisesRegex(ValueError, "too large"):
                config.validate()


if __name__ == "__main__":
    unittest.main()


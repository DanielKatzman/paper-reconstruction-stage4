from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_reconstruction_stage2.config import GeneratorConfig
from paper_reconstruction_stage2.generator import DatasetGenerator


class DatasetGeneratorTests(unittest.TestCase):
    def test_end_to_end_two_piece_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            config = GeneratorConfig(
                output_dir=output,
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
            manifest = DatasetGenerator(config).generate()
            self.assertEqual(len(manifest["puzzles"]), 1)
            summary = manifest["puzzles"][0]
            self.assertEqual(summary["target_piece_count"], 2)
            self.assertEqual(summary["successful_cut_count"], 1)
            puzzle_path = output / summary["puzzle_path"]
            puzzle = json.loads(puzzle_path.read_text(encoding="utf-8"))
            self.assertEqual(puzzle["actual_piece_count"], 2)
            self.assertTrue(puzzle["validation"]["valid"])
            self.assertEqual(puzzle["schema_version"], "stage2.canonical-puzzle.v2")
            self.assertIn("geometry_metrics", puzzle)
            self.assertIn("geometry_metrics", puzzle["pieces"][0])
            self.assertIn("geometry_metrics", puzzle["seams"][0])
            self.assertIn(puzzle["split"], config.split_ratios)
            self.assertIn(puzzle["difficulty_tier"], config.difficulty_mix)
            self.assertTrue((puzzle_path.parent / "truth.npz").exists())
            self.assertTrue((output / "source_split_manifest.json").exists())
            labels = np.load(puzzle_path.parent / "labels.npy", allow_pickle=False)
            self.assertEqual(len(np.unique(labels)), 2)

    def test_nonempty_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            config = GeneratorConfig(output_dir=output, puzzle_count=1)
            with self.assertRaises(FileExistsError):
                DatasetGenerator(config).generate()
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()

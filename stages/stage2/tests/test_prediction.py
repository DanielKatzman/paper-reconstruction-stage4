from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from paper_reconstruction_stage2.prediction import render_role_overlay


class PredictionTests(unittest.TestCase):
    def test_role_overlay_is_written_at_source_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "piece.png"
            output = root / "overlay.png"
            Image.new("RGB", (40, 30), "white").save(source)
            points = np.asarray([[5, 5], [34, 5], [34, 24], [5, 24]], dtype=np.float32)
            probabilities = np.eye(3, dtype=np.float32)[[0, 1, 2, 0]]
            render_role_overlay(source, points, probabilities, output)
            self.assertTrue(output.is_file())
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (40, 30))


if __name__ == "__main__":
    unittest.main()

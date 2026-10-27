from __future__ import annotations

import unittest

import numpy as np

from paper_reconstruction_stage2.model_metrics import (
    StreamingBoundaryMetrics,
    selection_score,
)


class ModelMetricTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_role_metrics(self) -> None:
        targets = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1, 2]]
        transitions = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        metrics = StreamingBoundaryMetrics()
        metrics.update(targets, targets, transitions, transitions)
        result = metrics.compute()
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["transition"]["f1"], 1.0)
        self.assertGreater(selection_score(result), 0.99)


if __name__ == "__main__":
    unittest.main()

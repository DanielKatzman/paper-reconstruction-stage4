from __future__ import annotations

import unittest

import numpy as np

from paper_reconstruction_stage2.calibration import (
    fit_role_temperature,
    fit_transition_temperature,
    role_probabilities,
)


class CalibrationTests(unittest.TestCase):
    def test_role_temperature_softens_overconfident_logits(self) -> None:
        logits = np.asarray(
            [[8.0, -4.0, -4.0], [-4.0, 8.0, -4.0], [-4.0, -4.0, 8.0]] * 20
        )
        targets = np.asarray(
            [[0.75, 0.15, 0.10], [0.10, 0.75, 0.15], [0.15, 0.10, 0.75]] * 20
        )
        temperature = fit_role_temperature(logits, targets)
        self.assertGreater(temperature, 1.0)
        before = -np.mean(np.sum(targets * np.log(role_probabilities(logits, 1.0)), axis=1))
        after = -np.mean(
            np.sum(targets * np.log(role_probabilities(logits, temperature)), axis=1)
        )
        self.assertLess(after, before)

    def test_transition_temperature_is_positive_and_bounded(self) -> None:
        logits = np.asarray([-8.0, 8.0] * 30)
        targets = np.asarray([0.20, 0.80] * 30)
        temperature = fit_transition_temperature(logits, targets)
        self.assertGreaterEqual(temperature, 0.20)
        self.assertLessEqual(temperature, 5.0)


if __name__ == "__main__":
    unittest.main()

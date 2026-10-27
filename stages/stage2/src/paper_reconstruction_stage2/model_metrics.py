from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .training_data import ROLE_NAMES


@dataclass(slots=True)
class StreamingBoundaryMetrics:
    calibration_bins: int = 15
    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.int64)
    )
    transition_confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.int64)
    )
    point_count: int = 0
    brier_sum: float = 0.0
    negative_log_likelihood_sum: float = 0.0
    calibration_count: np.ndarray = field(init=False)
    calibration_confidence_sum: np.ndarray = field(init=False)
    calibration_correct_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.calibration_count = np.zeros(self.calibration_bins, dtype=np.int64)
        self.calibration_confidence_sum = np.zeros(
            self.calibration_bins, dtype=np.float64
        )
        self.calibration_correct_sum = np.zeros(
            self.calibration_bins, dtype=np.float64
        )

    def update(
        self,
        role_probabilities: np.ndarray,
        role_targets: np.ndarray,
        transition_probabilities: np.ndarray | None = None,
        transition_targets: np.ndarray | None = None,
    ) -> None:
        probabilities = np.asarray(role_probabilities, dtype=np.float64).reshape(-1, 3)
        targets = np.asarray(role_targets, dtype=np.float64).reshape(-1, 3)
        if probabilities.shape != targets.shape:
            raise ValueError("role probabilities and targets must have the same shape")
        predicted = np.argmax(probabilities, axis=1)
        actual = np.argmax(targets, axis=1)
        np.add.at(self.confusion, (actual, predicted), 1)
        self.point_count += len(actual)
        self.brier_sum += float(np.square(probabilities - targets).sum())
        self.negative_log_likelihood_sum += float(
            -(targets * np.log(np.clip(probabilities, 1e-8, 1.0))).sum()
        )

        confidence = probabilities.max(axis=1)
        correct = (predicted == actual).astype(np.float64)
        bins = np.minimum(
            (confidence * self.calibration_bins).astype(np.int64),
            self.calibration_bins - 1,
        )
        np.add.at(self.calibration_count, bins, 1)
        np.add.at(self.calibration_confidence_sum, bins, confidence)
        np.add.at(self.calibration_correct_sum, bins, correct)

        if transition_probabilities is not None and transition_targets is not None:
            transition_predicted = (
                np.asarray(transition_probabilities).reshape(-1) >= 0.5
            ).astype(np.int64)
            transition_actual = (
                np.asarray(transition_targets).reshape(-1) >= 0.5
            ).astype(np.int64)
            np.add.at(
                self.transition_confusion,
                (transition_actual, transition_predicted),
                1,
            )

    @staticmethod
    def _class_metrics(confusion: np.ndarray, index: int) -> dict[str, float | int]:
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }

    def compute(self) -> dict[str, Any]:
        roles = {
            name: self._class_metrics(self.confusion, index)
            for index, name in enumerate(ROLE_NAMES)
        }
        macro_f1 = float(np.mean([record["f1"] for record in roles.values()]))
        total = max(self.point_count, 1)
        expected_calibration_error = 0.0
        calibration: list[dict[str, float | int]] = []
        for index in range(self.calibration_bins):
            count = int(self.calibration_count[index])
            if count:
                confidence = float(self.calibration_confidence_sum[index] / count)
                accuracy = float(self.calibration_correct_sum[index] / count)
                expected_calibration_error += count / total * abs(confidence - accuracy)
            else:
                confidence = 0.0
                accuracy = 0.0
            calibration.append(
                {
                    "bin": index,
                    "count": count,
                    "mean_confidence": confidence,
                    "accuracy": accuracy,
                }
            )
        transition = self._class_metrics(self.transition_confusion, 1)
        return {
            "point_count": self.point_count,
            "confusion": self.confusion.tolist(),
            "roles": roles,
            "macro_f1": macro_f1,
            "accuracy": float(np.trace(self.confusion) / total),
            "brier_score": self.brier_sum / total,
            "negative_log_likelihood": self.negative_log_likelihood_sum / total,
            "expected_calibration_error": expected_calibration_error,
            "calibration_bins": calibration,
            "transition": transition,
            "transition_confusion": self.transition_confusion.tolist(),
        }


def selection_score(metrics: dict[str, Any]) -> float:
    roles = metrics["roles"]
    return float(
        0.50 * roles["seam"]["f1"]
        + 0.25 * roles["page_border"]["f1"]
        + 0.15 * metrics["transition"]["f1"]
        + 0.10 * (1.0 - metrics["expected_calibration_error"])
    )

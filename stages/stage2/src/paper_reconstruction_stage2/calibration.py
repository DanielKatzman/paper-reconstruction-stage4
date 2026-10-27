from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .evaluation import numpy_transition_targets
from .fragments import (
    FragmentProposalConfig,
    fragment_recall,
    propose_fragments,
    true_seam_fragments,
)
from .inference import FullContourPredictor, checkpoint_sha256
from .model_metrics import StreamingBoundaryMetrics
from .training_data import BoundaryTrainingIndex, BoundaryWindowSampler


try:
    import torch
except ImportError:
    torch = None


CALIBRATION_SCHEMA_VERSION = "stage2.boundary-role-calibration.v2"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    if torch is None:
        raise RuntimeError("checkpoint calibration requires PyTorch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def role_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=-1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def transition_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.clip(np.asarray(logits, dtype=np.float64) / float(temperature), -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def fit_role_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    if values.shape != truth.shape or not len(values):
        raise ValueError("role logits and targets must be nonempty arrays of shape [N, 3]")

    def objective(log_temperature: float) -> float:
        probabilities = role_probabilities(values, math.exp(log_temperature))
        return float(
            -(truth * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=1).mean()
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.20), math.log(5.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(f"role temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def fit_transition_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    truth = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.shape != truth.shape or not len(values):
        raise ValueError("transition logits and targets must be nonempty equal-length arrays")

    def objective(log_temperature: float) -> float:
        probabilities = transition_probabilities(values, math.exp(log_temperature))
        return float(
            -(
                truth * np.log(np.clip(probabilities, 1e-9, 1.0))
                + (1.0 - truth) * np.log(np.clip(1.0 - probabilities, 1e-9, 1.0))
            ).mean()
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.20), math.log(5.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(
            f"transition temperature optimization failed: {result.message}"
        )
    return float(math.exp(result.x))


@dataclass(slots=True)
class CalibrationConfig:
    checkpoint_path: Path
    output_checkpoint_path: Path
    report_path: Path
    maximum_observations: int | None = 512
    maximum_temperature_points: int = 2_000_000
    seed: int = 20261006

    def validate(self) -> None:
        if self.maximum_observations is not None and self.maximum_observations < 1:
            raise ValueError("maximum_observations must be positive when provided")
        if self.maximum_temperature_points < 1:
            raise ValueError("maximum_temperature_points must be positive")
        if self.output_checkpoint_path.resolve() == self.checkpoint_path.resolve():
            raise ValueError("calibration output must not overwrite the input checkpoint")


def _deterministic_subsample(
    arrays: list[np.ndarray], maximum_points: int
) -> list[np.ndarray]:
    point_count = len(arrays[0])
    if point_count <= maximum_points:
        return arrays
    indices = np.linspace(0, point_count - 1, maximum_points, dtype=np.int64)
    return [array[indices] for array in arrays]


def _fragment_config_from_thresholds(
    base: FragmentProposalConfig,
    high: float,
    low: float,
    transition: float,
) -> FragmentProposalConfig:
    return replace(
        base,
        high_threshold=high,
        low_threshold=low,
        relaxed_high_threshold=max(0.05, high - 0.10),
        relaxed_low_threshold=max(0.01, low - 0.10),
        strict_high_threshold=min(0.99, high + 0.10),
        strict_low_threshold=min(min(0.99, high + 0.10), low + 0.10),
        transition_split_threshold=transition,
        relaxed_transition_split_threshold=max(0.20, transition - 0.25),
    )


def select_fragment_config(
    contours: list[dict[str, np.ndarray]],
    *,
    role_temperature: float,
    transition_temperature: float,
    base: FragmentProposalConfig | None = None,
) -> tuple[FragmentProposalConfig, dict[str, Any]]:
    settings = base or FragmentProposalConfig()
    candidates: list[dict[str, Any]] = []
    for high in (0.50, 0.60, 0.70, 0.80):
        for low in (0.25, 0.35, 0.45, 0.55):
            if low > high:
                continue
            for transition in (0.50, 0.65, 0.80):
                proposal_config = _fragment_config_from_thresholds(
                    settings, high, low, transition
                )
                truth_count = 0
                matches_50 = 0
                matches_80 = 0
                candidate_count = 0
                for contour in contours:
                    probabilities = role_probabilities(
                        contour["role_logits"], role_temperature
                    )
                    transition_values = transition_probabilities(
                        contour["transition_logits"], transition_temperature
                    )
                    proposals = propose_fragments(
                        probabilities, transition_values, proposal_config
                    )
                    truths = true_seam_fragments(
                        contour["targets"], proposal_config.minimum_fragment_points
                    )
                    matched_50, count, _ = fragment_recall(
                        proposals,
                        truths,
                        len(probabilities),
                        top_k=5,
                        iou_threshold=0.50,
                    )
                    matched_80, _, _ = fragment_recall(
                        proposals,
                        truths,
                        len(probabilities),
                        top_k=5,
                        iou_threshold=0.80,
                    )
                    truth_count += count
                    matches_50 += matched_50
                    matches_80 += matched_80
                    candidate_count += len(proposals)
                recall_50 = matches_50 / max(truth_count, 1)
                recall_80 = matches_80 / max(truth_count, 1)
                mean_candidates = candidate_count / max(len(contours), 1)
                # Accurate endpoints matter more to the matcher than a merely
                # overlapping proposal. The tiny tie-breaker avoids needlessly
                # broad proposal sets when two configurations recover the same seams.
                score = (
                    0.35 * recall_50
                    + 0.65 * recall_80
                    - 1e-4 * mean_candidates
                )
                candidates.append(
                    {
                        "score": score,
                        "recall_at_5_iou_50": recall_50,
                        "recall_at_5_iou_80": recall_80,
                        "truth_count": truth_count,
                        "mean_candidate_count": mean_candidates,
                        "config": proposal_config.public_dict(),
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    return FragmentProposalConfig(**best["config"]), {
        "selected": best,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:5],
    }


def calibrate_checkpoint(
    *,
    config: CalibrationConfig,
    index: BoundaryTrainingIndex,
    sampler: BoundaryWindowSampler,
    predictor: FullContourPredictor,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("checkpoint calibration requires PyTorch")
    config.validate()
    if config.output_checkpoint_path.exists() or config.report_path.exists():
        raise FileExistsError(
            "calibration output already exists; choose new output paths"
        )
    entry_indices = list(index.by_split.get("calibration", ()))
    if not entry_indices:
        raise ValueError("dataset has no calibration observations")
    rng = np.random.default_rng(config.seed)
    rng.shuffle(entry_indices)
    if config.maximum_observations is not None:
        entry_indices = entry_indices[: config.maximum_observations]

    contours: list[dict[str, np.ndarray]] = []
    started = time.perf_counter()
    for position, entry_index in enumerate(entry_indices, start=1):
        entry = index.entries[entry_index]
        prediction = predictor.predict(entry)
        truth = sampler.full_observation(entry)
        contours.append(
            {
                "role_logits": prediction["role_logits"],
                "transition_logits": prediction["transition_logits"],
                "targets": truth["targets"],
                "transition_targets": numpy_transition_targets(truth["targets"]),
            }
        )
        print(
            f"[{position}/{len(entry_indices)}] calibration {entry.puzzle_id}/"
            f"{entry.observation_id}",
            flush=True,
        )

    role_logits = np.concatenate([item["role_logits"] for item in contours])
    role_targets = np.concatenate([item["targets"] for item in contours])
    transition_logits = np.concatenate(
        [item["transition_logits"] for item in contours]
    )
    transition_targets = np.concatenate(
        [item["transition_targets"] for item in contours]
    )
    (
        role_logits_fit,
        role_targets_fit,
        transition_logits_fit,
        transition_targets_fit,
    ) = _deterministic_subsample(
        [role_logits, role_targets, transition_logits, transition_targets],
        config.maximum_temperature_points,
    )
    role_temperature = fit_role_temperature(role_logits_fit, role_targets_fit)
    transition_temperature = fit_transition_temperature(
        transition_logits_fit, transition_targets_fit
    )
    fragment_config, fragment_search = select_fragment_config(
        contours,
        role_temperature=role_temperature,
        transition_temperature=transition_temperature,
        base=predictor.fragment_config,
    )

    before = StreamingBoundaryMetrics()
    after = StreamingBoundaryMetrics()
    before.update(
        role_probabilities(role_logits, 1.0),
        role_targets,
        transition_probabilities(transition_logits, 1.0),
        transition_targets,
    )
    after.update(
        role_probabilities(role_logits, role_temperature),
        role_targets,
        transition_probabilities(transition_logits, transition_temperature),
        transition_targets,
    )
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "calibration",
        "observation_count": len(contours),
        "point_count": int(len(role_logits)),
        "temperature_fit_point_count": int(len(role_logits_fit)),
        "input_checkpoint_path": str(config.checkpoint_path.resolve()),
        "input_checkpoint_sha256": checkpoint_sha256(config.checkpoint_path),
        "output_checkpoint_path": str(config.output_checkpoint_path.resolve()),
        "temperatures": {
            "role_temperature": role_temperature,
            "transition_temperature": transition_temperature,
        },
        "fragment_config": fragment_config.public_dict(),
        "fragment_search": fragment_search,
        "metrics_before": before.compute(),
        "metrics_after": after.compute(),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "configuration": {
            **asdict(config),
            "checkpoint_path": str(config.checkpoint_path),
            "output_checkpoint_path": str(config.output_checkpoint_path),
            "report_path": str(config.report_path),
        },
    }
    output_checkpoint = dict(checkpoint)
    output_checkpoint["calibration"] = report["temperatures"]
    output_checkpoint["fragment_config"] = report["fragment_config"]
    output_checkpoint["calibration_report"] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_utc": report["created_utc"],
        "split": "calibration",
        "observation_count": len(contours),
        "input_checkpoint_sha256": report["input_checkpoint_sha256"],
    }
    _save_torch_atomic(config.output_checkpoint_path, output_checkpoint)
    report["output_checkpoint_sha256"] = checkpoint_sha256(
        config.output_checkpoint_path
    )
    _write_json_atomic(config.report_path, report)
    return report

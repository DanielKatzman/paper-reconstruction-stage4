from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .fragments import (
    best_fragment_matches,
    fragment_recall,
    true_seam_fragments,
)
from .inference import FullContourPredictor, checkpoint_sha256
from .model_metrics import StreamingBoundaryMetrics, selection_score
from .training_data import BoundaryTrainingIndex, BoundaryWindowSampler


FRAGMENT_TOP_K_VALUES = (1, 3, 5, 10)
FRAGMENT_IOU_THRESHOLDS = (0.50, 0.80)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def numpy_transition_targets(role_targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(role_targets, dtype=np.float64)
    uncertainty = targets[:, 2].copy()
    difference = 0.5 * np.abs(targets[1:] - targets[:-1]).sum(axis=1)
    uncertainty[1:] = np.maximum(uncertainty[1:], difference)
    uncertainty[:-1] = np.maximum(uncertainty[:-1], difference)
    return np.clip(uncertainty, 0.0, 1.0)


def _new_fragment_totals() -> dict[str, Any]:
    return {
        "truth_count": 0,
        "match_counts": {
            (top_k, iou_threshold): 0
            for top_k in FRAGMENT_TOP_K_VALUES
            for iou_threshold in FRAGMENT_IOU_THRESHOLDS
        },
        "best_ious": {top_k: [] for top_k in FRAGMENT_TOP_K_VALUES},
        "candidate_counts": [],
    }


def _fragment_values(totals: dict[str, Any]) -> dict[str, Any]:
    truth_count = int(totals["truth_count"])
    denominator = max(truth_count, 1)
    result: dict[str, Any] = {
        "truth_count": truth_count,
    }
    for top_k in FRAGMENT_TOP_K_VALUES:
        best_ious = np.asarray(totals["best_ious"][top_k], dtype=np.float64)
        result[f"median_best_iou_at_{top_k}"] = (
            float(np.median(best_ious)) if len(best_ious) else 0.0
        )
        for iou_threshold in FRAGMENT_IOU_THRESHOLDS:
            suffix = int(round(iou_threshold * 100))
            matches = int(totals["match_counts"][(top_k, iou_threshold)])
            result[f"recall_at_{top_k}_iou_{suffix}_matches"] = matches
            result[f"recall_at_{top_k}_iou_{suffix}"] = matches / denominator
    # Backward-compatible alias: v2 reports used the top-five median.
    result["median_best_iou"] = result["median_best_iou_at_5"]
    return result


def _candidate_statistics(counts: list[int]) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    if not len(values):
        return {
            "observation_count": 0,
            "minimum": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "maximum": 0,
            "zero_candidate_observations": 0,
            "observations_over_5": 0,
            "observations_over_10": 0,
        }
    return {
        "observation_count": int(len(values)),
        "minimum": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(values.max()),
        "zero_candidate_observations": int(np.sum(values == 0)),
        "observations_over_5": int(np.sum(values > 5)),
        "observations_over_10": int(np.sum(values > 10)),
    }


def _update_fragment_totals(
    totals: dict[str, Any],
    *,
    match_counts: dict[tuple[int, float], int],
    truth_count: int,
    best_ious: dict[int, list[float]],
    candidate_count: int,
) -> None:
    totals["truth_count"] += truth_count
    for key, value in match_counts.items():
        totals["match_counts"][key] += value
    for top_k, values in best_ious.items():
        totals["best_ious"][top_k].extend(values)
    totals["candidate_counts"].append(candidate_count)


def _draw_fragment(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    indices: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if not len(indices):
        return
    coordinates = [tuple(map(float, points[index])) for index in indices]
    if len(coordinates) > 1:
        draw.line(coordinates, fill=color, width=width, joint="curve")
    radius = max(2, width)
    for x, y in (coordinates[0], coordinates[-1]):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=max(1, width // 2),
        )


def _render_failure_overlay(
    *,
    image_path: Path,
    points: np.ndarray,
    truth: np.ndarray,
    candidate: dict[str, Any] | None,
    top_10_candidate: dict[str, Any] | None,
    output_path: Path,
) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    contour = [tuple(map(float, point)) for point in points]
    if len(contour) > 1:
        draw.line([*contour, contour[0]], fill=(90, 90, 90), width=1)
    if top_10_candidate is not None:
        top_10_predicted = (
            int(top_10_candidate["start_index"])
            + np.arange(int(top_10_candidate["length_points"]), dtype=np.int64)
        ) % len(points)
        _draw_fragment(
            draw,
            points,
            top_10_predicted,
            color=(0, 180, 255),
            width=3,
        )
    if candidate is not None:
        # Recover the candidate's circular ordering rather than numeric sort.
        predicted = (
            int(candidate["start_index"])
            + np.arange(int(candidate["length_points"]), dtype=np.int64)
        ) % len(points)
        _draw_fragment(
            draw,
            points,
            predicted,
            color=(255, 0, 220),
            width=4,
        )
    _draw_fragment(draw, points, truth, color=(0, 220, 80), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _circular_distance(first: int, second: int, length: int) -> int:
    difference = abs(int(first) - int(second))
    return min(difference, length - difference)


def evaluate_full_contours(
    *,
    checkpoint_path: Path,
    split: str,
    index: BoundaryTrainingIndex,
    sampler: BoundaryWindowSampler,
    predictor: FullContourPredictor,
    output_path: Path,
    maximum_observations: int | None = None,
    seed: int = 20261006,
    allow_holdout: bool = False,
    failure_diagnostics_dir: Path | None = None,
    maximum_failure_diagnostics: int = 200,
) -> dict[str, Any]:
    if split == "holdout" and not allow_holdout:
        raise PermissionError(
            "holdout evaluation is protected; pass --allow-holdout only for the "
            "final locked report"
        )
    candidate_indices = list(index.by_split.get(split, ()))
    if not candidate_indices:
        raise ValueError(f"split {split!r} contains no observations")
    if maximum_observations is not None and maximum_observations > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(candidate_indices)
        candidate_indices = candidate_indices[:maximum_observations]

    metrics = StreamingBoundaryMetrics()
    difficulty_metrics: dict[str, StreamingBoundaryMetrics] = defaultdict(
        StreamingBoundaryMetrics
    )
    if maximum_failure_diagnostics < 0:
        raise ValueError("maximum_failure_diagnostics must be non-negative")
    fragment_totals = _new_fragment_totals()
    difficulty_fragment_totals: dict[str, dict[str, Any]] = defaultdict(
        _new_fragment_totals
    )
    failures: list[dict[str, Any]] = []
    rendered_failure_count = 0
    started = time.perf_counter()
    for position, entry_index in enumerate(candidate_indices, start=1):
        entry = index.entries[entry_index]
        prediction = predictor.predict(entry)
        truth = sampler.full_observation(entry)
        transition_truth = numpy_transition_targets(truth["targets"])
        metrics.update(
            prediction["role_probabilities"],
            truth["targets"],
            prediction["transition_probabilities"],
            transition_truth,
        )
        difficulty_metrics[entry.difficulty_tier].update(
            prediction["role_probabilities"],
            truth["targets"],
            prediction["transition_probabilities"],
            transition_truth,
        )
        true_fragments = true_seam_fragments(
            truth["targets"],
            minimum_points=predictor.fragment_config.minimum_fragment_points,
        )
        match_counts: dict[tuple[int, float], int] = {}
        best_ious_by_k: dict[int, list[float]] = {}
        truth_count = len(true_fragments)
        for top_k in FRAGMENT_TOP_K_VALUES:
            for iou_threshold in FRAGMENT_IOU_THRESHOLDS:
                matched, count, best_ious = fragment_recall(
                    prediction["candidate_fragments"],
                    true_fragments,
                    entry.point_count,
                    top_k=top_k,
                    iou_threshold=iou_threshold,
                )
                if count != truth_count:
                    raise RuntimeError("fragment metric truth counts disagree")
                match_counts[(top_k, iou_threshold)] = matched
                if iou_threshold == FRAGMENT_IOU_THRESHOLDS[0]:
                    best_ious_by_k[top_k] = best_ious
        update_values = {
            "match_counts": match_counts,
            "truth_count": truth_count,
            "best_ious": best_ious_by_k,
            "candidate_count": len(prediction["candidate_fragments"]),
        }
        _update_fragment_totals(fragment_totals, **update_values)
        _update_fragment_totals(
            difficulty_fragment_totals[entry.difficulty_tier], **update_values
        )
        matches = best_fragment_matches(
            prediction["candidate_fragments"],
            true_fragments,
            entry.point_count,
            top_k=5,
        )
        matches_at_10 = best_fragment_matches(
            prediction["candidate_fragments"],
            true_fragments,
            entry.point_count,
            top_k=10,
        )
        for match, match_at_10 in zip(matches, matches_at_10, strict=True):
            if match["best_iou"] >= 0.80:
                continue
            true_fragment = true_fragments[match["truth_index"]]
            predicted_roles = np.argmax(
                prediction["role_probabilities"][true_fragment], axis=1
            )
            role_counts = {
                name: int(np.sum(predicted_roles == role_index))
                for role_index, name in enumerate(
                    ("seam", "page_border", "uncertain")
                )
            }
            failure: dict[str, Any] = {
                "puzzle_id": entry.puzzle_id,
                "observation_id": entry.observation_id,
                "piece_id": entry.piece_id,
                "view_index": entry.view_index,
                "difficulty_tier": entry.difficulty_tier,
                "puzzle_piece_count": entry.puzzle_piece_count,
                "truth_index": int(match["truth_index"]),
                "truth_start_index": int(true_fragment[0]),
                "truth_end_index": int(true_fragment[-1]),
                "truth_length_points": int(len(true_fragment)),
                "best_iou": float(match["best_iou"]),
                "failure_class": "missed_iou_50"
                if match["best_iou"] < 0.50
                else "missed_iou_80",
                "candidate_rank": match["candidate_rank"],
                "candidate": match["candidate"],
                "candidate_count": len(prediction["candidate_fragments"]),
                "top_10_best_iou": float(match_at_10["best_iou"]),
                "top_10_candidate_rank": match_at_10["candidate_rank"],
                "top_10_candidate": match_at_10["candidate"],
                "recovered_at_10_iou_50": bool(
                    match_at_10["best_iou"] >= 0.50
                ),
                "recovered_at_10_iou_80": bool(
                    match_at_10["best_iou"] >= 0.80
                ),
                "predicted_roles_on_truth": role_counts,
                "mean_seam_probability_on_truth": float(
                    prediction["role_probabilities"][true_fragment, 0].mean()
                ),
                "truth_endpoint_transition_probabilities": [
                    float(prediction["transition_probabilities"][true_fragment[0]]),
                    float(prediction["transition_probabilities"][true_fragment[-1]]),
                ],
            }
            if match["candidate"] is not None:
                candidate = match["candidate"]
                failure["endpoint_errors_points"] = {
                    "start": _circular_distance(
                        true_fragment[0],
                        int(candidate["start_index"]),
                        entry.point_count,
                    ),
                    "end": _circular_distance(
                        true_fragment[-1],
                        int(candidate["end_index"]),
                        entry.point_count,
                    ),
                }
            if match_at_10["candidate"] is not None:
                candidate_at_10 = match_at_10["candidate"]
                failure["top_10_endpoint_errors_points"] = {
                    "start": _circular_distance(
                        true_fragment[0],
                        int(candidate_at_10["start_index"]),
                        entry.point_count,
                    ),
                    "end": _circular_distance(
                        true_fragment[-1],
                        int(candidate_at_10["end_index"]),
                        entry.point_count,
                    ),
                }
            if (
                failure_diagnostics_dir is not None
                and rendered_failure_count < maximum_failure_diagnostics
            ):
                image_name = (
                    f"{position:05d}_{_safe_name(entry.observation_id)}_"
                    f"truth_{match['truth_index']:02d}.png"
                )
                _render_failure_overlay(
                    image_path=index.observation_dataset_dir / entry.image_path,
                    points=truth["points_xy"],
                    truth=true_fragment,
                    candidate=match["candidate"],
                    top_10_candidate=(
                        match_at_10["candidate"]
                        if match_at_10["candidate_rank"] is not None
                        and match_at_10["candidate_rank"] > 5
                        else None
                    ),
                    output_path=failure_diagnostics_dir / image_name,
                )
                failure["overlay_path"] = image_name
                rendered_failure_count += 1
            failures.append(failure)
        elapsed = time.perf_counter() - started
        rate = position / max(elapsed, 1e-9)
        remaining = (len(candidate_indices) - position) / max(rate, 1e-9)
        print(
            f"[{position}/{len(candidate_indices)}] {entry.puzzle_id}/"
            f"{entry.observation_id}, ETA {remaining:.1f}s",
            flush=True,
        )

    metric_values = metrics.compute()
    fragment_values = _fragment_values(fragment_totals)
    diagnostics_summary = {
        "iou_threshold": 0.80,
        "failure_count": len(failures),
        "recovered_at_10_iou_50": sum(
            int(failure["recovered_at_10_iou_50"]) for failure in failures
        ),
        "recovered_at_10_iou_80": sum(
            int(failure["recovered_at_10_iou_80"]) for failure in failures
        ),
        "remaining_at_10_iou_50": sum(
            int(not failure["recovered_at_10_iou_50"]) for failure in failures
        ),
        "remaining_at_10_iou_80": sum(
            int(not failure["recovered_at_10_iou_80"]) for failure in failures
        ),
        "rendered_count": rendered_failure_count,
        "maximum_rendered": maximum_failure_diagnostics,
    }
    if failure_diagnostics_dir is not None:
        diagnostics_path = failure_diagnostics_dir / "failure_diagnostics.json"
        _write_json_atomic(
            diagnostics_path,
            {
                "schema_version": "stage2.boundary-role-failures.v2",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "split": split,
                "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
                "summary": diagnostics_summary,
                "failures": failures,
            },
        )
        diagnostics_summary["manifest_path"] = str(diagnostics_path.resolve())
    report = {
        "schema_version": "stage2.boundary-role-evaluation.v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "boundary_dataset_fingerprint": index.boundary_dataset[
            "boundary_dataset_fingerprint"
        ],
        "observation_dataset_fingerprint": index.observation_dataset[
            "observation_dataset_fingerprint"
        ],
        "split": split,
        "observation_count": len(candidate_indices),
        "metrics": metric_values,
        "selection_score": selection_score(metric_values),
        "fragments": fragment_values,
        "candidate_statistics": _candidate_statistics(
            fragment_totals["candidate_counts"]
        ),
        "fragments_by_difficulty": {
            name: _fragment_values(values)
            for name, values in sorted(difficulty_fragment_totals.items())
        },
        "candidate_statistics_by_difficulty": {
            name: _candidate_statistics(values["candidate_counts"])
            for name, values in sorted(difficulty_fragment_totals.items())
        },
        "failure_diagnostics": diagnostics_summary,
        "by_difficulty": {
            name: values.compute()
            for name, values in sorted(difficulty_metrics.items())
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    _write_json_atomic(output_path, report)
    return report

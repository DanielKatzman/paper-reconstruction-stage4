from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(slots=True)
class FragmentProposalConfig:
    smoothing_sigma_samples: float = 1.5
    high_threshold: float = 0.65
    low_threshold: float = 0.40
    relaxed_high_threshold: float = 0.55
    relaxed_low_threshold: float = 0.30
    strict_high_threshold: float = 0.75
    strict_low_threshold: float = 0.50
    transition_split_threshold: float = 0.65
    relaxed_transition_split_threshold: float = 0.40
    maximum_transition_splits: int = 4
    minimum_fragment_points: int = 8
    merge_gap_points: int = 3
    endpoint_search_radius: int = 6
    maximum_candidates: int = 32

    def validate(self) -> None:
        if self.smoothing_sigma_samples < 0.0:
            raise ValueError("smoothing sigma must be non-negative")
        for high, low in (
            (self.high_threshold, self.low_threshold),
            (self.relaxed_high_threshold, self.relaxed_low_threshold),
            (self.strict_high_threshold, self.strict_low_threshold),
        ):
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("fragment thresholds must satisfy 0 <= low <= high <= 1")
        if self.minimum_fragment_points < 1 or self.maximum_candidates < 1:
            raise ValueError("fragment sizes and candidate counts must be positive")
        if not (
            0.0
            <= self.relaxed_transition_split_threshold
            <= self.transition_split_threshold
            <= 1.0
        ):
            raise ValueError(
                "transition split thresholds must satisfy "
                "0 <= relaxed <= standard <= 1"
            )
        if self.maximum_transition_splits < 1:
            raise ValueError("maximum transition splits must be positive")
        if self.merge_gap_points < 0 or self.endpoint_search_radius < 0:
            raise ValueError("gap and endpoint radii must be non-negative")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def circular_runs(mask: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    count = len(values)
    if count == 0 or not values.any():
        return []
    if values.all():
        return [np.arange(count, dtype=np.int64)]
    start = int(np.flatnonzero(~values)[0])
    rotated = np.roll(values, -start)
    padded = np.concatenate(([False], rotated, [False]))
    changes = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(changes == 1)
    run_ends = np.flatnonzero(changes == -1)
    return [
        (np.arange(begin, end, dtype=np.int64) + start) % count
        for begin, end in zip(run_starts, run_ends, strict=True)
    ]


def _merge_small_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    if maximum_gap <= 0:
        return np.asarray(mask, dtype=bool)
    result = np.asarray(mask, dtype=bool).copy()
    for gap in circular_runs(~result):
        if len(gap) <= maximum_gap and result.any():
            result[gap] = True
    return result


def _hysteresis_runs(
    scores: np.ndarray,
    *,
    high: float,
    low: float,
    minimum_points: int,
    merge_gap_points: int,
) -> list[np.ndarray]:
    low_mask = _merge_small_gaps(scores >= low, merge_gap_points)
    return [
        run
        for run in circular_runs(low_mask)
        if len(run) >= minimum_points and np.any(scores[run] >= high)
    ]


def _endpoint_adjust(
    run: np.ndarray,
    transition: np.ndarray,
    radius: int,
    contour_length: int,
    minimum_points: int,
) -> np.ndarray:
    if radius <= 0 or len(run) == contour_length:
        return run
    start_candidates = (run[0] + np.arange(-radius, radius + 1)) % contour_length
    end_candidates = (run[-1] + np.arange(-radius, radius + 1)) % contour_length
    start = int(start_candidates[np.argmax(transition[start_candidates])])
    end = int(end_candidates[np.argmax(transition[end_candidates])])
    length = (end - start) % contour_length + 1
    # Each endpoint may move by at most ``radius``, so a valid adjustment can
    # change the run length by at most twice that amount. Without this guard,
    # the two endpoint-search windows of a nearly complete circular run may
    # overlap and select the same transition peak, collapsing hundreds of
    # valid seam points into a one-point candidate.
    maximum_length_change = 2 * radius
    if (
        length < minimum_points
        or abs(length - len(run)) > maximum_length_change
    ):
        return run
    return (start + np.arange(length, dtype=np.int64)) % contour_length


def _split_at_transitions(
    run: np.ndarray,
    transition: np.ndarray,
    threshold: float,
    minimum_points: int,
    maximum_splits: int,
) -> list[np.ndarray]:
    """Split a long hysteresis run at strong internal endpoint predictions."""
    if len(run) < 2 * minimum_points + 1:
        return [run]
    values = transition[run]
    local_maximum = values >= np.maximum(
        np.roll(values, 1), np.roll(values, -1)
    )
    candidates = np.flatnonzero(
        (values >= threshold)
        & local_maximum
        & (np.arange(len(run)) >= minimum_points)
        & (np.arange(len(run)) <= len(run) - minimum_points)
    )
    if not len(candidates):
        return [run]
    # Non-maximum suppression keeps nearby transition peaks from creating tiny
    # duplicate fragments. Peaks are considered strongest-first.
    selected: list[int] = []
    for candidate in candidates[np.argsort(values[candidates])[::-1]]:
        position = int(candidate)
        if all(abs(position - other) >= minimum_points for other in selected):
            selected.append(position)
            if len(selected) >= maximum_splits:
                break
    selected.sort()
    boundaries = [0, *selected, len(run) - 1]
    pieces = [
        run[begin : end + 1]
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        if end - begin + 1 >= minimum_points
    ]
    return pieces or [run]


def _fragment_record(
    indices: np.ndarray,
    seam: np.ndarray,
    uncertainty: np.ndarray,
    transition: np.ndarray,
    source: str,
    hypothesis: str,
    split_threshold: float | None = None,
) -> dict[str, Any]:
    contour_length = len(seam)
    mean_seam = float(seam[indices].mean())
    mean_uncertainty = float(uncertainty[indices].mean())
    endpoint_transition = float(
        0.5 * (transition[int(indices[0])] + transition[int(indices[-1])])
    )
    length_fraction = len(indices) / max(contour_length, 1)
    score = (
        0.65 * mean_seam
        + 0.20 * endpoint_transition
        + 0.10 * (1.0 - mean_uncertainty)
        + 0.05 * min(1.0, np.sqrt(length_fraction * 4.0))
    )
    return {
        "start_index": int(indices[0]),
        "end_index": int(indices[-1]),
        "wraps": bool(indices[-1] < indices[0]),
        "length_points": int(len(indices)),
        "length_fraction": float(length_fraction),
        "mean_seam_probability": mean_seam,
        "mean_uncertainty_probability": mean_uncertainty,
        "endpoint_transition_probability": endpoint_transition,
        "score": float(score),
        "source": source,
        "hypothesis": hypothesis,
        "split_threshold": split_threshold,
    }


def fragment_mask(fragment: dict[str, Any], contour_length: int) -> np.ndarray:
    mask = np.zeros(contour_length, dtype=bool)
    start = int(fragment["start_index"])
    length = int(fragment["length_points"])
    mask[(start + np.arange(length, dtype=np.int64)) % contour_length] = True
    return mask


def propose_fragments(
    role_probabilities: np.ndarray,
    transition_probabilities: np.ndarray,
    config: FragmentProposalConfig | None = None,
) -> list[dict[str, Any]]:
    settings = config or FragmentProposalConfig()
    settings.validate()
    probabilities = np.asarray(role_probabilities, dtype=np.float64)
    transition = np.asarray(transition_probabilities, dtype=np.float64).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("role_probabilities must have shape [contour_points, 3]")
    if len(transition) != len(probabilities):
        raise ValueError("transition probability length does not match contour")
    if settings.smoothing_sigma_samples > 0.0:
        seam = ndimage.gaussian_filter1d(
            probabilities[:, 0], settings.smoothing_sigma_samples, mode="wrap"
        )
    else:
        seam = probabilities[:, 0].copy()
    uncertainty = probabilities[:, 2]
    threshold_sets = (
        ("default", settings.high_threshold, settings.low_threshold),
        ("relaxed", settings.relaxed_high_threshold, settings.relaxed_low_threshold),
        ("strict", settings.strict_high_threshold, settings.strict_low_threshold),
    )
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def add_record(
        indices: np.ndarray,
        *,
        source: str,
        hypothesis: str,
        split_threshold: float | None = None,
    ) -> None:
        if len(indices) < settings.minimum_fragment_points:
            return
        key = (
            int(indices[0]),
            int(indices[-1]),
            int(len(indices)),
        )
        if key in seen:
            return
        seen.add(key)
        records.append(
            _fragment_record(
                indices,
                seam,
                uncertainty,
                transition,
                source=source,
                hypothesis=hypothesis,
                split_threshold=split_threshold,
            )
        )

    for source, high, low in threshold_sets:
        runs = _hysteresis_runs(
            seam,
            high=high,
            low=low,
            minimum_points=settings.minimum_fragment_points,
            merge_gap_points=settings.merge_gap_points,
        )
        for run in runs:
            # Parent hypotheses are retained deliberately. A transition is an
            # uncertain split proposal, not permission to destroy an otherwise
            # strong seam run.
            add_record(run, source=source, hypothesis="parent_raw")
            adjusted = _endpoint_adjust(
                run,
                transition,
                settings.endpoint_search_radius,
                len(seam),
                settings.minimum_fragment_points,
            )
            add_record(adjusted, source=source, hypothesis="parent_adjusted")
            for split_name, split_threshold in (
                ("split_standard", settings.transition_split_threshold),
                ("split_relaxed", settings.relaxed_transition_split_threshold),
            ):
                split_runs = _split_at_transitions(
                    adjusted,
                    transition,
                    split_threshold,
                    settings.minimum_fragment_points,
                    settings.maximum_transition_splits,
                )
                if len(split_runs) <= 1:
                    continue
                for split_run in split_runs:
                    add_record(
                        split_run,
                        source=source,
                        hypothesis=split_name,
                        split_threshold=split_threshold,
                    )
    records.sort(key=lambda record: record["score"], reverse=True)
    for rank, record in enumerate(records[: settings.maximum_candidates], start=1):
        record["rank"] = rank
    return records[: settings.maximum_candidates]


def true_seam_fragments(role_targets: np.ndarray, minimum_points: int = 1) -> list[np.ndarray]:
    targets = np.asarray(role_targets)
    hard = np.argmax(targets, axis=1) == 0
    return [run for run in circular_runs(hard) if len(run) >= minimum_points]


def fragment_recall(
    candidates: list[dict[str, Any]],
    truths: list[np.ndarray],
    contour_length: int,
    *,
    top_k: int,
    iou_threshold: float,
) -> tuple[int, int, list[float]]:
    candidate_masks = [
        fragment_mask(candidate, contour_length) for candidate in candidates[:top_k]
    ]
    matched = 0
    best_ious: list[float] = []
    for truth in truths:
        truth_mask = np.zeros(contour_length, dtype=bool)
        truth_mask[truth] = True
        best = 0.0
        for candidate_mask in candidate_masks:
            union = int(np.sum(truth_mask | candidate_mask))
            intersection = int(np.sum(truth_mask & candidate_mask))
            best = max(best, intersection / max(union, 1))
        best_ious.append(best)
        matched += int(best >= iou_threshold)
    return matched, len(truths), best_ious


def best_fragment_matches(
    candidates: list[dict[str, Any]],
    truths: list[np.ndarray],
    contour_length: int,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the best top-k candidate and IoU for every true seam run."""

    selected = candidates[:top_k]
    candidate_masks = [
        fragment_mask(candidate, contour_length) for candidate in selected
    ]
    matches: list[dict[str, Any]] = []
    for truth_index, truth in enumerate(truths):
        truth_mask = np.zeros(contour_length, dtype=bool)
        truth_mask[truth] = True
        best_iou = 0.0
        best_position: int | None = None
        for candidate_position, candidate_mask in enumerate(candidate_masks):
            union = int(np.sum(truth_mask | candidate_mask))
            intersection = int(np.sum(truth_mask & candidate_mask))
            iou = intersection / max(union, 1)
            if best_position is None or iou > best_iou:
                best_iou = iou
                best_position = candidate_position
        matches.append(
            {
                "truth_index": truth_index,
                "best_iou": float(best_iou),
                "candidate_rank": best_position + 1
                if best_position is not None
                else None,
                "candidate": selected[best_position]
                if best_position is not None
                else None,
            }
        )
    return matches

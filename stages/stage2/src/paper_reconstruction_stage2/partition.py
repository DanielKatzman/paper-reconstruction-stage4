from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .config import GeneratorConfig
from .profiles import make_profile


FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


@dataclass(slots=True)
class CutEvent:
    cut_index: int
    parent_label: int
    retained_child_label: int
    new_child_label: int
    family: str
    angle_rad: float
    offset_px: float
    profile_parameters: dict[str, float | int]
    polyline_xy: list[list[float]]
    child_areas_px: list[int]
    attempts_used: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PartitionResult:
    labels: np.ndarray
    target_piece_count: int
    successful_cut_count: int
    cut_events: list[CutEvent]
    rejected_attempts: int
    difficulty_tier: str


def _is_one_component(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    _, count = ndimage.label(mask, structure=FOUR_CONNECTED)
    return count == 1


def _bbox_dimensions(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _sample_family(
    config: GeneratorConfig,
    rng: np.random.Generator,
    difficulty_tier: str,
) -> str:
    names, weights = config.normalized_cut_weights(difficulty_tier)
    return str(rng.choice(names, p=weights))


def _try_split(
    labels: np.ndarray,
    parent_label: int,
    new_label: int,
    family: str,
    config: GeneratorConfig,
    rng: np.random.Generator,
    difficulty_tier: str,
) -> tuple[np.ndarray, CutEvent] | None:
    parent = labels == parent_label
    ys, xs = np.nonzero(parent)
    if len(xs) < config.min_piece_area_px * 2:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())
    angle = float(rng.uniform(0.0, np.pi))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    local_u = (xs - cx) * cosine + (ys - cy) * sine
    local_v = -(xs - cx) * sine + (ys - cy) * cosine
    u_min = float(local_u.min())
    u_max = float(local_u.max())
    v_min = float(local_v.min())
    v_max = float(local_v.max())
    if u_max - u_min < config.min_piece_dimension_px * 2:
        return None
    if v_max - v_min < config.min_piece_dimension_px * 2:
        return None

    t = (local_u - u_min) / max(u_max - u_min, 1e-8)
    profile = make_profile(
        family, t, v_max - v_min, rng, difficulty_tier=difficulty_tier
    )
    offset_limit = 0.14 * (v_max - v_min)
    offset = float(rng.uniform(-offset_limit, offset_limit))
    threshold = profile.values + offset
    choose_a = local_v <= threshold

    child_a = np.zeros_like(parent)
    child_b = np.zeros_like(parent)
    child_a[ys[choose_a], xs[choose_a]] = True
    child_b[ys[~choose_a], xs[~choose_a]] = True
    area_a = int(child_a.sum())
    area_b = int(child_b.sum())
    if min(area_a, area_b) < config.min_piece_area_px:
        return None
    if not _is_one_component(child_a) or not _is_one_component(child_b):
        return None
    width_a, height_a = _bbox_dimensions(child_a)
    width_b, height_b = _bbox_dimensions(child_b)
    if min(width_a, height_a, width_b, height_b) < config.min_piece_dimension_px:
        return None

    result = labels.copy()
    result[child_b] = new_label

    sample_count = max(96, min(1024, round((u_max - u_min) * 1.2)))
    sample_u = np.linspace(u_min, u_max, sample_count)
    sample_t = (sample_u - u_min) / max(u_max - u_min, 1e-8)
    # Recreate the accepted profile by interpolation rather than consuming new RNG state.
    profile_order = np.argsort(t)
    sample_v = np.interp(sample_t, t[profile_order], threshold[profile_order])
    sample_x = cx + sample_u * cosine - sample_v * sine
    sample_y = cy + sample_u * sine + sample_v * cosine
    polyline = np.column_stack((sample_x, sample_y)).round(4).tolist()

    event = CutEvent(
        cut_index=0,
        parent_label=parent_label,
        retained_child_label=parent_label,
        new_child_label=new_label,
        family=family,
        angle_rad=round(angle, 8),
        offset_px=round(offset, 5),
        profile_parameters=profile.parameters,
        polyline_xy=polyline,
        child_areas_px=[area_a, area_b],
        attempts_used=0,
    )
    return result, event


def generate_partition(
    config: GeneratorConfig,
    rng: np.random.Generator,
    *,
    target_piece_count: int | None = None,
    difficulty_tier: str = "moderate",
) -> PartitionResult:
    if target_piece_count is None:
        target_piece_count = int(rng.integers(config.min_pieces, config.max_pieces + 1))
    if not config.min_pieces <= target_piece_count <= config.max_pieces:
        raise ValueError("target_piece_count is outside configured bounds")
    if difficulty_tier not in config.difficulty_mix:
        raise ValueError(f"unknown difficulty tier: {difficulty_tier}")

    labels = np.ones((config.page_height, config.page_width), dtype=np.int32)
    events: list[CutEvent] = []
    rejected_attempts = 0

    for next_label in range(2, target_piece_count + 1):
        accepted = False
        attempts_for_cut = 0
        for _ in range(config.max_split_attempts):
            attempts_for_cut += 1
            unique, counts = np.unique(labels, return_counts=True)
            eligible = counts >= config.min_piece_area_px * 2
            candidates = unique[eligible]
            candidate_areas = counts[eligible].astype(np.float64)
            if not len(candidates):
                break
            weights = candidate_areas**1.35
            weights /= weights.sum()
            parent_label = int(rng.choice(candidates, p=weights))
            family = _sample_family(config, rng, difficulty_tier)
            outcome = _try_split(
                labels,
                parent_label,
                next_label,
                family,
                config,
                rng,
                difficulty_tier,
            )
            if outcome is None:
                rejected_attempts += 1
                continue
            labels, event = outcome
            event.cut_index = len(events) + 1
            event.attempts_used = attempts_for_cut
            events.append(event)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"could not create piece {next_label}/{target_piece_count} after "
                f"{config.max_split_attempts} attempts; lower min_piece_area_fraction "
                "or increase page resolution"
            )

    return PartitionResult(
        labels=labels,
        target_piece_count=target_piece_count,
        successful_cut_count=len(events),
        cut_events=events,
        rejected_attempts=rejected_attempts,
        difficulty_tier=difficulty_tier,
    )

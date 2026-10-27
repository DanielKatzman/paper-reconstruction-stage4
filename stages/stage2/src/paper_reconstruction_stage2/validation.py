from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage

from .config import GeneratorConfig
from .partition import FOUR_CONNECTED, PartitionResult


class DatasetValidationError(RuntimeError):
    pass


def validate_partition(
    partition: PartitionResult,
    pieces: list[dict],
    seams: list[dict],
    config: GeneratorConfig,
) -> dict:
    errors: list[str] = []
    labels = partition.labels
    unique = sorted(int(value) for value in np.unique(labels))
    expected = list(range(1, partition.target_piece_count + 1))
    if unique != expected:
        errors.append(f"labels are not contiguous: expected {expected}, got {unique}")
    if partition.successful_cut_count != partition.target_piece_count - 1:
        errors.append("successful_cut_count must equal target_piece_count - 1")
    if len(pieces) != partition.target_piece_count:
        errors.append("piece record count does not match target_piece_count")
    if labels.shape != (config.page_height, config.page_width):
        errors.append("label map has unexpected dimensions")
    if np.any(labels <= 0):
        errors.append("label map contains uncovered pixels")

    piece_ids = {piece["piece_id"] for piece in pieces}
    for piece in pieces:
        mask = labels == piece["label"]
        _, component_count = ndimage.label(mask, structure=FOUR_CONNECTED)
        if component_count != 1:
            errors.append(f"{piece['piece_id']} has {component_count} components")
        if piece["area_px"] < config.min_piece_area_px:
            errors.append(f"{piece['piece_id']} is below minimum area")

    adjacency: dict[str, set[str]] = defaultdict(set)
    seam_ids: set[str] = set()
    for seam in seams:
        seam_id = seam["seam_id"]
        if seam_id in seam_ids:
            errors.append(f"duplicate seam id {seam_id}")
        seam_ids.add(seam_id)
        a, b = seam["owner_a"], seam["owner_b"]
        if a == b or a not in piece_ids or b not in piece_ids:
            errors.append(f"invalid seam owners for {seam_id}")
        if len(seam["canonical_polyline_xy"]) < 2:
            errors.append(f"{seam_id} has an empty polyline")
        adjacency[a].add(b)
        adjacency[b].add(a)

    for piece in pieces:
        identifier = piece["piece_id"]
        if sorted(adjacency[identifier]) != piece["neighbors"]:
            errors.append(f"neighbor list mismatch for {identifier}")

    if errors:
        raise DatasetValidationError("; ".join(errors))
    return {
        "valid": True,
        "piece_count": len(pieces),
        "seam_count": len(seams),
        "covered_pixels": int(labels.size),
        "minimum_piece_area_px": min(piece["area_px"] for piece in pieces),
        "maximum_piece_area_px": max(piece["area_px"] for piece in pieces),
    }


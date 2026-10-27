from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .contracts import GEOMETRY, ROLE_NAMES, circular_indices, require, map_points

_BOUNDARY_CACHE: OrderedDict = OrderedDict()


def boundary_geometry(path: str) -> dict[str, np.ndarray]:
    """One puzzle in RAM, selectively reading observable NPZ members only."""
    p = Path(path)
    key = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
    if key not in _BOUNDARY_CACHE:
        _BOUNDARY_CACHE.clear()
        with np.load(p, allow_pickle=False) as archive:
            _BOUNDARY_CACHE[key] = {k: archive[k].copy() for k in (*GEOMETRY, "points_xy", "observation_offsets")}
    return _BOUNDARY_CACHE[key]


@dataclass
class ObservableContour:
    """Inference inputs only. Canonical transforms/targets never enter this object."""
    arrays: dict[str, np.ndarray]
    image_path: Path | None
    mask_path: Path | None
    candidates: list[dict[str, Any]]

    def fragment(self, candidate: dict) -> dict[str, np.ndarray]:
        indices = circular_indices(candidate, len(self.arrays["points_xy"]))
        return {name: value[indices].copy() for name, value in self.arrays.items()}


def load_contour(entry: dict, *, allow_protected: bool = False) -> ObservableContour:
    require(allow_protected or entry["split"] not in {"holdout", "test"}, "Protected data access is disabled; audit may inspect observable integrity only")
    # np.load does not decompress unrequested members. Truth arrays are not read.
    with np.load(entry["arrays_path"], allow_pickle=False) as predictions:
        point_key = "points_xy_crop" if entry["mode"] == "production" else "points_xy"
        arrays = {"points_xy": predictions[point_key].copy()}
        for name in ("role_probabilities", "transition_probabilities"):
            arrays[name] = predictions[name].copy()
        if entry["mode"] == "production":
            for name in GEOMETRY:
                arrays[name] = predictions[name].copy()
            source_points = predictions["points_xy_source"].copy()
        else:
            source_points = None
    if entry["mode"] == "research":
        start, end = entry["offset_start"], entry["offset_end"]
        boundary = boundary_geometry(entry["boundaries_path"])
        for name in GEOMETRY:
            arrays[name] = boundary[name][start:end].copy()
        require(np.allclose(arrays["points_xy"], boundary["points_xy"][start:end], atol=1e-4), "Prediction and compiled contour points disagree")
        offsets = boundary["observation_offsets"]
        index = entry["observation_index"]
        require(int(offsets[index]) == start and int(offsets[index + 1]) == end, "Compiled observation offsets disagree")
    n = entry["point_count"]
    require(arrays["points_xy"].shape == (n, 2), "Invalid contour shape")
    for name, value in arrays.items():
        expected = (n, 3) if name == "role_probabilities" else (n, 2) if name in {"points_xy", "tangent_xy", "inward_normal_xy"} else (n,)
        require(value.shape == expected and np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(), f"Invalid/nonfinite observable array: {name}")
    for name in ("role_probabilities", "transition_probabilities"):
        require(np.all((arrays[name] >= -1e-6) & (arrays[name] <= 1 + 1e-6)), f"Invalid probability range: {name}")
    require(np.allclose(arrays["role_probabilities"].sum(axis=1), 1, atol=1e-5), "Role probabilities do not sum to one")
    require(np.all(np.diff(arrays["arc_length_px"]) >= 0), "Arc lengths are not ordered")
    for name in ("tangent_xy", "inward_normal_xy"):
        require(np.allclose(np.linalg.norm(arrays[name], axis=1), 1, atol=0.02), f"Non-unit {name}")
    require(np.all(arrays["inward_thickness_px"] >= 0), "Negative inward thickness")
    xy = arrays["points_xy"]
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    require(area > 0, "Contour is not clockwise in image coordinates")
    if source_points is not None:
        mapped = map_points(xy, np.asarray(entry["crop_to_source_homography"]))
        require(source_points.shape == (n, 2) and np.allclose(mapped, source_points, atol=1e-3, rtol=0), "Crop/source coordinate mapping mismatch")
        for c in entry["candidates"]:
            for end_name, i in (("start", c["start_index"]), ("end", c["end_index"])):
                require(np.allclose(c[end_name + "_xy_crop"], xy[i], atol=1e-3), "Candidate crop endpoint mismatch")
                require(np.allclose(c[end_name + "_xy_source"], source_points[i], atol=1e-3), "Candidate source endpoint mismatch")
    return ObservableContour(arrays, Path(entry["image_path"]) if entry["image_path"] else None, Path(entry["mask_path"]) if entry["mask_path"] else None, entry["candidates"])


def check_images(entry: dict, contour: ObservableContour) -> None:
    require(contour.image_path is not None and contour.mask_path is not None, "RGB/mask inputs required for appearance-ready audit")
    with Image.open(contour.image_path) as image, Image.open(contour.mask_path) as mask:
        require(image.size == mask.size, "Piece RGB/mask dimensions disagree")
        width, height = image.size
        if entry.get("size_hw"):
            require([height, width] == entry["size_hw"], "Recorded image dimensions disagree")
        pixels = np.asarray(mask.convert("L"))
        require(set(np.unique(pixels)).issubset({0, 255}) and np.any(pixels), "Mask must be nonempty binary 0/255")
        image.verify()
    xy = contour.arrays["points_xy"]
    require(np.all(xy >= -0.01) and np.all(xy[:, 0] <= width - 1 + 0.01) and np.all(xy[:, 1] <= height - 1 + 0.01), "Contour exceeds crop bounds")

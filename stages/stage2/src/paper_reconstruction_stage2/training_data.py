from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from scipy import ndimage


ROLE_NAMES = ("seam", "page_border", "uncertain")
SAMPLING_CATEGORY_NAMES = (*ROLE_NAMES, "transition")
GEOMETRY_FEATURE_NAMES = (
    "tangent_x",
    "tangent_y",
    "inward_normal_x",
    "inward_normal_y",
    "curvature_per_px",
    "corner_strength",
    "log1p_inward_thickness_px",
)
DEFAULT_ROLE_SAMPLING_WEIGHTS = {
    "seam": 0.40,
    "page_border": 0.25,
    "uncertain": 0.15,
    "transition": 0.20,
}


def normalized_sampling_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize weights, accepting the pre-v0.6.2 role map."""

    values = dict(weights or DEFAULT_ROLE_SAMPLING_WEIGHTS)
    if set(values) == set(ROLE_NAMES):
        values["transition"] = 0.0
    if set(values) != set(SAMPLING_CATEGORY_NAMES) or any(
        float(value) < 0.0 for value in values.values()
    ):
        raise ValueError(
            f"role_sampling_weights must define {SAMPLING_CATEGORY_NAMES}"
        )
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("role_sampling_weights must have a positive sum")
    return {
        name: float(values[name]) / total for name in SAMPLING_CATEGORY_NAMES
    }


try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:  # Torch is intentionally an optional training dependency.
    class _TorchIterableDataset:  # type: ignore[no-redef]
        pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class ObservationBoundaryEntry:
    puzzle_id: str
    split: str
    difficulty_tier: str
    boundary_manifest_path: str
    boundaries_path: str
    observation_index: int
    observation_id: str
    piece_id: str
    view_index: int
    image_path: str
    mask_path: str
    point_count: int
    boundary_offset_start: int
    boundary_offset_end: int
    hard_role_counts: dict[str, int]
    puzzle_piece_count: int


@dataclass(slots=True)
class BoundaryWindowConfig:
    sequence_length: int = 129
    strip_width: int = 33
    outside_extent_px: float = 8.0
    inside_extent_px: float = 24.0
    minimum_label_confidence: float = 0.25
    puzzle_cache_size: int = 2
    image_cache_size: int = 8
    global_context_cache_size: int = 64

    def validate(self) -> None:
        if self.sequence_length < 3 or self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd and at least 3")
        if self.strip_width < 3:
            raise ValueError("strip_width must be at least 3")
        if self.outside_extent_px < 0.0 or self.inside_extent_px <= 0.0:
            raise ValueError("strip extents must be non-negative outside and positive inside")
        if not 0.0 <= self.minimum_label_confidence <= 1.0:
            raise ValueError("minimum_label_confidence must be in [0, 1]")
        if (
            self.puzzle_cache_size < 1
            or self.image_cache_size < 1
            or self.global_context_cache_size < 1
        ):
            raise ValueError("cache sizes must be positive")


class BoundaryTrainingIndex:
    """Small JSON-only index over a large boundary dataset.

    It reads puzzle manifests but never loads the large NPZ arrays. Each entry
    represents one observed piece contour and carries enough metadata for
    balanced role sampling.
    """

    def __init__(self, boundary_dataset_dir: Path, observation_dataset_dir: Path):
        self.boundary_dataset_dir = boundary_dataset_dir
        self.observation_dataset_dir = observation_dataset_dir
        boundary_manifest_path = boundary_dataset_dir / "boundary_dataset_manifest.json"
        observation_manifest_path = (
            observation_dataset_dir / "observation_dataset_manifest.json"
        )
        if not boundary_manifest_path.is_file():
            raise FileNotFoundError(
                f"boundary dataset manifest not found: {boundary_manifest_path}"
            )
        if not observation_manifest_path.is_file():
            raise FileNotFoundError(
                f"observation dataset manifest not found: {observation_manifest_path}"
            )
        self.boundary_dataset = _read_json(boundary_manifest_path)
        self.observation_dataset = _read_json(observation_manifest_path)
        if (
            self.boundary_dataset.get("observation_dataset_fingerprint")
            != self.observation_dataset.get("observation_dataset_fingerprint")
        ):
            raise ValueError(
                "boundary and observation dataset fingerprints do not match"
            )

        entries: list[ObservationBoundaryEntry] = []
        for puzzle_summary in self.boundary_dataset.get("puzzles", []):
            relative_manifest = puzzle_summary["boundary_manifest_path"]
            manifest_path = boundary_dataset_dir / relative_manifest
            manifest = _read_json(manifest_path)
            puzzle_piece_count = len(
                {observation["piece_id"] for observation in manifest["observations"]}
            )
            boundaries_path = (
                manifest_path.parent / manifest["boundaries_path"]
            ).relative_to(boundary_dataset_dir).as_posix()
            for observation in manifest["observations"]:
                entries.append(
                    ObservationBoundaryEntry(
                        puzzle_id=manifest["puzzle_id"],
                        split=str(manifest.get("split")),
                        difficulty_tier=str(manifest.get("difficulty_tier")),
                        boundary_manifest_path=relative_manifest,
                        boundaries_path=boundaries_path,
                        observation_index=int(observation["observation_index"]),
                        observation_id=observation["observation_id"],
                        piece_id=observation["piece_id"],
                        view_index=int(observation["view_index"]),
                        image_path=observation["observed_piece_dataset_path"],
                        mask_path=observation["observed_mask_dataset_path"],
                        point_count=int(observation["point_count"]),
                        boundary_offset_start=int(
                            observation["boundary_offset_start"]
                        ),
                        boundary_offset_end=int(observation["boundary_offset_end"]),
                        hard_role_counts={
                            name: int(observation["hard_role_counts"].get(name, 0))
                            for name in ROLE_NAMES
                        },
                        puzzle_piece_count=puzzle_piece_count,
                    )
                )
        self.entries = tuple(entries)
        self.by_split: dict[str, tuple[int, ...]] = {}
        self.by_split_role: dict[tuple[str, str], tuple[int, ...]] = {}
        self.by_split_transition: dict[str, tuple[int, ...]] = {}
        split_indices: dict[str, list[int]] = defaultdict(list)
        role_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, entry in enumerate(self.entries):
            split_indices[entry.split].append(index)
            for role in ROLE_NAMES:
                if entry.hard_role_counts[role] > 0:
                    role_indices[(entry.split, role)].append(index)
        self.by_split = {
            split: tuple(indices) for split, indices in split_indices.items()
        }
        self.by_split_role = {
            key: tuple(indices) for key, indices in role_indices.items()
        }
        self.by_split_transition = {
            split: tuple(
                index
                for index in indices
                if self.entries[index].hard_role_counts["uncertain"] > 0
                or (
                    self.entries[index].hard_role_counts["seam"] > 0
                    and self.entries[index].hard_role_counts["page_border"] > 0
                )
            )
            for split, indices in self.by_split.items()
        }
        self._split_point_cumulative: dict[str, np.ndarray] = {}
        for split, indices in self.by_split.items():
            self._split_point_cumulative[split] = np.cumsum(
                [self.entries[index].point_count for index in indices],
                dtype=np.int64,
            )

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_split))

    def summary(self) -> dict[str, Any]:
        split_observations = {
            split: len(indices) for split, indices in sorted(self.by_split.items())
        }
        split_role_points: dict[str, dict[str, int]] = {}
        for split, indices in sorted(self.by_split.items()):
            split_role_points[split] = {
                role: sum(
                    self.entries[index].hard_role_counts[role] for index in indices
                )
                for role in ROLE_NAMES
            }
        return {
            "schema_version": "stage2.boundary-training-index.v1",
            "boundary_dataset_fingerprint": self.boundary_dataset[
                "boundary_dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": self.observation_dataset[
                "observation_dataset_fingerprint"
            ],
            "observation_count": len(self.entries),
            "splits": split_observations,
            "hard_role_point_counts": split_role_points,
        }


class BoundaryWindowSampler:
    """Role-balanced, bounded-memory sampler for contour training windows."""

    ARRAY_NAMES = (
        "points_xy",
        "tangent_xy",
        "inward_normal_xy",
        "curvature_per_px",
        "corner_strength",
        "inward_thickness_px",
        "seam_target",
        "page_border_target",
        "uncertainty_target",
        "label_confidence",
        "nearest_truth_distance_px",
    )

    def __init__(
        self,
        index: BoundaryTrainingIndex,
        config: BoundaryWindowConfig | None = None,
    ):
        self.index = index
        self.config = config or BoundaryWindowConfig()
        self.config.validate()
        self._puzzle_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._transition_cache: OrderedDict[
            tuple[str, int], np.ndarray
        ] = OrderedDict()
        self._global_context_cache: OrderedDict[
            tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()

    @staticmethod
    def _observation_cache_key(
        entry: ObservationBoundaryEntry,
    ) -> tuple[str, int]:
        # observation_id repeats in every puzzle (piece_000__view_00, etc.).
        # The puzzle array path plus its observation index is dataset-unique.
        return entry.boundaries_path, entry.observation_index

    def _load_puzzle(self, entry: ObservationBoundaryEntry) -> dict[str, np.ndarray]:
        key = entry.boundaries_path
        if key in self._puzzle_cache:
            arrays = self._puzzle_cache.pop(key)
            self._puzzle_cache[key] = arrays
            return arrays
        path = self.index.boundary_dataset_dir / entry.boundaries_path
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in self.ARRAY_NAMES}
        self._puzzle_cache[key] = arrays
        while len(self._puzzle_cache) > self.config.puzzle_cache_size:
            self._puzzle_cache.popitem(last=False)
        return arrays

    def load_entry_arrays(
        self, entry: ObservationBoundaryEntry
    ) -> dict[str, np.ndarray]:
        """Return the cached puzzle arrays that contain an observation entry."""
        return self._load_puzzle(entry)

    def _load_image(self, relative_path: str, *, mask: bool) -> np.ndarray:
        key = f"{'mask' if mask else 'image'}:{relative_path}"
        if key in self._image_cache:
            image = self._image_cache.pop(key)
            self._image_cache[key] = image
            return image
        path = self.index.observation_dataset_dir / relative_path
        with Image.open(path) as opened:
            if mask:
                image = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
            else:
                image = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        self._image_cache[key] = image
        while len(self._image_cache) > self.config.image_cache_size:
            self._image_cache.popitem(last=False)
        return image

    def _entry_for_role(
        self, rng: np.random.Generator, split: str, role: str
    ) -> ObservationBoundaryEntry:
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLE_NAMES}")
        candidates = self.index.by_split_role.get((split, role), ())
        if not candidates:
            raise ValueError(f"split {split!r} has no {role!r} boundary points")
        return self.index.entries[candidates[int(rng.integers(len(candidates)))]]

    def _transition_entry_and_center(
        self, rng: np.random.Generator, split: str
    ) -> tuple[ObservationBoundaryEntry, dict[str, np.ndarray], int]:
        candidates = self.index.by_split_transition.get(split, ())
        if not candidates:
            raise ValueError(f"split {split!r} has no transition boundary points")
        # The manifest's hard-role counts are only a cheap pre-index. Soft labels
        # can identify an uncertain region whose transition strength is below the
        # primary 0.50 cutoff, so deterministically try the remaining indexed
        # observations instead of failing the worker on that one approximation.
        start = int(rng.integers(len(candidates)))
        for offset in range(len(candidates)):
            entry = self.index.entries[candidates[(start + offset) % len(candidates)]]
            arrays = self._load_puzzle(entry)
            transition_indices = self._transition_indices(arrays, entry)
            if len(transition_indices):
                center = int(
                    transition_indices[int(rng.integers(len(transition_indices)))]
                )
                return entry, arrays, center
        raise RuntimeError(
            f"split {split!r} has no usable soft transition targets"
        )

    def _transition_indices(
        self,
        arrays: dict[str, np.ndarray],
        entry: ObservationBoundaryEntry,
    ) -> np.ndarray:
        key = self._observation_cache_key(entry)
        if key in self._transition_cache:
            values = self._transition_cache.pop(key)
            self._transition_cache[key] = values
            return values
        start, end = entry.boundary_offset_start, entry.boundary_offset_end
        targets = np.column_stack(
            (
                arrays["seam_target"][start:end],
                arrays["page_border_target"][start:end],
                arrays["uncertainty_target"][start:end],
            )
        ).astype(np.float32, copy=False)
        transition = targets[:, 2].copy()
        difference = 0.5 * np.abs(targets[1:] - targets[:-1]).sum(axis=1)
        transition[1:] = np.maximum(transition[1:], difference)
        transition[:-1] = np.maximum(transition[:-1], difference)
        confidence = arrays["label_confidence"][start:end]
        candidates = np.flatnonzero(
            (transition >= 0.50)
            & (confidence >= self.config.minimum_label_confidence)
        )
        if not len(candidates):
            candidates = np.flatnonzero(transition >= 0.50)
        if not len(candidates):
            # Some valid soft junctions peak below 0.50 (for example a target
            # distribution near 0.30/0.30/0.40). Preserve their training value by
            # sampling a small set around the strongest available signal.
            maximum = float(np.max(transition)) if len(transition) else 0.0
            if maximum > 1e-6:
                candidates = np.flatnonzero(
                    transition >= max(0.05, 0.80 * maximum)
                )
                if len(candidates) > 16:
                    strongest = np.argsort(transition[candidates])[-16:]
                    candidates = np.sort(candidates[strongest])
        values = candidates.astype(np.int64, copy=False)
        self._transition_cache[key] = values
        while len(self._transition_cache) > self.config.global_context_cache_size:
            self._transition_cache.popitem(last=False)
        return values

    def _center_index(
        self,
        arrays: dict[str, np.ndarray],
        entry: ObservationBoundaryEntry,
        role: str,
        rng: np.random.Generator,
    ) -> int:
        start, end = entry.boundary_offset_start, entry.boundary_offset_end
        role_targets = np.column_stack(
            (
                arrays["seam_target"][start:end],
                arrays["page_border_target"][start:end],
                arrays["uncertainty_target"][start:end],
            )
        )
        role_index = ROLE_NAMES.index(role)
        hard_roles = np.argmax(role_targets, axis=1)
        confidence = arrays["label_confidence"][start:end]
        candidates = np.flatnonzero(
            (hard_roles == role_index)
            & (confidence >= self.config.minimum_label_confidence)
        )
        if len(candidates) == 0:
            candidates = np.flatnonzero(hard_roles == role_index)
        if len(candidates) == 0:
            raise RuntimeError(
                f"indexed role {role!r} disappeared from {entry.observation_id}"
            )
        return int(candidates[int(rng.integers(len(candidates)))])

    def _transition_center_index(
        self,
        arrays: dict[str, np.ndarray],
        entry: ObservationBoundaryEntry,
        rng: np.random.Generator,
    ) -> int:
        candidates = self._transition_indices(arrays, entry)
        if not len(candidates):
            raise RuntimeError(
                f"indexed transition disappeared from {entry.observation_id}"
            )
        return int(candidates[int(rng.integers(len(candidates)))])

    def _sequence_indices(
        self, entry: ObservationBoundaryEntry, center_local: int
    ) -> np.ndarray:
        half = self.config.sequence_length // 2
        relative = (
            center_local + np.arange(-half, half + 1, dtype=np.int64)
        ) % entry.point_count
        return relative + entry.boundary_offset_start

    def _natural_entry_and_center(
        self, rng: np.random.Generator, split: str
    ) -> tuple[ObservationBoundaryEntry, int]:
        indices = self.index.by_split.get(split, ())
        cumulative = self.index._split_point_cumulative.get(split)
        if not indices or cumulative is None or len(cumulative) == 0:
            raise ValueError(f"split {split!r} contains no boundary points")
        selected_point = int(rng.integers(int(cumulative[-1])))
        entry_position = int(np.searchsorted(cumulative, selected_point, side="right"))
        previous = int(cumulative[entry_position - 1]) if entry_position else 0
        return self.index.entries[indices[entry_position]], selected_point - previous

    def _raster_strip(
        self,
        entry: ObservationBoundaryEntry,
        points_xy: np.ndarray,
        inward_normal_xy: np.ndarray,
    ) -> np.ndarray:
        rgb = self._load_image(entry.image_path, mask=False)
        mask = self._load_image(entry.mask_path, mask=True)
        offsets = np.linspace(
            -self.config.outside_extent_px,
            self.config.inside_extent_px,
            self.config.strip_width,
            dtype=np.float32,
        )
        coordinates = points_xy[:, None, :] + inward_normal_xy[:, None, :] * offsets[
            None, :, None
        ]
        sample_y = coordinates[..., 1]
        sample_x = coordinates[..., 0]
        channels = [
            ndimage.map_coordinates(
                rgb[..., channel],
                [sample_y, sample_x],
                order=1,
                mode="constant",
                cval=0.0,
            )
            for channel in range(3)
        ]
        channels.append(
            ndimage.map_coordinates(
                mask,
                [sample_y, sample_x],
                order=1,
                mode="constant",
                cval=0.0,
            )
        )
        return np.stack(channels, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _periodic_resample(values: np.ndarray, point_count: int) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        if len(source) == point_count:
            return source.copy()
        positions = np.linspace(0.0, len(source), point_count, endpoint=False)
        lower = np.floor(positions).astype(np.int64) % len(source)
        upper = (lower + 1) % len(source)
        fraction = (positions - np.floor(positions)).astype(np.float32)
        return (
            source[lower] * (1.0 - fraction[:, None])
            + source[upper] * fraction[:, None]
        ).astype(np.float32, copy=False)

    def _global_context(
        self,
        entry: ObservationBoundaryEntry,
        arrays: dict[str, np.ndarray],
        *,
        point_count: int,
        mask_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (*self._observation_cache_key(entry), point_count, mask_size)
        if key in self._global_context_cache:
            context = self._global_context_cache.pop(key)
            self._global_context_cache[key] = context
            return context
        start, end = entry.boundary_offset_start, entry.boundary_offset_end
        points = arrays["points_xy"][start:end].astype(np.float32, copy=False)
        center = points.mean(axis=0, keepdims=True)
        scale = max(float(np.max(np.ptp(points, axis=0))), 1.0)
        normalized_points = (points - center) / scale
        geometry = np.column_stack(
            (
                normalized_points,
                arrays["tangent_xy"][start:end],
                arrays["inward_normal_xy"][start:end],
                arrays["curvature_per_px"][start:end],
                arrays["corner_strength"][start:end],
                np.log1p(arrays["inward_thickness_px"][start:end]),
            )
        ).astype(np.float32, copy=False)
        global_geometry = self._periodic_resample(geometry, point_count)

        mask = self._load_image(entry.mask_path, mask=True)
        foreground = np.argwhere(mask >= 0.5)
        if not len(foreground):
            raise RuntimeError(f"empty observed mask for {entry.observation_id}")
        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0) + 1
        cropped = mask[y0:y1, x0:x1]
        side = max(cropped.shape)
        padding = max(2, int(round(side * 0.05)))
        canvas_side = side + 2 * padding
        canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
        top = padding + (side - cropped.shape[0]) // 2
        left = padding + (side - cropped.shape[1]) // 2
        canvas[
            top : top + cropped.shape[0], left : left + cropped.shape[1]
        ] = np.clip(cropped * 255.0, 0, 255).astype(np.uint8)
        resized = Image.fromarray(canvas, mode="L").resize(
            (mask_size, mask_size), resample=Image.Resampling.BILINEAR
        )
        global_mask = (
            np.asarray(resized, dtype=np.float32)[None, :, :] / 255.0
        )
        context = (global_geometry, global_mask)
        self._global_context_cache[key] = context
        while len(self._global_context_cache) > self.config.global_context_cache_size:
            self._global_context_cache.popitem(last=False)
        return context

    def sample_entry_window(
        self,
        entry: ObservationBoundaryEntry,
        center_local: int,
        *,
        sampled_role: str = "natural",
        include_raster: bool = True,
        include_global_context: bool = False,
        global_context_points: int = 256,
        global_mask_size: int = 64,
    ) -> dict[str, Any]:
        if not 0 <= center_local < entry.point_count:
            raise IndexError(
                f"center_local {center_local} is outside contour with "
                f"{entry.point_count} points"
            )
        arrays = self._load_puzzle(entry)
        indices = self._sequence_indices(entry, center_local)
        local_indices = indices - entry.boundary_offset_start
        points = arrays["points_xy"][indices].astype(np.float32, copy=False)
        tangent = arrays["tangent_xy"][indices].astype(np.float32, copy=False)
        normal = arrays["inward_normal_xy"][indices].astype(np.float32, copy=False)
        geometry = np.column_stack(
            (
                tangent,
                normal,
                arrays["curvature_per_px"][indices],
                arrays["corner_strength"][indices],
                np.log1p(arrays["inward_thickness_px"][indices]),
            )
        ).astype(np.float32, copy=False)
        targets = np.column_stack(
            (
                arrays["seam_target"][indices],
                arrays["page_border_target"][indices],
                arrays["uncertainty_target"][indices],
            )
        ).astype(np.float32, copy=False)
        center = self.config.sequence_length // 2
        result: dict[str, Any] = {
            "geometry": geometry,
            "targets": targets,
            "label_confidence": arrays["label_confidence"][indices].astype(
                np.float32, copy=False
            ),
            "center_target": targets[center],
            "sampled_role": sampled_role,
            "boundary_indices_local": local_indices.astype(np.int64, copy=False),
            "metadata": {
                "puzzle_id": entry.puzzle_id,
                "observation_id": entry.observation_id,
                "piece_id": entry.piece_id,
                "view_index": entry.view_index,
                "split": entry.split,
                "difficulty_tier": entry.difficulty_tier,
                "center_boundary_index_local": center_local,
            },
        }
        if include_raster:
            result["raster"] = self._raster_strip(entry, points, normal)
        if include_global_context:
            global_geometry, global_mask = self._global_context(
                entry,
                arrays,
                point_count=global_context_points,
                mask_size=global_mask_size,
            )
            result["global_geometry"] = global_geometry
            result["global_mask"] = global_mask
        return result

    def full_observation(self, entry: ObservationBoundaryEntry) -> dict[str, np.ndarray]:
        """Return all ordered features and targets for one observed contour."""
        arrays = self._load_puzzle(entry)
        start, end = entry.boundary_offset_start, entry.boundary_offset_end
        geometry = np.column_stack(
            (
                arrays["tangent_xy"][start:end],
                arrays["inward_normal_xy"][start:end],
                arrays["curvature_per_px"][start:end],
                arrays["corner_strength"][start:end],
                np.log1p(arrays["inward_thickness_px"][start:end]),
            )
        ).astype(np.float32, copy=False)
        targets = np.column_stack(
            (
                arrays["seam_target"][start:end],
                arrays["page_border_target"][start:end],
                arrays["uncertainty_target"][start:end],
            )
        ).astype(np.float32, copy=False)
        return {
            "points_xy": arrays["points_xy"][start:end].astype(
                np.float32, copy=False
            ),
            "geometry": geometry,
            "targets": targets,
            "label_confidence": arrays["label_confidence"][start:end].astype(
                np.float32, copy=False
            ),
        }

    def sample(
        self,
        rng: np.random.Generator,
        *,
        split: str = "train",
        role: str | None = None,
        include_raster: bool = True,
        include_global_context: bool = False,
        global_context_points: int = 256,
        global_mask_size: int = 64,
        natural: bool = False,
    ) -> dict[str, Any]:
        if natural:
            entry, center_local = self._natural_entry_and_center(rng, split)
            return self.sample_entry_window(
                entry,
                center_local,
                sampled_role="natural",
                include_raster=include_raster,
                include_global_context=include_global_context,
                global_context_points=global_context_points,
                global_mask_size=global_mask_size,
            )
        if role is None:
            available_roles = [
                name
                for name in SAMPLING_CATEGORY_NAMES
                if (
                    self.index.by_split_transition.get(split)
                    if name == "transition"
                    else self.index.by_split_role.get((split, name))
                )
            ]
            if not available_roles:
                raise ValueError(f"split {split!r} contains no boundary-role samples")
            role = available_roles[int(rng.integers(len(available_roles)))]
        if role == "transition":
            entry, arrays, center_local = self._transition_entry_and_center(
                rng, split
            )
        else:
            entry = self._entry_for_role(rng, split, role)
            arrays = self._load_puzzle(entry)
            center_local = self._center_index(arrays, entry, role, rng)
        return self.sample_entry_window(
            entry,
            center_local,
            sampled_role=role,
            include_raster=include_raster,
            include_global_context=include_global_context,
            global_context_points=global_context_points,
            global_mask_size=global_mask_size,
        )


class TorchBoundaryWindowDataset(_TorchIterableDataset):
    """Optional PyTorch iterable wrapper with deterministic epoch/worker seeds."""

    def __init__(
        self,
        sampler: BoundaryWindowSampler,
        *,
        split: str = "train",
        samples_per_epoch: int = 16_384,
        seed: int = 20261006,
        include_raster: bool = True,
        include_global_context: bool = False,
        global_context_points: int = 256,
        global_mask_size: int = 64,
        role_sampling_weights: dict[str, float] | None = None,
        natural: bool = False,
    ):
        super().__init__()
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.sampler = sampler
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.include_raster = include_raster
        self.include_global_context = include_global_context
        self.global_context_points = global_context_points
        self.global_mask_size = global_mask_size
        weights = normalized_sampling_weights(role_sampling_weights)
        self.role_probabilities = np.asarray(
            [weights[name] for name in SAMPLING_CATEGORY_NAMES],
            dtype=np.float64,
        )
        self.natural = natural
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TorchBoundaryWindowDataset requires PyTorch; install the CUDA/CPU "
                "build appropriate for this machine"
            ) from exc
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        sequence = np.random.SeedSequence([self.seed, self.epoch, worker_id])
        rng = np.random.default_rng(sequence)
        for sample_index in range(worker_id, self.samples_per_epoch, worker_count):
            del sample_index
            role = None
            if not self.natural:
                role = SAMPLING_CATEGORY_NAMES[
                    int(
                        rng.choice(
                            len(SAMPLING_CATEGORY_NAMES), p=self.role_probabilities
                        )
                    )
                ]
            sample = self.sampler.sample(
                rng,
                split=self.split,
                role=role,
                include_raster=self.include_raster,
                include_global_context=self.include_global_context,
                global_context_points=self.global_context_points,
                global_mask_size=self.global_mask_size,
                natural=self.natural,
            )
            converted = {
                "geometry": torch.from_numpy(sample["geometry"]),
                "targets": torch.from_numpy(sample["targets"]),
                "label_confidence": torch.from_numpy(sample["label_confidence"]),
                "center_target": torch.from_numpy(sample["center_target"]),
                "sampled_role": sample["sampled_role"],
                "metadata": sample["metadata"],
                "boundary_indices_local": torch.from_numpy(
                    sample["boundary_indices_local"]
                ),
            }
            if self.include_raster:
                converted["raster"] = torch.from_numpy(sample["raster"])
            if self.include_global_context:
                converted["global_geometry"] = torch.from_numpy(
                    sample["global_geometry"]
                )
                converted["global_mask"] = torch.from_numpy(sample["global_mask"])
            yield converted


def inspect_samples(
    index: BoundaryTrainingIndex,
    config: BoundaryWindowConfig,
    *,
    split: str,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    sampler = BoundaryWindowSampler(index, config)
    rng = np.random.default_rng(seed)
    role_counts = {role: 0 for role in SAMPLING_CATEGORY_NAMES}
    target_sum_error = 0.0
    finite = True
    geometry_shape: list[int] | None = None
    raster_shape: list[int] | None = None
    for _ in range(sample_count):
        sample = sampler.sample(rng, split=split, include_raster=True)
        role_counts[sample["sampled_role"]] += 1
        finite = finite and bool(
            np.all(np.isfinite(sample["geometry"]))
            and np.all(np.isfinite(sample["raster"]))
            and np.all(np.isfinite(sample["targets"]))
        )
        target_sum_error = max(
            target_sum_error,
            float(np.max(np.abs(sample["targets"].sum(axis=1) - 1.0))),
        )
        geometry_shape = list(sample["geometry"].shape)
        raster_shape = list(sample["raster"].shape)
    return {
        "valid": finite and target_sum_error <= 1e-4,
        "index": index.summary(),
        "window_config": asdict(config),
        "inspected_split": split,
        "sample_count": sample_count,
        "sampled_role_counts": role_counts,
        "geometry_shape": geometry_shape,
        "raster_shape": raster_shape,
        "maximum_target_sum_error": target_sum_error,
        "all_values_finite": finite,
    }

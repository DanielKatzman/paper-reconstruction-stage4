from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .training_data import GEOMETRY_FEATURE_NAMES, ROLE_NAMES, BoundaryTrainingIndex


STATISTICS_SCHEMA_VERSION = "stage2.boundary-training-statistics.v2"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _geometry_from_archive(archive: Any) -> np.ndarray:
    return np.column_stack(
        (
            archive["tangent_xy"],
            archive["inward_normal_xy"],
            archive["curvature_per_px"],
            archive["corner_strength"],
            np.log1p(archive["inward_thickness_px"]),
        )
    ).astype(np.float64, copy=False)


@dataclass(slots=True)
class StatisticsConfig:
    boundary_dataset_dir: Path
    observation_dataset_dir: Path
    output_path: Path
    split: str = "train"
    resume: bool = False


class TrainingStatisticsCompiler:
    def __init__(self, config: StatisticsConfig):
        self.config = config
        self.index = BoundaryTrainingIndex(
            config.boundary_dataset_dir, config.observation_dataset_dir
        )
        if config.split not in self.index.by_split:
            raise ValueError(
                f"unknown split {config.split!r}; available: {self.index.splits}"
            )
        paths: list[str] = []
        seen: set[str] = set()
        for entry_index in self.index.by_split[config.split]:
            path = self.index.entries[entry_index].boundaries_path
            if path not in seen:
                seen.add(path)
                paths.append(path)
        self.boundary_paths = tuple(paths)

    def _initial_state(self) -> dict[str, Any]:
        feature_count = len(GEOMETRY_FEATURE_NAMES)
        return {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "status": "running",
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "split": self.config.split,
            "boundary_dataset_fingerprint": self.index.boundary_dataset[
                "boundary_dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": self.index.observation_dataset[
                "observation_dataset_fingerprint"
            ],
            "completed_boundary_paths": [],
            "accumulators": {
                "point_count": 0,
                "geometry_sum": [0.0] * feature_count,
                "geometry_sum_sq": [0.0] * feature_count,
                "geometry_min": [float("inf")] * feature_count,
                "geometry_max": [float("-inf")] * feature_count,
                "soft_role_sum": [0.0] * len(ROLE_NAMES),
                "hard_role_counts": [0] * len(ROLE_NAMES),
                "transition_point_count": 0,
            },
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.config.output_path.exists():
            return self._initial_state()
        if not self.config.resume:
            raise FileExistsError(
                f"statistics output already exists: {self.config.output_path}; "
                "pass --resume to validate or continue it"
            )
        state = _read_json(self.config.output_path)
        if state.get("schema_version") != STATISTICS_SCHEMA_VERSION:
            raise RuntimeError("statistics schema is incompatible")
        if state.get("split") != self.config.split:
            raise RuntimeError("statistics split changed")
        if (
            state.get("boundary_dataset_fingerprint")
            != self.index.boundary_dataset.get("boundary_dataset_fingerprint")
            or state.get("observation_dataset_fingerprint")
            != self.index.observation_dataset.get("observation_dataset_fingerprint")
        ):
            raise RuntimeError("statistics dataset fingerprints changed")
        return state

    @staticmethod
    def _update_state(state: dict[str, Any], archive: Any) -> None:
        geometry = _geometry_from_archive(archive)
        targets = np.column_stack(
            (
                archive["seam_target"],
                archive["page_border_target"],
                archive["uncertainty_target"],
            )
        ).astype(np.float64, copy=False)
        accumulators = state["accumulators"]
        accumulators["point_count"] += int(len(geometry))
        accumulators["geometry_sum"] = (
            np.asarray(accumulators["geometry_sum"], dtype=np.float64)
            + geometry.sum(axis=0, dtype=np.float64)
        ).tolist()
        accumulators["geometry_sum_sq"] = (
            np.asarray(accumulators["geometry_sum_sq"], dtype=np.float64)
            + np.square(geometry).sum(axis=0, dtype=np.float64)
        ).tolist()
        accumulators["geometry_min"] = np.minimum(
            np.asarray(accumulators["geometry_min"], dtype=np.float64),
            geometry.min(axis=0),
        ).tolist()
        accumulators["geometry_max"] = np.maximum(
            np.asarray(accumulators["geometry_max"], dtype=np.float64),
            geometry.max(axis=0),
        ).tolist()
        accumulators["soft_role_sum"] = (
            np.asarray(accumulators["soft_role_sum"], dtype=np.float64)
            + targets.sum(axis=0, dtype=np.float64)
        ).tolist()
        hard_roles = np.argmax(targets, axis=1)
        accumulators["hard_role_counts"] = (
            np.asarray(accumulators["hard_role_counts"], dtype=np.int64)
            + np.bincount(hard_roles, minlength=len(ROLE_NAMES))
        ).tolist()
        transition_count = 0
        offsets = np.asarray(archive["observation_offsets"], dtype=np.int64)
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            roles = hard_roles[int(start) : int(end)]
            if len(roles):
                transition_count += int(np.sum(roles != np.roll(roles, 1)))
        accumulators["transition_point_count"] += transition_count

    @staticmethod
    def _finalize(state: dict[str, Any]) -> dict[str, Any]:
        accumulators = state["accumulators"]
        count = int(accumulators["point_count"])
        if count <= 0:
            raise RuntimeError("statistics split contains no boundary points")
        sums = np.asarray(accumulators["geometry_sum"], dtype=np.float64)
        sum_sq = np.asarray(accumulators["geometry_sum_sq"], dtype=np.float64)
        mean = sums / count
        variance = np.maximum(sum_sq / count - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        hard_counts = np.asarray(accumulators["hard_role_counts"], dtype=np.int64)
        soft_sums = np.asarray(accumulators["soft_role_sum"], dtype=np.float64)
        state["status"] = "complete"
        state["completed_utc"] = _utc_now()
        state["geometry"] = {
            "feature_names": list(GEOMETRY_FEATURE_NAMES),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "minimum": accumulators["geometry_min"],
            "maximum": accumulators["geometry_max"],
        }
        state["roles"] = {
            "names": list(ROLE_NAMES),
            "hard_counts": {
                name: int(hard_counts[index])
                for index, name in enumerate(ROLE_NAMES)
            },
            "hard_priors": {
                name: float(hard_counts[index] / count)
                for index, name in enumerate(ROLE_NAMES)
            },
            "mean_soft_targets": {
                name: float(soft_sums[index] / count)
                for index, name in enumerate(ROLE_NAMES)
            },
            "transition_point_count": int(accumulators["transition_point_count"]),
            "transition_point_prior": float(
                accumulators["transition_point_count"] / count
            ),
        }
        return state

    def compile(self) -> dict[str, Any]:
        state = self._load_or_initialize()
        if state.get("status") == "complete":
            return state
        completed = set(state.get("completed_boundary_paths", []))
        started = time.perf_counter()
        for index, relative_path in enumerate(self.boundary_paths):
            if relative_path in completed:
                continue
            path = self.config.boundary_dataset_dir / relative_path
            with np.load(path, allow_pickle=False) as archive:
                self._update_state(state, archive)
            state["completed_boundary_paths"].append(relative_path)
            state["updated_utc"] = _utc_now()
            _write_json_atomic(self.config.output_path, state)
            elapsed = time.perf_counter() - started
            processed_now = index + 1
            rate = processed_now / max(elapsed, 1e-9)
            remaining = (len(self.boundary_paths) - processed_now) / max(rate, 1e-9)
            print(
                f"[{processed_now}/{len(self.boundary_paths)}] {relative_path}, "
                f"points={state['accumulators']['point_count']}, ETA {remaining:.1f}s",
                flush=True,
            )
        state = self._finalize(state)
        state["updated_utc"] = _utc_now()
        _write_json_atomic(self.config.output_path, state)
        return state

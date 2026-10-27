from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from . import __version__
from .boundary import BoundaryConfig, BoundaryDatasetCompiler
from .config import GeneratorConfig
from .generator import DatasetGenerator
from .observation import ObservationCompiler, ObservationConfig


PIPELINE_SCHEMA_VERSION = "stage2.pipeline-state.v1"
STAGE_ORDER = ("canonical", "observations", "boundaries")


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


class PipelineLock(AbstractContextManager["PipelineLock"]):
    """A process lock that is automatically released after crashes or exits."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"another Stage 2 pipeline process is using {self.path.parent}"
            ) from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


@dataclass(slots=True)
class PipelinePaths:
    root: Path
    canonical: Path
    observations: Path
    boundaries: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        canonical: Path | None = None,
        observations: Path | None = None,
        boundaries: Path | None = None,
    ) -> "PipelinePaths":
        return cls(
            root=root,
            canonical=canonical or root / "canonical",
            observations=observations or root / "observations",
            boundaries=boundaries or root / "boundaries",
        )


class Stage2Pipeline:
    def __init__(
        self,
        *,
        paths: PipelinePaths,
        canonical_config: GeneratorConfig,
        observation_config: ObservationConfig,
        boundary_config: BoundaryConfig,
        resume: bool,
    ):
        self.paths = paths
        self.canonical_config = canonical_config
        self.observation_config = observation_config
        self.boundary_config = boundary_config
        self.resume = resume
        self.state_path = paths.root / "pipeline_state.json"
        self.lock_path = paths.root / "pipeline.lock"

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "pipeline_version": __version__,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "status": "created",
            "paths": {
                "canonical": str(self.paths.canonical.resolve()),
                "observations": str(self.paths.observations.resolve()),
                "boundaries": str(self.paths.boundaries.resolve()),
            },
            "stages": {
                name: {"status": "pending", "completed_puzzles": 0}
                for name in STAGE_ORDER
            },
        }

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = _read_json(self.state_path)
            if state.get("schema_version") != PIPELINE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported pipeline state schema: {state.get('schema_version')}"
                )
            return state
        return self._new_state()

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_utc"] = _utc_now()
        _write_json_atomic(self.state_path, state)

    @staticmethod
    def _committed_puzzle_count(stage_dir: Path, manifest_name: str) -> int:
        puzzles = stage_dir / "puzzles"
        if not puzzles.is_dir():
            return 0
        return sum(1 for _ in puzzles.glob(f"*/{manifest_name}"))

    def _stage_progress(self, stage: str) -> tuple[int, int]:
        if stage == "canonical":
            completed = self._committed_puzzle_count(
                self.paths.canonical, "puzzle.json"
            )
            total = self.canonical_config.puzzle_count
        elif stage == "observations":
            completed = self._committed_puzzle_count(
                self.paths.observations, "observation_manifest.json"
            )
            total = self.canonical_config.puzzle_count
        else:
            completed = self._committed_puzzle_count(
                self.paths.boundaries, "boundary_manifest.json"
            )
            total = self.canonical_config.puzzle_count
        return completed, total

    @staticmethod
    def _assert_puzzle_manifests(
        dataset_dir: Path,
        summaries: list[dict[str, Any]],
        path_key: str,
    ) -> None:
        missing = [
            str(dataset_dir / summary[path_key])
            for summary in summaries
            if not (dataset_dir / summary[path_key]).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"dataset is missing {len(missing)} committed puzzle manifests; "
                f"first missing path: {missing[0]}"
            )

    def _validate_canonical(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema_version") != "stage2.canonical-dataset.v2":
            raise RuntimeError("unsupported canonical dataset schema")
        if len(manifest.get("puzzles", [])) != self.canonical_config.puzzle_count:
            raise RuntimeError(
                "completed canonical dataset puzzle count does not match --count"
            )
        expected = self.canonical_config.public_dict()
        expected.pop("output_dir", None)
        expected.pop("source_dir", None)
        if manifest.get("generation_config") != expected:
            raise RuntimeError(
                "completed canonical dataset configuration does not match the "
                "requested pipeline configuration"
            )
        self._assert_puzzle_manifests(
            self.paths.canonical, manifest["puzzles"], "puzzle_path"
        )

    def _validate_observations(
        self, manifest: dict[str, Any], canonical: dict[str, Any]
    ) -> None:
        if manifest.get("schema_version") != "stage2.observation-dataset.v1":
            raise RuntimeError("unsupported observation dataset schema")
        if (
            manifest.get("canonical_dataset_fingerprint")
            != canonical.get("dataset_fingerprint")
        ):
            raise RuntimeError(
                "observation dataset belongs to a different canonical dataset"
            )
        if manifest.get("puzzle_count") != len(canonical.get("puzzles", [])):
            raise RuntimeError("observation dataset puzzle count is incomplete")
        if manifest.get("observation_config") != self.observation_config.public_dict():
            raise RuntimeError(
                "completed observation dataset configuration does not match the "
                "requested pipeline configuration"
            )
        self._assert_puzzle_manifests(
            self.paths.observations,
            manifest.get("puzzles", []),
            "observation_manifest_path",
        )

    def _validate_boundaries(
        self,
        manifest: dict[str, Any],
        canonical: dict[str, Any],
        observations: dict[str, Any],
    ) -> None:
        if manifest.get("schema_version") != "stage2.boundary-dataset.v1":
            raise RuntimeError("unsupported boundary dataset schema")
        if (
            manifest.get("canonical_dataset_fingerprint")
            != canonical.get("dataset_fingerprint")
            or manifest.get("observation_dataset_fingerprint")
            != observations.get("observation_dataset_fingerprint")
        ):
            raise RuntimeError("boundary dataset input fingerprints do not match")
        if manifest.get("puzzle_count") != len(canonical.get("puzzles", [])):
            raise RuntimeError("boundary dataset puzzle count is incomplete")
        if manifest.get("boundary_config") != self.boundary_config.public_dict():
            raise RuntimeError(
                "completed boundary dataset configuration does not match the "
                "requested pipeline configuration"
            )
        self._assert_puzzle_manifests(
            self.paths.boundaries,
            manifest.get("puzzles", []),
            "boundary_manifest_path",
        )

    def _mark_stage(
        self,
        state: dict[str, Any],
        stage: str,
        status: str,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        completed, total = self._stage_progress(stage)
        record = state["stages"][stage]
        record.update(
            {
                "status": status,
                "completed_puzzles": completed,
                "total_puzzles": total,
            }
        )
        if manifest is not None:
            fingerprint_keys = {
                "canonical": "dataset_fingerprint",
                "observations": "observation_dataset_fingerprint",
                "boundaries": "boundary_dataset_fingerprint",
            }
            record["fingerprint"] = manifest[fingerprint_keys[stage]]
            record["manifest_version"] = manifest.get(
                "generator_version", manifest.get("compiler_version")
            )
        self._save_state(state)

    def status(self) -> dict[str, Any]:
        state = self._load_state()
        for stage in STAGE_ORDER:
            completed, total = self._stage_progress(stage)
            state["stages"].setdefault(stage, {})
            state["stages"][stage]["completed_puzzles"] = completed
            state["stages"][stage]["total_puzzles"] = total
        return state

    def verify(self) -> dict[str, Any]:
        canonical = _read_json(self.paths.canonical / "dataset_manifest.json")
        self._validate_canonical(canonical)
        observations = _read_json(
            self.paths.observations / "observation_dataset_manifest.json"
        )
        self._validate_observations(observations, canonical)
        boundaries = _read_json(
            self.paths.boundaries / "boundary_dataset_manifest.json"
        )
        self._validate_boundaries(boundaries, canonical, observations)
        return {
            "valid": True,
            "canonical_fingerprint": canonical["dataset_fingerprint"],
            "observation_fingerprint": observations[
                "observation_dataset_fingerprint"
            ],
            "boundary_fingerprint": boundaries["boundary_dataset_fingerprint"],
            "puzzle_count": len(canonical["puzzles"]),
            "observation_count": boundaries["observation_count"],
            "boundary_point_count": boundaries["boundary_point_count"],
        }

    def run(self, *, stop_after: str = "boundaries") -> dict[str, Any]:
        if stop_after not in STAGE_ORDER:
            raise ValueError(f"unknown stop_after stage: {stop_after}")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if any(self.paths.root.iterdir()) and not self.resume:
            raise FileExistsError(
                f"pipeline output is not empty: {self.paths.root}. Pass --resume "
                "to reuse completed puzzles."
            )
        with PipelineLock(self.lock_path):
            state = self._load_state()
            state["status"] = "running"
            self._save_state(state)
            try:
                canonical_path = self.paths.canonical / "dataset_manifest.json"
                if canonical_path.is_file():
                    canonical = _read_json(canonical_path)
                    self._validate_canonical(canonical)
                else:
                    self._mark_stage(state, "canonical", "running")
                    canonical = DatasetGenerator(self.canonical_config).generate()
                self._mark_stage(state, "canonical", "complete", manifest=canonical)
                if stop_after == "canonical":
                    state["status"] = "stopped_after_canonical"
                    self._save_state(state)
                    return deepcopy(state)

                observation_path = (
                    self.paths.observations / "observation_dataset_manifest.json"
                )
                if observation_path.is_file():
                    observations = _read_json(observation_path)
                    self._validate_observations(observations, canonical)
                else:
                    self._mark_stage(state, "observations", "running")
                    observations = ObservationCompiler(
                        self.observation_config
                    ).compile()
                self._mark_stage(
                    state, "observations", "complete", manifest=observations
                )
                if stop_after == "observations":
                    state["status"] = "stopped_after_observations"
                    self._save_state(state)
                    return deepcopy(state)

                boundary_path = (
                    self.paths.boundaries / "boundary_dataset_manifest.json"
                )
                if boundary_path.is_file():
                    boundaries = _read_json(boundary_path)
                    self._validate_boundaries(boundaries, canonical, observations)
                else:
                    self._mark_stage(state, "boundaries", "running")
                    boundaries = BoundaryDatasetCompiler(
                        self.boundary_config
                    ).compile()
                self._mark_stage(
                    state, "boundaries", "complete", manifest=boundaries
                )
                state["status"] = "complete"
                state["verification"] = self.verify()
                self._save_state(state)
                return deepcopy(state)
            except KeyboardInterrupt:
                state["status"] = "interrupted"
                for stage in STAGE_ORDER:
                    completed, total = self._stage_progress(stage)
                    state["stages"][stage]["completed_puzzles"] = completed
                    state["stages"][stage]["total_puzzles"] = total
                self._save_state(state)
                raise
            except Exception as exc:
                state["status"] = "failed"
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                for stage in STAGE_ORDER:
                    completed, total = self._stage_progress(stage)
                    state["stages"][stage]["completed_puzzles"] = completed
                    state["stages"][stage]["total_puzzles"] = total
                self._save_state(state)
                raise

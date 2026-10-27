from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .atomic_io import atomic_output_path, atomic_write_json
from .inference import FullContourPredictor, checkpoint_sha256
from .pipeline import PipelineLock
from .training_data import ROLE_NAMES, BoundaryTrainingIndex, BoundaryWindowSampler


PREDICTION_STATE_SCHEMA_VERSION = "stage2.boundary-role-prediction-state.v1"
PREDICTION_MANIFEST_SCHEMA_VERSION = "stage2.boundary-role-predictions.v1"
PREDICTION_OBSERVATION_SCHEMA_VERSION = "stage2.boundary-role-prediction.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_role_overlay(
    image_path: Path,
    points_xy: np.ndarray,
    role_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    colors = np.asarray(
        (
            (235, 52, 52),    # seam
            (0, 190, 225),    # original page border
            (245, 185, 35),   # uncertain/junction
        ),
        dtype=np.float64,
    )
    with Image.open(image_path) as opened:
        canvas = opened.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(points_xy, dtype=np.float64)
    probabilities = np.asarray(role_probabilities, dtype=np.float64)
    for index in range(len(points)):
        following = (index + 1) % len(points)
        color = tuple(
            int(value)
            for value in np.clip(probabilities[index] @ colors, 0, 255)
        )
        draw.line(
            [tuple(points[index]), tuple(points[following])],
            fill=color,
            width=3,
        )
    with atomic_output_path(output_path, preserve_suffix=True) as temporary:
        canvas.save(temporary)


@dataclass(slots=True)
class PredictionExportConfig:
    checkpoint_path: Path
    output_dir: Path
    split: str | None = None
    maximum_observations: int | None = None
    write_overlays: bool = True
    resume: bool = False

    def validate(self) -> None:
        if self.split not in {None, "train", "validation", "calibration", "holdout"}:
            raise ValueError("split must be train, validation, calibration, holdout, or all")
        if self.maximum_observations is not None and self.maximum_observations < 1:
            raise ValueError("maximum_observations must be positive")


class BoundaryPredictionExporter:
    def __init__(
        self,
        *,
        config: PredictionExportConfig,
        index: BoundaryTrainingIndex,
        sampler: BoundaryWindowSampler,
        predictor: FullContourPredictor,
    ):
        config.validate()
        self.config = config
        self.index = index
        self.sampler = sampler
        self.predictor = predictor
        self.checkpoint_sha = checkpoint_sha256(config.checkpoint_path)
        self.config_payload = {
            "checkpoint_sha256": self.checkpoint_sha,
            "boundary_dataset_fingerprint": index.boundary_dataset[
                "boundary_dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": index.observation_dataset[
                "observation_dataset_fingerprint"
            ],
            "split": config.split,
            "maximum_observations": config.maximum_observations,
            "write_overlays": config.write_overlays,
            "inference_config": asdict(predictor.config),
            "fragment_config": predictor.fragment_config.public_dict(),
        }
        self.config_hash = _hash_payload(self.config_payload)
        self.state_path = config.output_dir / "prediction_state.json"
        self.manifest_path = config.output_dir / "prediction_manifest.json"

    def _entries(self) -> list[Any]:
        if self.config.split is None:
            indices = list(range(len(self.index.entries)))
        else:
            indices = list(self.index.by_split.get(self.config.split, ()))
        if self.config.maximum_observations is not None:
            indices = indices[: self.config.maximum_observations]
        if not indices:
            raise ValueError("no observations match the requested prediction selection")
        return [self.index.entries[index] for index in indices]

    def _prepare_output(self) -> dict[str, Any]:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"prediction output is not empty: {output}; pass --resume or choose "
                "a new output directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not self.config.resume:
                raise FileExistsError("prediction state exists; pass --resume")
            if state.get("schema_version") != PREDICTION_STATE_SCHEMA_VERSION:
                raise RuntimeError("prediction state schema is incompatible")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError("prediction configuration changed; refusing unsafe resume")
            return state
        if self.config.resume:
            unexpected = [
                path.name
                for path in output.iterdir()
                if path.name != "prediction.lock"
            ]
            if unexpected:
                raise RuntimeError(
                    "resume requested but prediction_state.json is missing from a "
                    f"non-empty directory: {unexpected}"
                )
        state = {
            "schema_version": PREDICTION_STATE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "status": "running",
            "config_hash": self.config_hash,
            "completed_observation_count": 0,
        }
        _write_json_atomic(self.state_path, state)
        return state

    def _paths(self, entry: Any) -> tuple[Path, Path, Path]:
        directory = self.config.output_dir / "puzzles" / entry.puzzle_id
        return (
            directory / f"{entry.observation_id}.npz",
            directory / f"{entry.observation_id}.json",
            directory / "overlays" / f"{entry.observation_id}.png",
        )

    def _is_committed(self, entry: Any) -> bool:
        npz_path, json_path, overlay_path = self._paths(entry)
        if not json_path.is_file() or not npz_path.is_file():
            return False
        if self.config.write_overlays and not overlay_path.is_file():
            return False
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            record.get("schema_version") == PREDICTION_OBSERVATION_SCHEMA_VERSION
            and record.get("config_hash") == self.config_hash
            and int(record.get("point_count", -1)) == entry.point_count
        )

    def export(self) -> dict[str, Any]:
        entries = self._entries()
        state = self._prepare_output()
        with PipelineLock(self.config.output_dir / "prediction.lock"):
            started = time.perf_counter()
            completed = 0
            reused = 0
            for position, entry in enumerate(entries, start=1):
                if self._is_committed(entry):
                    completed += 1
                    reused += 1
                    continue
                prediction = self.predictor.predict(entry)
                contour = self.sampler.full_observation(entry)
                npz_path, json_path, overlay_path = self._paths(entry)
                _write_npz_atomic(
                    npz_path,
                    {
                        "points_xy": contour["points_xy"].astype(np.float32),
                        "role_logits": prediction["role_logits"],
                        "role_probabilities": prediction["role_probabilities"],
                        "transition_logits": prediction["transition_logits"],
                        "transition_probabilities": prediction[
                            "transition_probabilities"
                        ],
                    },
                )
                if self.config.write_overlays:
                    render_role_overlay(
                        self.index.observation_dataset_dir / entry.image_path,
                        contour["points_xy"],
                        prediction["role_probabilities"],
                        overlay_path,
                    )
                record = {
                    "schema_version": PREDICTION_OBSERVATION_SCHEMA_VERSION,
                    "created_utc": _utc_now(),
                    "config_hash": self.config_hash,
                    "puzzle_id": entry.puzzle_id,
                    "observation_id": entry.observation_id,
                    "piece_id": entry.piece_id,
                    "view_index": entry.view_index,
                    "split": entry.split,
                    "difficulty_tier": entry.difficulty_tier,
                    "point_count": entry.point_count,
                    "arrays_path": npz_path.relative_to(
                        self.config.output_dir
                    ).as_posix(),
                    "overlay_path": overlay_path.relative_to(
                        self.config.output_dir
                    ).as_posix()
                    if self.config.write_overlays
                    else None,
                    "role_names": list(ROLE_NAMES),
                    "candidate_fragments": prediction["candidate_fragments"],
                }
                # JSON is the commit marker and is intentionally written last.
                _write_json_atomic(json_path, record)
                completed += 1
                state.update(
                    {
                        "updated_utc": _utc_now(),
                        "status": "running",
                        "completed_observation_count": completed,
                        "current_puzzle_id": entry.puzzle_id,
                        "current_observation_id": entry.observation_id,
                    }
                )
                _write_json_atomic(self.state_path, state)
                elapsed = time.perf_counter() - started
                rate = (position - reused) / max(elapsed, 1e-9)
                remaining = (len(entries) - position) / max(rate, 1e-9)
                print(
                    f"[{position}/{len(entries)}] {entry.puzzle_id}/"
                    f"{entry.observation_id}, ETA {remaining:.1f}s",
                    flush=True,
                )

            manifest = {
                "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "complete",
                "config_hash": self.config_hash,
                "configuration": self.config_payload,
                "checkpoint_path": str(self.config.checkpoint_path.resolve()),
                "observation_count": len(entries),
                "reused_observation_count": reused,
                "puzzle_count": len({entry.puzzle_id for entry in entries}),
                "output_layout": "puzzles/<puzzle_id>/<observation_id>.{json,npz}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
            _write_json_atomic(self.manifest_path, manifest)
            state.update(
                {
                    "updated_utc": _utc_now(),
                    "completed_utc": _utc_now(),
                    "status": "complete",
                    "completed_observation_count": len(entries),
                    "current_puzzle_id": None,
                    "current_observation_id": None,
                    "manifest_path": "prediction_manifest.json",
                }
            )
            _write_json_atomic(self.state_path, state)
            return manifest

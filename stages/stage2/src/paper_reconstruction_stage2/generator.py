from __future__ import annotations

import hashlib
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import GeneratorConfig
from .metrics import add_piece_metrics, add_seam_metrics, puzzle_geometry_metrics
from .partition import generate_partition
from .render import (
    save_partition_preview,
    save_piece_artifacts,
    save_scattered_preview,
    save_seam_preview,
)
from .sources import discover_sources, load_source_page
from .splits import assign_source_splits, exact_schedule
from .truth import extract_truth
from .validation import validate_partition


SCHEMA_VERSION = "stage2.canonical-puzzle.v2"
DATASET_SCHEMA_VERSION = "stage2.canonical-dataset.v2"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _public_generation_config(config: GeneratorConfig) -> dict[str, Any]:
    result = config.public_dict()
    result.pop("output_dir", None)
    result.pop("source_dir", None)
    result.pop("resume", None)
    return result


def _externalize_truth(
    puzzle_dir: Path,
    seams: list[dict],
    borders: list[dict],
) -> tuple[list[dict], list[dict]]:
    seam_arrays = [
        np.asarray(seam["canonical_polyline_xy"], dtype=np.float32) for seam in seams
    ]
    border_arrays = [
        np.asarray(border["canonical_polyline_xy"], dtype=np.float32) for border in borders
    ]

    def concatenate(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        offsets = [0]
        for array in arrays:
            offsets.append(offsets[-1] + len(array))
        points = (
            np.concatenate(arrays, axis=0)
            if arrays
            else np.empty((0, 2), dtype=np.float32)
        )
        return points, np.asarray(offsets, dtype=np.int64)

    seam_points, seam_offsets = concatenate(seam_arrays)
    border_points, border_offsets = concatenate(border_arrays)
    np.savez_compressed(
        puzzle_dir / "truth.npz",
        seam_points_xy=seam_points,
        seam_offsets=seam_offsets,
        border_points_xy=border_points,
        border_offsets=border_offsets,
    )

    compact_seams = deepcopy(seams)
    for index, seam in enumerate(compact_seams):
        seam.pop("canonical_polyline_xy")
        seam["canonical_polyline_ref"] = {
            "path": "truth.npz",
            "points_array": "seam_points_xy",
            "offsets_array": "seam_offsets",
            "index": index,
            "point_count": int(seam_offsets[index + 1] - seam_offsets[index]),
        }

    compact_borders = deepcopy(borders)
    for index, border in enumerate(compact_borders):
        border.pop("canonical_polyline_xy")
        border["canonical_polyline_ref"] = {
            "path": "truth.npz",
            "points_array": "border_points_xy",
            "offsets_array": "border_offsets",
            "index": index,
            "point_count": int(border_offsets[index + 1] - border_offsets[index]),
        }
    return compact_seams, compact_borders


def _compact_cut_event(event: dict) -> dict:
    result = deepcopy(event)
    points = result.get("polyline_xy", [])
    if len(points) > 128:
        indices = np.linspace(0, len(points) - 1, 128).round().astype(int)
        result["polyline_xy"] = [points[index] for index in indices]
        result["polyline_decimated"] = True
        result["original_point_count"] = len(points)
    else:
        result["polyline_decimated"] = False
        result["original_point_count"] = len(points)
    return result


class DatasetGenerator:
    def __init__(self, config: GeneratorConfig):
        config.validate()
        self.config = config
        self.sources = discover_sources(config.source_dir)
        if self.sources:
            assert config.source_dir is not None
            self.source_assignments, self.source_split_records = assign_source_splits(
                self.sources,
                config.source_dir,
                config.split_ratios,
                seed=config.seed + 17,
            )
        else:
            self.source_assignments = {}
            self.source_split_records = []

    def _prepare_output(self) -> None:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"output directory is not empty: {output}. "
                "Choose a new directory or pass --resume; completed puzzles are "
                "never overwritten."
            )
        if output.exists() and self.config.resume:
            allowed = {
                "dataset_manifest.json",
                "dataset_manifest.json.tmp",
                "source_split_manifest.json",
                "source_split_manifest.json.tmp",
                "sources",
                "puzzles",
            }
            unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
            if unexpected:
                raise FileExistsError(
                    f"cannot resume canonical dataset with unrelated entries in {output}: "
                    f"{unexpected}"
                )
        output.mkdir(parents=True, exist_ok=True)
        (output / "sources").mkdir(exist_ok=True)
        (output / "puzzles").mkdir(exist_ok=True)

    def _completed_puzzles(self) -> dict[int, tuple[Path, dict[str, Any]]]:
        if not self.config.resume:
            return {}
        completed: dict[int, tuple[Path, dict[str, Any]]] = {}
        puzzles_dir = self.config.output_dir / "puzzles"
        for manifest_path in sorted(puzzles_dir.glob("*/puzzle.json")):
            if manifest_path.parent.name.endswith(".partial"):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("generator_version") != __version__:
                raise RuntimeError(
                    f"cannot resume {manifest.get('puzzle_id')}: generator version "
                    f"{manifest.get('generator_version')} does not match {__version__}"
                )
            if manifest.get("root_seed") != self.config.seed:
                raise RuntimeError(
                    f"cannot resume {manifest.get('puzzle_id')}: root seed changed"
                )
            if manifest.get("generation_config") != _public_generation_config(
                self.config
            ):
                raise RuntimeError(
                    f"cannot resume {manifest.get('puzzle_id')}: generation "
                    "configuration changed"
                )
            puzzle_index = int(manifest["puzzle_index"])
            if puzzle_index in completed:
                raise RuntimeError(
                    f"cannot resume: duplicate canonical puzzle index {puzzle_index}"
                )
            puzzle_dir = manifest_path.parent
            required = [puzzle_dir / "labels.npy", puzzle_dir / "truth.npz"]
            for piece in manifest.get("pieces", []):
                required.extend(
                    [
                        puzzle_dir / piece["canonical_piece_path"],
                        puzzle_dir / piece["canonical_mask_path"],
                    ]
                )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(
                    f"cannot resume incomplete committed puzzle {puzzle_dir}: "
                    f"missing {missing[:3]}"
                )
            completed[puzzle_index] = (puzzle_dir, manifest)
        return completed

    @staticmethod
    def _summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        return {
            "puzzle_id": manifest["puzzle_id"],
            "puzzle_path": f"puzzles/{manifest['puzzle_id']}/puzzle.json",
            "puzzle_fingerprint": manifest["puzzle_fingerprint"],
            "source_id": manifest["source"]["source_id"],
            "split": manifest["split"],
            "difficulty_tier": manifest["difficulty_tier"],
            "target_piece_count": manifest["target_piece_count"],
            "actual_piece_count": manifest["actual_piece_count"],
            "successful_cut_count": manifest["successful_cut_count"],
            "seam_count": len(manifest["seams"]),
            "rejected_split_attempts": manifest["rejected_split_attempts"],
        }

    def _sources_for_split(self, split_name: str) -> list[Path]:
        if not self.sources:
            return []
        return [
            path
            for path in self.sources
            if self.source_assignments[path] == split_name
        ]

    def generate(self) -> dict[str, Any]:
        self._prepare_output()
        started = time.perf_counter()
        puzzle_summaries: list[dict[str, Any]] = []
        completed = self._completed_puzzles()
        split_schedule = exact_schedule(
            self.config.puzzle_count,
            self.config.split_ratios,
            seed=self.config.seed + 101,
        )
        difficulty_schedule = exact_schedule(
            self.config.puzzle_count,
            self.config.difficulty_mix,
            seed=self.config.seed + 202,
        )
        source_split_payload = {
            "schema_version": "stage2.source-splits.v1",
            "root_seed": self.config.seed,
            "split_ratios": self.config.split_ratios,
            "source_grouping_unit": "entire source file; all PDF pages stay together",
            "sources": self.source_split_records,
        }
        source_split_payload["source_split_fingerprint"] = _sha256_bytes(
            _canonical_json(source_split_payload)
        )
        _write_json(
            self.config.output_dir / "source_split_manifest.json",
            source_split_payload,
        )

        for puzzle_index in range(self.config.puzzle_count):
            if puzzle_index in completed:
                _, existing_manifest = completed[puzzle_index]
                if existing_manifest.get("split") != split_schedule[puzzle_index]:
                    raise RuntimeError(
                        f"cannot resume {existing_manifest['puzzle_id']}: split schedule changed"
                    )
                if (
                    existing_manifest.get("difficulty_tier")
                    != difficulty_schedule[puzzle_index]
                ):
                    raise RuntimeError(
                        f"cannot resume {existing_manifest['puzzle_id']}: difficulty schedule changed"
                    )
                summary = self._summary_from_manifest(existing_manifest)
                puzzle_summaries.append(summary)
                elapsed = time.perf_counter() - started
                rate = (puzzle_index + 1) / max(elapsed, 1e-9)
                remaining = (self.config.puzzle_count - puzzle_index - 1) / max(
                    rate, 1e-9
                )
                print(
                    f"[{puzzle_index + 1}/{self.config.puzzle_count}] "
                    f"{summary['puzzle_id']}: {summary['actual_piece_count']} pieces, "
                    f"{summary['seam_count']} seams (reused), ETA {remaining:.1f}s",
                    flush=True,
                )
                continue
            sequence = np.random.SeedSequence([self.config.seed, puzzle_index])
            puzzle_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(sequence)
            target_piece_count = int(
                rng.integers(self.config.min_pieces, self.config.max_pieces + 1)
            )
            split_name = split_schedule[puzzle_index]
            difficulty_tier = difficulty_schedule[puzzle_index]
            split_sources = self._sources_for_split(split_name)
            source = load_source_page(
                split_sources,
                self.config.page_width,
                self.config.page_height,
                rng,
                puzzle_index,
            )
            source_path = self.config.output_dir / "sources" / f"{source.source_id}.png"
            if not source_path.exists():
                temporary_source = source_path.with_name(f"{source_path.name}.tmp")
                source.image.save(temporary_source, format="PNG")
                temporary_source.replace(source_path)

            partition = generate_partition(
                self.config,
                rng,
                target_piece_count=target_piece_count,
                difficulty_tier=difficulty_tier,
            )
            pieces, seams, borders = extract_truth(partition.labels)
            add_piece_metrics(partition.labels, pieces)
            add_seam_metrics(seams)
            geometry_metrics = puzzle_geometry_metrics(pieces, seams)
            validation = validate_partition(
                partition, pieces, seams, self.config
            )
            identity_payload = {
                "schema_version": SCHEMA_VERSION,
                "generator_version": __version__,
                "root_seed": self.config.seed,
                "puzzle_index": puzzle_index,
                "puzzle_seed": puzzle_seed,
                "source_id": source.source_id,
                "target_piece_count": target_piece_count,
                "split": split_name,
                "difficulty_tier": difficulty_tier,
                "generation_config": _public_generation_config(self.config),
                "label_sha256": _sha256_bytes(partition.labels.tobytes()),
            }
            fingerprint = _sha256_bytes(_canonical_json(identity_payload))
            puzzle_id = f"puzzle_{puzzle_index:06d}_{fingerprint[:12]}"
            final_puzzle_dir = self.config.output_dir / "puzzles" / puzzle_id
            puzzle_dir = self.config.output_dir / "puzzles" / f"{puzzle_id}.partial"
            if puzzle_dir.exists():
                shutil.rmtree(puzzle_dir)
            if final_puzzle_dir.exists():
                raise RuntimeError(
                    f"incompatible committed puzzle output exists: {final_puzzle_dir}"
                )
            preview_dir = puzzle_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=False)

            np.save(puzzle_dir / "labels.npy", partition.labels, allow_pickle=False)
            save_piece_artifacts(source.image, partition.labels, pieces, puzzle_dir)
            save_partition_preview(source.image, partition.labels, preview_dir / "partition.png")
            save_seam_preview(source.image, seams, preview_dir / "seams.png")
            save_scattered_preview(
                pieces,
                puzzle_dir,
                preview_dir / "scattered.png",
                rng,
                self.config.preview_max_side,
            )
            compact_seams, compact_borders = _externalize_truth(
                puzzle_dir, seams, borders
            )

            puzzle_manifest = {
                "schema_version": SCHEMA_VERSION,
                "generator_version": __version__,
                "puzzle_id": puzzle_id,
                "puzzle_fingerprint": fingerprint,
                "root_seed": self.config.seed,
                "puzzle_index": puzzle_index,
                "puzzle_seed": puzzle_seed,
                "split": split_name,
                "difficulty_tier": difficulty_tier,
                "generation_config": _public_generation_config(self.config),
                "source": {
                    "source_id": source.source_id,
                    "dataset_path": f"../../sources/{source.source_id}.png",
                    "origin": source.origin,
                    "page_index": source.page_index,
                    "size_hw": [source.image.height, source.image.width],
                },
                "coordinate_system": {
                    "frame": "source_page_pixels",
                    "x_direction": "right",
                    "y_direction": "down",
                    "polyline_coordinates": "grid_vertices",
                    "crop_to_source": "add bbox_xywh x,y origin",
                },
                "target_piece_count": target_piece_count,
                "actual_piece_count": len(pieces),
                "successful_cut_count": partition.successful_cut_count,
                "rejected_split_attempts": partition.rejected_attempts,
                "geometry_metrics": geometry_metrics,
                "label_map_path": "labels.npy",
                "truth_path": "truth.npz",
                "pieces": pieces,
                "seams": compact_seams,
                "page_border_segments": compact_borders,
                "cut_events": [
                    _compact_cut_event(event.to_dict()) for event in partition.cut_events
                ],
                "validation": validation,
            }
            _write_json(puzzle_dir / "puzzle.json", puzzle_manifest)
            puzzle_dir.replace(final_puzzle_dir)
            puzzle_summaries.append(self._summary_from_manifest(puzzle_manifest))
            elapsed = time.perf_counter() - started
            rate = (puzzle_index + 1) / max(elapsed, 1e-9)
            remaining = (self.config.puzzle_count - puzzle_index - 1) / max(rate, 1e-9)
            print(
                f"[{puzzle_index + 1}/{self.config.puzzle_count}] {puzzle_id}: "
                f"{len(pieces)} pieces, {len(seams)} seams, {split_name}/{difficulty_tier}, "
                f"ETA {remaining:.1f}s",
                flush=True,
            )

        dataset_payload = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "generator_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generation_config": _public_generation_config(self.config),
            "source_input": str(self.config.source_dir.resolve()) if self.config.source_dir else None,
            "source_file_count": len(self.sources),
            "source_split_manifest_path": "source_split_manifest.json",
            "source_split_fingerprint": source_split_payload[
                "source_split_fingerprint"
            ],
            "split_counts": {
                name: split_schedule.count(name) for name in self.config.split_ratios
            },
            "difficulty_counts": {
                name: difficulty_schedule.count(name)
                for name in self.config.difficulty_mix
            },
            "puzzles": puzzle_summaries,
        }
        fingerprint_payload = deepcopy(dataset_payload)
        fingerprint_payload.pop("created_utc", None)
        fingerprint_payload.pop("source_input", None)
        dataset_payload["dataset_fingerprint"] = _sha256_bytes(
            _canonical_json(fingerprint_payload)
        )
        dataset_payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write_json(self.config.output_dir / "dataset_manifest.json", dataset_payload)
        return dataset_payload

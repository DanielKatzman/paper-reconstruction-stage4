from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .production import (
    EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION,
    EDGE_HYPOTHESES_SCHEMA_VERSION,
)
from .onnx_runtime import OnnxModelRelease


REQUIRED_ARRAYS = {
    "points_xy_crop",
    "points_xy_source",
    "arc_length_px",
    "tangent_xy",
    "inward_normal_xy",
    "curvature_per_px",
    "corner_strength",
    "inward_thickness_px",
    "role_logits",
    "role_probabilities",
    "transition_logits",
    "transition_probabilities",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _contained_file(root: Path, relative_value: Any, field: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes the Stage 2 output directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _checked_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"manifest {field} is not a lowercase SHA-256 digest")
    return value


def verify_stage2_output(
    output_dir: Path,
    *,
    release_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a completed Stage 2 handoff without modifying it."""

    root = Path(output_dir).resolve()
    manifest_path = root / "edge_hypotheses_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION:
        raise ValueError("incompatible Stage 2 manifest schema")
    if manifest.get("status") != "complete" or manifest.get("truth_free") is not True:
        raise ValueError("Stage 2 manifest is not a complete truth-free export")

    expected_model_sha = _checked_sha256(
        manifest.get("model_sha256"), "model_sha256"
    )
    expected_source_sha = _checked_sha256(
        manifest.get("source_checkpoint_sha256"), "source_checkpoint_sha256"
    )
    expected_release_sha = _checked_sha256(
        manifest.get("release_manifest_sha256"), "release_manifest_sha256"
    )
    if manifest.get("model_format") != "onnx":
        raise ValueError("Stage 2 manifest model_format must be onnx")
    if manifest.get("runtime") != "onnxruntime-cpu":
        raise ValueError("Stage 2 manifest runtime must be onnxruntime-cpu")
    if release_path is not None:
        release = OnnxModelRelease.load(release_path)
        if release.model_sha256 != expected_model_sha:
            raise ValueError("release ONNX SHA-256 does not match the Stage 2 manifest")
        if release.source_checkpoint_sha256 != expected_source_sha:
            raise ValueError(
                "release source checkpoint SHA-256 does not match the Stage 2 manifest"
            )
        if release.release_version != manifest.get("release_version"):
            raise ValueError("release version does not match the Stage 2 manifest")
        if release.release_manifest_sha256 != expected_release_sha:
            raise ValueError("release.json SHA-256 does not match the Stage 2 manifest")
    if model_path is not None:
        model = Path(model_path)
        if not model.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model}")
        if _sha256_file(model) != expected_model_sha:
            raise ValueError("ONNX SHA-256 does not match the Stage 2 manifest")

    summaries = manifest.get("pieces")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("Stage 2 manifest contains no pieces")
    seen_capture_ids: set[str] = set()
    total_candidates = 0
    total_points = 0

    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("piece summary must be an object")
        record_path = _contained_file(root, summary.get("record_path"), "record_path")
        record = _load_json(record_path)
        capture_id = record.get("capture_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError(f"missing capture_id in {record_path}")
        if capture_id in seen_capture_ids:
            raise ValueError(f"duplicate capture_id in Stage 2 output: {capture_id}")
        seen_capture_ids.add(capture_id)
        if record.get("schema_version") != EDGE_HYPOTHESES_SCHEMA_VERSION:
            raise ValueError(f"{capture_id}: incompatible piece schema")
        for field in (
            "config_hash",
            "release_version",
            "release_manifest_sha256",
            "model_format",
            "runtime",
            "model_sha256",
            "source_checkpoint_sha256",
            "stage1_input_fingerprint",
        ):
            if record.get(field) != manifest.get(field):
                raise ValueError(f"{capture_id}: {field} disagrees with manifest")
        if record.get("stage1_record_fingerprint") != summary.get(
            "stage1_record_fingerprint"
        ):
            raise ValueError(f"{capture_id}: Stage 1 fingerprint mismatch")

        arrays_path = _contained_file(root, record.get("arrays_path"), "arrays_path")
        if summary.get("arrays_path") != record.get("arrays_path"):
            raise ValueError(f"{capture_id}: arrays path disagrees with manifest")
        if _sha256_file(arrays_path) != record.get("arrays_sha256"):
            raise ValueError(f"{capture_id}: arrays SHA-256 mismatch")

        point_count = int(record.get("point_count", -1))
        if point_count < 16:
            raise ValueError(f"{capture_id}: invalid point_count")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            missing = REQUIRED_ARRAYS.difference(arrays.files)
            if missing:
                raise ValueError(f"{capture_id}: missing arrays: {sorted(missing)}")
            expected_shapes = {
                "points_xy_crop": (point_count, 2),
                "points_xy_source": (point_count, 2),
                "role_logits": (point_count, 3),
                "role_probabilities": (point_count, 3),
                "transition_logits": (point_count,),
                "transition_probabilities": (point_count,),
            }
            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    raise ValueError(
                        f"{capture_id}: {name} has shape {arrays[name].shape}, "
                        f"expected {shape}"
                    )
                if not np.isfinite(arrays[name]).all():
                    raise ValueError(f"{capture_id}: {name} contains non-finite values")
            probabilities = arrays["role_probabilities"]
            if np.any(probabilities < -1e-6) or np.any(probabilities > 1.0 + 1e-6):
                raise ValueError(f"{capture_id}: role probabilities are outside [0, 1]")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError(f"{capture_id}: role probabilities do not sum to one")
            transitions = arrays["transition_probabilities"]
            if np.any(transitions < -1e-6) or np.any(transitions > 1.0 + 1e-6):
                raise ValueError(
                    f"{capture_id}: transition probabilities are outside [0, 1]"
                )

            provenance = record.get("stage1_provenance", {})
            bbox = provenance.get("bbox_xywh")
            if isinstance(bbox, list) and len(bbox) == 4:
                offset = np.asarray(bbox[:2], dtype=np.float32)
                if not np.allclose(
                    arrays["points_xy_source"],
                    arrays["points_xy_crop"] + offset,
                    atol=1e-4,
                ):
                    raise ValueError(f"{capture_id}: crop/source coordinates disagree")

        candidates = record.get("candidate_fragments")
        if not isinstance(candidates, list) or len(candidates) != record.get(
            "candidate_count"
        ):
            raise ValueError(f"{capture_id}: candidate_count mismatch")
        for position, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise ValueError(f"{capture_id}: candidate must be an object")
            if candidate.get("candidate_id") != f"candidate_{position:04d}":
                raise ValueError(f"{capture_id}: candidate IDs are not stable/ordered")
            if candidate.get("rank") != position:
                raise ValueError(f"{capture_id}: candidate ranks are not ordered")
            for field in ("start_index", "end_index"):
                index = candidate.get(field)
                if not isinstance(index, int) or not 0 <= index < point_count:
                    raise ValueError(f"{capture_id}: invalid candidate {field}")

        overlay_value = record.get("overlay_path")
        if overlay_value is not None:
            overlay_path = _contained_file(root, overlay_value, "overlay_path")
            if _sha256_file(overlay_path) != record.get("overlay_sha256"):
                raise ValueError(f"{capture_id}: overlay SHA-256 mismatch")

        total_candidates += len(candidates)
        total_points += point_count

    if len(summaries) != manifest.get("piece_count"):
        raise ValueError("piece_count does not match the manifest piece list")
    if total_candidates != manifest.get("candidate_count"):
        raise ValueError("candidate_count does not match the piece records")

    state_path = root / "inference_state.json"
    if not state_path.is_file() or _load_json(state_path).get("status") != "complete":
        raise ValueError("inference_state.json is missing or incomplete")
    return {
        "status": "ok",
        "output_dir": str(root),
        "piece_count": len(summaries),
        "point_count": total_points,
        "candidate_count": total_candidates,
        "release_version": manifest.get("release_version"),
        "release_manifest_sha256": expected_release_sha,
        "model_sha256": expected_model_sha,
        "source_checkpoint_sha256": expected_source_sha,
        "runtime": manifest.get("runtime"),
        "stage1_input_fingerprint": manifest.get("stage1_input_fingerprint"),
    }

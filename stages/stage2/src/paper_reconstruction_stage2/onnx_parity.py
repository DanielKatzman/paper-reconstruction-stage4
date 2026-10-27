from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_json
from .inference import ContourInferenceConfig, FullContourPredictor, load_boundary_role_model
from .onnx_runtime import OnnxModelRelease, predictor_from_release
from .production import (
    BoundaryGeometryConfig,
    ProductionBoundarySampler,
    load_stage1_output,
)


@dataclass(slots=True)
class EndToEndParityConfig:
    checkpoint_path: Path
    release_path: Path
    stage1_input_path: Path
    output_path: Path | None = None
    maximum_pieces: int | None = None
    maximum_absolute_error: float = 1e-4
    cpu_threads: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["rank"],
        candidate["start_index"],
        candidate["end_index"],
        candidate["length_points"],
        candidate["wraps"],
        candidate["source"],
        candidate["hypothesis"],
        candidate["split_threshold"],
    )


def validate_end_to_end_parity(config: EndToEndParityConfig) -> dict[str, Any]:
    if config.maximum_absolute_error <= 0.0:
        raise ValueError("maximum_absolute_error must be positive")
    release = OnnxModelRelease.load(config.release_path)
    inputs = load_stage1_output(
        config.stage1_input_path, maximum_pieces=config.maximum_pieces
    )
    model, checkpoint = load_boundary_role_model(config.checkpoint_path, "cpu")
    if checkpoint.get("model_config") != release.model_config.public_dict():
        raise RuntimeError("checkpoint and ONNX release model configurations differ")
    geometry_config = BoundaryGeometryConfig(
        **release.metadata.get("geometry_config", {})
    )
    torch_sampler = ProductionBoundarySampler(
        release.window_config, geometry_config
    )
    onnx_sampler = ProductionBoundarySampler(release.window_config, geometry_config)
    inference_config = ContourInferenceConfig(
        **{
            key: getattr(release.default_inference_config, key)
            for key in (
                "sequence_length",
                "strip_width",
                "stride",
                "batch_size",
                "role_temperature",
                "transition_temperature",
            )
        }
    )
    torch_predictor = FullContourPredictor(
        model,
        torch_sampler,
        inference_config,
        release.fragment_config,
        device="cpu",
    )
    onnx_predictor = predictor_from_release(
        release,
        onnx_sampler,
        cpu_threads=config.cpu_threads,
    )
    cases: list[dict[str, Any]] = []
    worst_error = 0.0
    candidate_structures_match = True
    total_points = 0
    for capture in inputs.captures:
        torch_entry, _ = torch_sampler.prepare_capture(capture)
        onnx_entry, _ = onnx_sampler.prepare_capture(capture)
        try:
            torch_result = torch_predictor.predict(torch_entry)
            onnx_result = onnx_predictor.predict(onnx_entry)
        finally:
            torch_sampler.release_capture(torch_entry)
            onnx_sampler.release_capture(onnx_entry)
        array_errors: dict[str, float] = {}
        for name in (
            "role_logits",
            "role_probabilities",
            "transition_logits",
            "transition_probabilities",
        ):
            error = float(
                np.max(np.abs(torch_result[name] - onnx_result[name]))
            )
            array_errors[f"{name}_max_abs_error"] = error
            worst_error = max(worst_error, error)
        torch_signatures = [
            _candidate_signature(item)
            for item in torch_result["candidate_fragments"]
        ]
        onnx_signatures = [
            _candidate_signature(item)
            for item in onnx_result["candidate_fragments"]
        ]
        structures_match = torch_signatures == onnx_signatures
        candidate_structures_match &= structures_match
        total_points += int(torch_result["role_logits"].shape[0])
        cases.append(
            {
                "capture_id": capture.capture_id,
                "point_count": int(torch_result["role_logits"].shape[0]),
                "pytorch_candidate_count": len(torch_signatures),
                "onnx_candidate_count": len(onnx_signatures),
                "candidate_structures_match": structures_match,
                **array_errors,
            }
        )
    if worst_error > config.maximum_absolute_error:
        raise RuntimeError(
            "end-to-end ONNX parity failed: maximum absolute error "
            f"{worst_error:.9g} exceeds {config.maximum_absolute_error:.9g}"
        )
    if not candidate_structures_match:
        raise RuntimeError("end-to-end ONNX parity changed candidate structure")
    report = {
        "schema_version": "stage2.onnx-end-to-end-parity.v1",
        "created_utc": _utc_now(),
        "status": "passed",
        "release_version": release.release_version,
        "release_manifest_sha256": release.release_manifest_sha256,
        "model_sha256": release.model_sha256,
        "source_checkpoint_sha256": release.source_checkpoint_sha256,
        "stage1_input_fingerprint": inputs.input_fingerprint,
        "piece_count": len(cases),
        "point_count": total_points,
        "candidate_structures_match": candidate_structures_match,
        "maximum_allowed_absolute_error": config.maximum_absolute_error,
        "maximum_observed_absolute_error": worst_error,
        "cases": cases,
    }
    if config.output_path is not None:
        atomic_write_json(config.output_path, report)
    return report

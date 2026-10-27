from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_output_path, atomic_write_json
from .fragments import FragmentProposalConfig
from .inference import ContourInferenceConfig, checkpoint_sha256, load_boundary_role_model
from .model import BoundaryRoleModelConfig
from .onnx_runtime import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    RELEASE_SCHEMA_VERSION,
    create_cpu_session,
    file_sha256,
    validate_session_contract,
)
from .training_data import BoundaryWindowConfig


DEFAULT_OPSET = 18


@dataclass(slots=True)
class OnnxExportConfig:
    checkpoint_path: Path
    model_output_path: Path
    release_output_path: Path
    base_release_path: Path | None = None
    expected_checkpoint_sha256: str | None = None
    opset: int = DEFAULT_OPSET
    parity_batches: tuple[int, ...] = (1, 2, 8)
    parity_seed: int = 20261008
    maximum_absolute_error: float = 1e-4
    default_batch_size: int = 8
    default_stride: int = 32

    def validate(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        if self.opset < 17:
            raise ValueError("opset must be at least 17")
        if not self.parity_batches or any(value < 1 for value in self.parity_batches):
            raise ValueError("parity batch sizes must be positive")
        if self.maximum_absolute_error <= 0.0:
            raise ValueError("maximum_absolute_error must be positive")
        if self.default_batch_size < 1 or self.default_stride < 1:
            raise ValueError("default batch size and stride must be positive")
        if self.model_output_path.resolve().parent != self.release_output_path.resolve().parent:
            raise ValueError("ONNX model and release.json must share a directory")


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires PyTorch, onnx, and onnxruntime; install "
            "requirements-export.txt"
        ) from exc
    return torch, onnx


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing_release_fields(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"base release must be a JSON object: {path}")
    return {
        key: value[key]
        for key in ("validation", "holdout", "known_limitations")
        if key in value
    }


def _random_inputs(
    model: BoundaryRoleModelConfig,
    window: BoundaryWindowConfig,
    batch: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        "geometry": rng.normal(
            size=(batch, window.sequence_length, model.geometry_features)
        ).astype(np.float32),
        "raster": rng.uniform(
            0.0,
            1.0,
            size=(
                batch,
                model.raster_channels,
                window.sequence_length,
                window.strip_width,
            ),
        ).astype(np.float32),
        "global_geometry": rng.normal(
            size=(
                batch,
                model.global_context_points,
                model.global_geometry_features,
            )
        ).astype(np.float32),
        "global_mask": rng.uniform(
            0.0,
            1.0,
            size=(batch, 1, model.global_mask_size, model.global_mask_size),
        ).astype(np.float32),
    }


def validate_random_parity(
    model: Any,
    session: Any,
    model_config: BoundaryRoleModelConfig,
    window_config: BoundaryWindowConfig,
    *,
    batch_sizes: tuple[int, ...],
    seed: int,
    maximum_absolute_error: float,
) -> dict[str, Any]:
    torch, _ = _require_dependencies()
    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    worst_error = 0.0
    model.eval()
    for batch_size in batch_sizes:
        inputs = _random_inputs(model_config, window_config, batch_size, rng)
        tensors = [torch.from_numpy(inputs[name]) for name in ONNX_INPUT_NAMES]
        with torch.no_grad():
            expected = model(*tensors)
        actual_role, actual_transition = session.run(
            list(ONNX_OUTPUT_NAMES), inputs
        )
        role_error = float(
            np.max(
                np.abs(
                    expected["role_logits"].detach().cpu().numpy() - actual_role
                )
            )
        )
        transition_error = float(
            np.max(
                np.abs(
                    expected["transition_logits"].detach().cpu().numpy()
                    - actual_transition
                )
            )
        )
        worst_error = max(worst_error, role_error, transition_error)
        cases.append(
            {
                "batch_size": batch_size,
                "role_logits_max_abs_error": role_error,
                "transition_logits_max_abs_error": transition_error,
            }
        )
    if worst_error > maximum_absolute_error:
        raise RuntimeError(
            "ONNX parity failed: maximum absolute error "
            f"{worst_error:.9g} exceeds {maximum_absolute_error:.9g}"
        )
    return {
        "status": "passed",
        "seed": seed,
        "maximum_allowed_absolute_error": maximum_absolute_error,
        "maximum_observed_absolute_error": worst_error,
        "cases": cases,
    }


def export_onnx_release(config: OnnxExportConfig) -> dict[str, Any]:
    config.validate()
    torch, onnx = _require_dependencies()
    checkpoint_sha = checkpoint_sha256(config.checkpoint_path)
    if (
        config.expected_checkpoint_sha256 is not None
        and checkpoint_sha != config.expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "source checkpoint SHA-256 does not match the frozen release: "
            f"expected {config.expected_checkpoint_sha256}, got {checkpoint_sha}"
        )
    model, checkpoint = load_boundary_role_model(config.checkpoint_path, "cpu")
    model_config = BoundaryRoleModelConfig(**checkpoint["model_config"])
    window_config = BoundaryWindowConfig(**checkpoint["window_config"])
    model_config.validate()
    window_config.validate()
    calibration = checkpoint.get("calibration", {})
    fragment_config = FragmentProposalConfig(
        **checkpoint.get("fragment_config", {})
    )
    fragment_config.validate()
    inference_config = ContourInferenceConfig(
        sequence_length=window_config.sequence_length,
        strip_width=window_config.strip_width,
        stride=config.default_stride,
        batch_size=config.default_batch_size,
        role_temperature=float(calibration.get("role_temperature", 1.0)),
        transition_temperature=float(
            calibration.get("transition_temperature", 1.0)
        ),
    )
    inference_config.validate()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            geometry: Any,
            raster: Any,
            global_geometry: Any,
            global_mask: Any,
        ) -> tuple[Any, Any]:
            output = self.wrapped(
                geometry, raster, global_geometry, global_mask
            )
            return output["role_logits"], output["transition_logits"]

    wrapper = ExportWrapper(model).eval()
    example = _random_inputs(
        model_config,
        window_config,
        1,
        np.random.default_rng(config.parity_seed),
    )
    example_tensors = tuple(
        torch.from_numpy(example[name]) for name in ONNX_INPUT_NAMES
    )
    with atomic_output_path(config.model_output_path, preserve_suffix=True) as temporary:
        torch.onnx.export(
            wrapper,
            example_tensors,
            str(temporary),
            export_params=True,
            opset_version=config.opset,
            do_constant_folding=True,
            input_names=list(ONNX_INPUT_NAMES),
            output_names=list(ONNX_OUTPUT_NAMES),
            dynamic_axes={
                name: {0: "batch"}
                for name in (*ONNX_INPUT_NAMES, *ONNX_OUTPUT_NAMES)
            },
            dynamo=False,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        properties = {
            "stage2.release_version": "1.0.0",
            "stage2.source_checkpoint_sha256": checkpoint_sha,
            "stage2.model_name": model_config.architecture,
        }
        del graph.metadata_props[:]
        for key, value in properties.items():
            item = graph.metadata_props.add()
            item.key = key
            item.value = value
        onnx.save(graph, str(temporary))

    model_sha = file_sha256(config.model_output_path)
    session = create_cpu_session(config.model_output_path)
    validate_session_contract(
        session,
        model_config,
        sequence_length=window_config.sequence_length,
        strip_width=window_config.strip_width,
    )
    parity = validate_random_parity(
        model,
        session,
        model_config,
        window_config,
        batch_sizes=config.parity_batches,
        seed=config.parity_seed,
        maximum_absolute_error=config.maximum_absolute_error,
    )
    release: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_version": "1.0.0",
        "created_utc": _utc_now(),
        "model_name": model_config.architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_checkpoint": {
            "asset": "stage2_boundary_v1.0.0.pt",
            "sha256": checkpoint_sha,
            "schema_version": checkpoint.get("schema_version"),
            "package_version": checkpoint.get("package_version"),
        },
        "production_model": {
            "asset": config.model_output_path.name,
            "format": "onnx",
            "sha256": model_sha,
            "size_bytes": config.model_output_path.stat().st_size,
            "opset": config.opset,
            "execution_provider": "CPUExecutionProvider",
            "dynamic_axes": {"batch": True, "sequence": False},
            "inputs": list(ONNX_INPUT_NAMES),
            "outputs": list(ONNX_OUTPUT_NAMES),
        },
        "boundary_dataset_fingerprint": checkpoint.get(
            "boundary_dataset_fingerprint"
        ),
        "observation_dataset_fingerprint": checkpoint.get(
            "observation_dataset_fingerprint"
        ),
        "model_config": model_config.public_dict(),
        "window_config": asdict(window_config),
        "geometry_config": {
            "boundary_spacing_px": 2.0,
            "minimum_boundary_points": 64,
            "maximum_boundary_points": 4096,
            "tangent_smoothing_sigma_samples": 1.5,
            "max_inward_thickness_px": 96,
        },
        "default_inference_config": asdict(inference_config),
        "fragment_config": fragment_config.public_dict(),
        "onnx_parity": parity,
        "validation_artifacts": {
            "end_to_end_parity": "onnx_end_to_end_parity.json"
        },
        **_existing_release_fields(config.base_release_path),
    }
    atomic_write_json(config.release_output_path, release)
    return release

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fragments import FragmentProposalConfig
from .inference import ContourInferenceConfig, ContourPredictorBase
from .model import BoundaryRoleModelConfig
from .training_data import BoundaryWindowConfig, BoundaryWindowSampler


RELEASE_SCHEMA_VERSION = "stage2.model-release.v2"
ONNX_INPUT_NAMES = (
    "geometry",
    "raster",
    "global_geometry",
    "global_mask",
)
ONNX_OUTPUT_NAMES = ("role_logits", "transition_logits")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _contained_model_path(release_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("production_model.asset must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("production_model.asset must be relative to release.json")
    release_dir = release_path.resolve().parent
    model_path = (release_dir / relative).resolve()
    if not model_path.is_relative_to(release_dir):
        raise ValueError("production_model.asset escapes the model release directory")
    if not model_path.is_file():
        raise FileNotFoundError(f"production ONNX model not found: {model_path}")
    return model_path


@dataclass(frozen=True, slots=True)
class OnnxModelRelease:
    release_path: Path
    release_manifest_sha256: str
    release_version: str
    model_name: str
    model_path: Path
    model_sha256: str
    source_checkpoint_sha256: str
    opset: int
    model_config: BoundaryRoleModelConfig
    window_config: BoundaryWindowConfig
    default_inference_config: ContourInferenceConfig
    fragment_config: FragmentProposalConfig
    metadata: dict[str, Any]

    @classmethod
    def load(cls, release_path: Path) -> "OnnxModelRelease":
        path = Path(release_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model release metadata not found: {path}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model release JSON: {path}: {exc}") from exc
        metadata = _require_mapping(metadata, "release")
        if metadata.get("schema_version") != RELEASE_SCHEMA_VERSION:
            raise ValueError(
                "incompatible model release schema: "
                f"{metadata.get('schema_version')!r}"
            )
        production = _require_mapping(
            metadata.get("production_model"), "production_model"
        )
        if production.get("format") != "onnx":
            raise ValueError("production_model.format must be 'onnx'")
        if production.get("execution_provider") != "CPUExecutionProvider":
            raise ValueError(
                "the frozen production release must use CPUExecutionProvider"
            )
        model_path = _contained_model_path(path, production.get("asset"))
        expected_model_sha = _require_sha256(
            production.get("sha256"), "production_model.sha256"
        )
        actual_model_sha = file_sha256(model_path)
        if actual_model_sha != expected_model_sha:
            raise RuntimeError(
                "production ONNX SHA-256 does not match release.json: "
                f"expected {expected_model_sha}, got {actual_model_sha}"
            )
        source = _require_mapping(
            metadata.get("source_checkpoint"), "source_checkpoint"
        )
        source_sha = _require_sha256(source.get("sha256"), "source_checkpoint.sha256")
        model_config = BoundaryRoleModelConfig(
            **_require_mapping(metadata.get("model_config"), "model_config")
        )
        model_config.validate()
        if not (
            model_config.use_geometry
            and model_config.use_raster
            and model_config.use_global_context
        ):
            raise ValueError(
                "the v1 production ONNX contract requires geometry, raster, and "
                "global-context inputs"
            )
        window_config = BoundaryWindowConfig(
            **_require_mapping(metadata.get("window_config"), "window_config")
        )
        window_config.validate()
        inference_values = _require_mapping(
            metadata.get("default_inference_config"), "default_inference_config"
        )
        inference_config = ContourInferenceConfig(**inference_values)
        inference_config.validate()
        if inference_config.sequence_length != window_config.sequence_length:
            raise ValueError("release inference/window sequence lengths disagree")
        if inference_config.strip_width != window_config.strip_width:
            raise ValueError("release inference/window strip widths disagree")
        fragment_config = FragmentProposalConfig(
            **_require_mapping(metadata.get("fragment_config"), "fragment_config")
        )
        fragment_config.validate()
        release_version = metadata.get("release_version")
        model_name = metadata.get("model_name")
        if not isinstance(release_version, str) or not release_version:
            raise ValueError("release_version must be a non-empty string")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        opset = production.get("opset")
        if not isinstance(opset, int) or opset < 17:
            raise ValueError("production_model.opset must be an integer >= 17")
        return cls(
            release_path=path,
            release_manifest_sha256=file_sha256(path),
            release_version=release_version,
            model_name=model_name,
            model_path=model_path,
            model_sha256=actual_model_sha,
            source_checkpoint_sha256=source_sha,
            opset=opset,
            model_config=model_config,
            window_config=window_config,
            default_inference_config=inference_config,
            fragment_config=fragment_config,
            metadata=metadata,
        )


def create_cpu_session(model_path: Path, cpu_threads: int = 0) -> Any:
    if cpu_threads < 0:
        raise ValueError("cpu_threads must be zero (automatic) or positive")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "CPU inference requires onnxruntime; install requirements-inference.txt"
        ) from exc
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "this onnxruntime installation does not provide CPUExecutionProvider"
        )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    if cpu_threads > 0:
        options.intra_op_num_threads = cpu_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(Path(model_path).resolve()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def validate_session_contract(
    session: Any,
    model_config: BoundaryRoleModelConfig,
    *,
    sequence_length: int | None = None,
    strip_width: int | None = None,
) -> None:
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if tuple(inputs) != ONNX_INPUT_NAMES:
        raise RuntimeError(
            f"unexpected ONNX inputs: {tuple(inputs)}; expected {ONNX_INPUT_NAMES}"
        )
    if tuple(outputs) != ONNX_OUTPUT_NAMES:
        raise RuntimeError(
            f"unexpected ONNX outputs: {tuple(outputs)}; expected {ONNX_OUTPUT_NAMES}"
        )
    expected_ranks = {
        "geometry": 3,
        "raster": 4,
        "global_geometry": 3,
        "global_mask": 4,
        "role_logits": 3,
        "transition_logits": 2,
    }
    for name, rank in expected_ranks.items():
        item = inputs.get(name)
        if item is None:
            item = outputs.get(name)
        if item is None or item.type != "tensor(float)" or len(item.shape) != rank:
            raise RuntimeError(f"ONNX tensor contract mismatch for {name}")
    fixed_dimensions = {
        "geometry": {1: sequence_length, 2: model_config.geometry_features},
        "raster": {
            1: model_config.raster_channels,
            2: sequence_length,
            3: strip_width,
        },
        "global_geometry": {
            1: model_config.global_context_points,
            2: model_config.global_geometry_features,
        },
        "global_mask": {
            1: 1,
            2: model_config.global_mask_size,
            3: model_config.global_mask_size,
        },
        "role_logits": {1: sequence_length, 2: 3},
        "transition_logits": {1: sequence_length},
    }
    for name, dimensions in fixed_dimensions.items():
        item = inputs.get(name)
        if item is None:
            item = outputs.get(name)
        for index, expected in dimensions.items():
            if expected is not None and item.shape[index] != expected:
                raise RuntimeError(
                    f"ONNX tensor {name} dimension {index} is {item.shape[index]}, "
                    f"expected {expected}"
                )


class OnnxContourPredictor(ContourPredictorBase):
    """CPU-only ONNX Runtime implementation of full-contour inference."""

    def __init__(
        self,
        session: Any,
        sampler: BoundaryWindowSampler,
        model_config: BoundaryRoleModelConfig,
        config: ContourInferenceConfig,
        fragment_config: FragmentProposalConfig,
    ):
        validate_session_contract(
            session,
            model_config,
            sequence_length=sampler.config.sequence_length,
            strip_width=sampler.config.strip_width,
        )
        self.session = session
        super().__init__(sampler, model_config, config, fragment_config)

    def _predict_batch(
        self, samples: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        feeds = {
            "geometry": np.ascontiguousarray(
                np.stack([sample["geometry"] for sample in samples]),
                dtype=np.float32,
            ),
            "raster": np.ascontiguousarray(
                np.stack([sample["raster"] for sample in samples]),
                dtype=np.float32,
            ),
            "global_geometry": np.ascontiguousarray(
                np.stack([sample["global_geometry"] for sample in samples]),
                dtype=np.float32,
            ),
            "global_mask": np.ascontiguousarray(
                np.stack([sample["global_mask"] for sample in samples]),
                dtype=np.float32,
            ),
        }
        role_logits, transition_logits = self.session.run(
            list(ONNX_OUTPUT_NAMES), feeds
        )
        return (
            np.asarray(role_logits, dtype=np.float32),
            np.asarray(transition_logits, dtype=np.float32),
        )


def predictor_from_release(
    release: OnnxModelRelease,
    sampler: BoundaryWindowSampler,
    *,
    batch_size: int | None = None,
    stride: int | None = None,
    cpu_threads: int = 0,
) -> OnnxContourPredictor:
    defaults = release.default_inference_config
    config = ContourInferenceConfig(
        sequence_length=defaults.sequence_length,
        strip_width=defaults.strip_width,
        stride=defaults.stride if stride is None else stride,
        batch_size=defaults.batch_size if batch_size is None else batch_size,
        role_temperature=defaults.role_temperature,
        transition_temperature=defaults.transition_temperature,
    )
    config.validate()
    session = create_cpu_session(release.model_path, cpu_threads=cpu_threads)
    return OnnxContourPredictor(
        session,
        sampler,
        release.model_config,
        config,
        release.fragment_config,
    )

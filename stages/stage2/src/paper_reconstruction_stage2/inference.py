from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fragments import FragmentProposalConfig, propose_fragments
from .model import BoundaryRoleModelConfig, BoundaryRoleNetV1
from .training_data import BoundaryWindowSampler, ObservationBoundaryEntry


try:
    import torch
except ImportError:
    torch = None


@dataclass(slots=True)
class ContourInferenceConfig:
    sequence_length: int = 129
    strip_width: int = 33
    stride: int = 32
    batch_size: int = 32
    role_temperature: float = 1.0
    transition_temperature: float = 1.0

    def validate(self) -> None:
        if self.sequence_length < 3 or self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd and at least 3")
        if self.strip_width < 3 or self.stride < 1 or self.batch_size < 1:
            raise ValueError("strip width, stride, and batch size must be positive")
        if self.stride > self.sequence_length // 2:
            raise ValueError("stride must be at most half the sequence length")
        if self.role_temperature <= 0.0 or self.transition_temperature <= 0.0:
            raise ValueError("temperatures must be positive")


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _sigmoid(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.clip(values / temperature, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_boundary_role_model(
    checkpoint_path: Path, device: str
) -> tuple[Any, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("model inference requires PyTorch")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = BoundaryRoleModelConfig(**checkpoint["model_config"])
    model = BoundaryRoleNetV1(
        model_config,
        geometry_mean=checkpoint["geometry_normalization"]["mean"],
        geometry_std=checkpoint["geometry_normalization"]["std"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


class ContourPredictorBase:
    """Backend-independent overlapping-window contour inference."""

    def __init__(
        self,
        sampler: BoundaryWindowSampler,
        model_config: BoundaryRoleModelConfig,
        config: ContourInferenceConfig | None = None,
        fragment_config: FragmentProposalConfig | None = None,
    ):
        self.sampler = sampler
        self.model_config = model_config
        self.config = config or ContourInferenceConfig(
            sequence_length=sampler.config.sequence_length,
            strip_width=sampler.config.strip_width,
        )
        self.config.validate()
        if self.config.sequence_length != sampler.config.sequence_length:
            raise ValueError("inference and sampler sequence lengths differ")
        if self.config.strip_width != sampler.config.strip_width:
            raise ValueError("inference and sampler strip widths differ")
        self.fragment_config = fragment_config or FragmentProposalConfig()

    def _predict_batch(
        self, samples: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(self, entry: ObservationBoundaryEntry) -> dict[str, Any]:
        point_count = entry.point_count
        centers = list(range(0, point_count, self.config.stride))
        if centers[-1] != point_count - 1:
            centers.append(point_count - 1)
        role_sum = np.zeros((point_count, 3), dtype=np.float64)
        transition_sum = np.zeros(point_count, dtype=np.float64)
        weight_sum = np.zeros(point_count, dtype=np.float64)
        window_weights = np.hanning(self.config.sequence_length).astype(np.float64) + 0.10

        for batch_start in range(0, len(centers), self.config.batch_size):
            batch_centers = centers[batch_start : batch_start + self.config.batch_size]
            samples = [
                self.sampler.sample_entry_window(
                    entry,
                    center,
                    include_raster=self.model_config.use_raster,
                    include_global_context=self.model_config.use_global_context,
                    global_context_points=self.model_config.global_context_points,
                    global_mask_size=self.model_config.global_mask_size,
                )
                for center in batch_centers
            ]
            role_logits, transition_logits = self._predict_batch(samples)
            for sample, roles, transitions in zip(
                samples, role_logits, transition_logits, strict=True
            ):
                indices = sample["boundary_indices_local"]
                for role_index in range(3):
                    np.add.at(
                        role_sum[:, role_index],
                        indices,
                        roles[:, role_index] * window_weights,
                    )
                np.add.at(
                    transition_sum, indices, transitions * window_weights
                )
                np.add.at(weight_sum, indices, window_weights)
        if np.any(weight_sum <= 0.0):
            raise RuntimeError("sliding windows did not cover the complete contour")
        role_logits = role_sum / weight_sum[:, None]
        transition_logits = transition_sum / weight_sum
        role_probabilities = _softmax(
            role_logits, self.config.role_temperature
        ).astype(np.float32)
        transition_probabilities = _sigmoid(
            transition_logits, self.config.transition_temperature
        ).astype(np.float32)
        fragments = propose_fragments(
            role_probabilities, transition_probabilities, self.fragment_config
        )
        return {
            "role_logits": role_logits.astype(np.float32),
            "role_probabilities": role_probabilities,
            "transition_logits": transition_logits.astype(np.float32),
            "transition_probabilities": transition_probabilities,
            "candidate_fragments": fragments,
            "inference_config": asdict(self.config),
            "fragment_config": self.fragment_config.public_dict(),
        }


class FullContourPredictor(ContourPredictorBase):
    """PyTorch contour predictor retained for training and research parity."""

    def __init__(
        self,
        model: Any,
        sampler: BoundaryWindowSampler,
        config: ContourInferenceConfig | None = None,
        fragment_config: FragmentProposalConfig | None = None,
        *,
        device: str = "cpu",
    ):
        if torch is None:
            raise RuntimeError("full-contour inference requires PyTorch")
        self.model = model
        self.device = device
        super().__init__(sampler, model.config, config, fragment_config)

    def _predict_batch(
        self, samples: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = torch.from_numpy(
            np.stack([sample["geometry"] for sample in samples])
        ).to(self.device)
        raster = None
        if self.model_config.use_raster:
            raster = torch.from_numpy(
                np.stack([sample["raster"] for sample in samples])
            ).to(self.device)
        global_geometry = None
        global_mask = None
        if self.model_config.use_global_context:
            global_geometry = torch.from_numpy(
                np.stack([sample["global_geometry"] for sample in samples])
            ).to(self.device)
            global_mask = torch.from_numpy(
                np.stack([sample["global_mask"] for sample in samples])
            ).to(self.device)
        with torch.no_grad():
            outputs = self.model(geometry, raster, global_geometry, global_mask)
        return (
            outputs["role_logits"].detach().cpu().numpy(),
            outputs["transition_logits"].detach().cpu().numpy(),
        )

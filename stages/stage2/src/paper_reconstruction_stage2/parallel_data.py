from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .training_data import (
    SAMPLING_CATEGORY_NAMES,
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
    normalized_sampling_weights,
)


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # PyTorch is optional outside model training.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


class DeterministicBoundaryWindowDataset(Dataset):
    """Random-looking windows whose sample index completely determines the draw.

    This is deliberately map-style instead of iterable. A worker may prefetch any
    number of future samples, but a stopped run resumes at the same sample index
    and therefore receives the exact same window. Worker count and prefetch depth
    do not alter the optimization data stream.
    """

    def __init__(
        self,
        boundary_dataset_dir: Path,
        observation_dataset_dir: Path,
        *,
        window_config: BoundaryWindowConfig,
        split: str,
        sample_count: int,
        seed: int,
        epoch: int,
        include_raster: bool,
        include_global_context: bool = False,
        global_context_points: int = 256,
        global_mask_size: int = 64,
        role_sampling_weights: dict[str, float] | None = None,
        natural: bool = False,
        stream_id: int = 0,
    ):
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        self.boundary_dataset_dir = Path(boundary_dataset_dir)
        self.observation_dataset_dir = Path(observation_dataset_dir)
        self.window_config_dict = asdict(window_config)
        self.split = split
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.include_raster = bool(include_raster)
        self.include_global_context = bool(include_global_context)
        self.global_context_points = int(global_context_points)
        self.global_mask_size = int(global_mask_size)
        self.natural = bool(natural)
        self.stream_id = int(stream_id)
        weights = normalized_sampling_weights(role_sampling_weights)
        self.role_probabilities = np.asarray(
            [weights[name] for name in SAMPLING_CATEGORY_NAMES],
            dtype=np.float64,
        )
        self._index: BoundaryTrainingIndex | None = None
        self._sampler: BoundaryWindowSampler | None = None

    def __len__(self) -> int:
        return self.sample_count

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Each Windows worker opens its own bounded caches after spawning.
        state["_index"] = None
        state["_sampler"] = None
        return state

    def _get_sampler(self) -> BoundaryWindowSampler:
        if self._sampler is None:
            self._index = BoundaryTrainingIndex(
                self.boundary_dataset_dir, self.observation_dataset_dir
            )
            self._sampler = BoundaryWindowSampler(
                self._index, BoundaryWindowConfig(**self.window_config_dict)
            )
        return self._sampler

    def __getitem__(self, sample_index: int) -> dict[str, np.ndarray]:
        sample_index = int(sample_index)
        if not 0 <= sample_index < self.sample_count:
            raise IndexError(sample_index)
        # Independent counter-based seeds make scheduling and prefetch invisible.
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch, self.stream_id, sample_index]
            )
        )
        role = None
        if not self.natural:
            role = SAMPLING_CATEGORY_NAMES[
                int(
                    rng.choice(
                        len(SAMPLING_CATEGORY_NAMES), p=self.role_probabilities
                    )
                )
            ]
        sample = self._get_sampler().sample(
            rng,
            split=self.split,
            role=role,
            include_raster=self.include_raster,
            include_global_context=self.include_global_context,
            global_context_points=self.global_context_points,
            global_mask_size=self.global_mask_size,
            natural=self.natural,
        )
        result = {
            "geometry": sample["geometry"],
            "targets": sample["targets"],
            "label_confidence": sample["label_confidence"],
        }
        if self.include_raster:
            result["raster"] = sample["raster"]
        if self.include_global_context:
            result["global_geometry"] = sample["global_geometry"]
            result["global_mask"] = sample["global_mask"]
        return result


def make_boundary_window_loader(
    *,
    boundary_dataset_dir: Path,
    observation_dataset_dir: Path,
    window_config: BoundaryWindowConfig,
    split: str,
    sample_count: int,
    seed: int,
    epoch: int,
    include_raster: bool,
    include_global_context: bool = False,
    global_context_points: int = 256,
    global_mask_size: int = 64,
    role_sampling_weights: dict[str, float],
    natural: bool,
    stream_id: int,
    batch_size: int,
    start_sample: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> Any:
    if torch is None:
        raise RuntimeError("parallel boundary loading requires PyTorch")
    if not 0 <= start_sample <= sample_count:
        raise ValueError("start_sample must be inside the dataset")
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be non-negative and prefetch_factor positive")
    dataset = DeterministicBoundaryWindowDataset(
        boundary_dataset_dir,
        observation_dataset_dir,
        window_config=window_config,
        split=split,
        sample_count=sample_count,
        seed=seed,
        epoch=epoch,
        include_raster=include_raster,
        include_global_context=include_global_context,
        global_context_points=global_context_points,
        global_mask_size=global_mask_size,
        role_sampling_weights=role_sampling_weights,
        natural=natural,
        stream_id=stream_id,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": range(start_sample, sample_count),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = False
    return torch.utils.data.DataLoader(**kwargs)

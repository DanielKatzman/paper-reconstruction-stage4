from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .losses import BoundaryRoleLoss
from .model import BoundaryRoleModelConfig, BoundaryRoleNetV1
from .model_candidates import candidate_model_config
from .parallel_data import make_boundary_window_loader
from .training_data import (
    DEFAULT_ROLE_SAMPLING_WEIGHTS,
    SAMPLING_CATEGORY_NAMES,
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
)


try:
    import torch
except ImportError:
    torch = None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _effective_role_weights(index: BoundaryTrainingIndex) -> dict[str, float]:
    values = {
        name: DEFAULT_ROLE_SAMPLING_WEIGHTS[name]
        if (
            index.by_split_transition.get("train")
            if name == "transition"
            else index.by_split_role.get(("train", name))
        )
        else 0.0
        for name in SAMPLING_CATEGORY_NAMES
    }
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("training split contains no labelled boundary roles")
    return {name: value / total for name, value in values.items()}


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def select_profile_recommendations(
    records: list[dict[str, Any]],
    model_names: list[str],
    *,
    memory_headroom: float,
    total_device_mb: float | None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Choose one fair common recipe plus the fastest recipe for each model."""

    usable: list[dict[str, Any]] = []
    for record in records:
        if record.get("phase") != "model_matrix" or record.get("status") != "ok":
            continue
        peak = record.get("peak_allocated_mb")
        if (
            peak is not None
            and total_device_mb is not None
            and float(peak) > total_device_mb * memory_headroom
        ):
            continue
        usable.append(record)

    per_model: dict[str, dict[str, Any]] = {}
    for name in model_names:
        candidates = [record for record in usable if record["model"] == name]
        if candidates:
            best = max(candidates, key=lambda item: item["windows_per_second"])
            per_model[name] = {
                key: best[key]
                for key in (
                    "batch_size",
                    "workers",
                    "precision",
                    "windows_per_second",
                    "peak_allocated_mb",
                )
            }

    grouped: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = {}
    for record in usable:
        key = (record["batch_size"], record["workers"], record["precision"])
        grouped.setdefault(key, {})[record["model"]] = record
    common: list[dict[str, Any]] = []
    for (batch_size, workers, precision), by_model in grouped.items():
        if any(name not in by_model for name in model_names):
            continue
        rates = [float(by_model[name]["windows_per_second"]) for name in model_names]
        geometric_rate = math.exp(
            sum(math.log(max(rate, 1e-9)) for rate in rates) / len(rates)
        )
        common.append(
            {
                "batch_size": batch_size,
                "workers": workers,
                "precision": precision,
                "geometric_mean_windows_per_second": geometric_rate,
                "slowest_model_windows_per_second": min(rates),
                "maximum_peak_allocated_mb": max(
                    float(by_model[name].get("peak_allocated_mb") or 0.0)
                    for name in model_names
                ),
            }
        )
    recommended = (
        max(common, key=lambda item: item["geometric_mean_windows_per_second"])
        if common
        else None
    )
    return recommended, per_model


def _profile_one(
    *,
    boundary_dataset_dir: Path,
    observation_dataset_dir: Path,
    settings: BoundaryWindowConfig,
    statistics: dict[str, Any],
    role_weights: dict[str, float],
    model_name: str,
    model_config: BoundaryRoleModelConfig,
    batch_size: int,
    workers: int,
    precision: str,
    batches: int,
    warmup_batches: int,
    device: str,
    phase: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": phase,
        "model": model_name,
        "batch_size": batch_size,
        "workers": workers,
        "precision": precision,
    }
    loader = model = criterion = optimizer = scaler = iterator = None
    try:
        loader = make_boundary_window_loader(
            boundary_dataset_dir=boundary_dataset_dir,
            observation_dataset_dir=observation_dataset_dir,
            window_config=settings,
            split="train",
            sample_count=(batches + warmup_batches) * batch_size,
            seed=20261006,
            epoch=0,
            include_raster=model_config.use_raster,
            include_global_context=model_config.use_global_context,
            global_context_points=model_config.global_context_points,
            global_mask_size=model_config.global_mask_size,
            role_sampling_weights=role_weights,
            natural=False,
            stream_id=701,
            batch_size=batch_size,
            num_workers=workers,
            prefetch_factor=2,
            pin_memory=device.startswith("cuda"),
        )
        model = BoundaryRoleNetV1(
            model_config,
            geometry_mean=statistics["geometry"]["mean"],
            geometry_std=statistics["geometry"]["std"],
        ).to(device)
        criterion = BoundaryRoleLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        amp_enabled = precision == "amp"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        iterator = iter(loader)
        fetch_times: list[float] = []
        compute_times: list[float] = []
        step_times: list[float] = []
        for step in range(warmup_batches + batches):
            step_started = time.perf_counter()
            cpu_batch = next(iterator)
            fetch_elapsed = time.perf_counter() - step_started
            compute_started = time.perf_counter()
            geometry = cpu_batch["geometry"].to(device, non_blocking=True)
            raster = (
                cpu_batch["raster"].to(device, non_blocking=True)
                if model_config.use_raster
                else None
            )
            targets = cpu_batch["targets"].to(device, non_blocking=True)
            confidence = cpu_batch["label_confidence"].to(device, non_blocking=True)
            global_geometry = (
                cpu_batch["global_geometry"].to(device, non_blocking=True)
                if model_config.use_global_context
                else None
            )
            global_mask = (
                cpu_batch["global_mask"].to(device, non_blocking=True)
                if model_config.use_global_context
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(geometry, raster, global_geometry, global_mask)
                loss = criterion(outputs, targets, confidence)["total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            compute_elapsed = time.perf_counter() - compute_started
            if step >= warmup_batches:
                fetch_times.append(fetch_elapsed)
                compute_times.append(compute_elapsed)
                step_times.append(time.perf_counter() - step_started)
        mean_step = float(np.mean(step_times))
        record.update(
            {
                "status": "ok",
                "mean_step_seconds": mean_step,
                "mean_fetch_seconds": float(np.mean(fetch_times)),
                "mean_compute_seconds": float(np.mean(compute_times)),
                "windows_per_second": batch_size / max(mean_step, 1e-9),
                "parameter_count": model.parameter_count,
                "peak_allocated_mb": float(
                    torch.cuda.max_memory_allocated() / (1024**2)
                )
                if device.startswith("cuda")
                else None,
            }
        )
    except (RuntimeError, StopIteration) as error:
        record.update(
            {
                "status": "oom" if _is_oom(error) else "error",
                "error": str(error).splitlines()[0][:500],
            }
        )
    finally:
        del iterator, loader, model, criterion, optimizer, scaler
        gc.collect()
        if torch is not None and device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(record, flush=True)
    return record


def profile_training_pipeline(
    *,
    boundary_dataset_dir: Path,
    observation_dataset_dir: Path,
    statistics_path: Path,
    output_path: Path,
    batch_sizes: list[int],
    worker_counts: list[int],
    precisions: list[str],
    model_names: list[str],
    batches: int,
    warmup_batches: int,
    device: str,
    memory_headroom: float = 0.82,
    window_config: BoundaryWindowConfig | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("training profiling requires PyTorch")
    if not batch_sizes or not worker_counts or not precisions or not model_names:
        raise ValueError("batch sizes, workers, precisions, and models cannot be empty")
    if not 0.1 <= memory_headroom <= 1.0:
        raise ValueError("memory_headroom must be between 0.1 and 1.0")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    index = BoundaryTrainingIndex(boundary_dataset_dir, observation_dataset_dir)
    settings = window_config or BoundaryWindowConfig()
    role_weights = _effective_role_weights(index)
    results: list[dict[str, Any]] = []

    reference_batch = 16 if 16 in batch_sizes else sorted(batch_sizes)[0]
    worker_sweep_models = [
        name
        for name in ("base_tcn", "global_context_hybrid")
        if name in model_names
    ] or [model_names[0]]
    for workers in worker_counts:
        for model_name in worker_sweep_models:
            results.append(
                _profile_one(
                    boundary_dataset_dir=boundary_dataset_dir,
                    observation_dataset_dir=observation_dataset_dir,
                    settings=settings,
                    statistics=statistics,
                    role_weights=role_weights,
                    model_name=model_name,
                    model_config=candidate_model_config(model_name),
                    batch_size=reference_batch,
                    workers=workers,
                    precision="fp32",
                    batches=batches,
                    warmup_batches=warmup_batches,
                    device=device,
                    phase="worker_sweep",
                )
            )
    worker_groups: list[dict[str, Any]] = []
    for workers in worker_counts:
        matching = [
            record
            for record in results
            if record["phase"] == "worker_sweep"
            and record["workers"] == workers
            and record["status"] == "ok"
        ]
        by_model = {record["model"]: record for record in matching}
        if all(name in by_model for name in worker_sweep_models):
            rates = [
                float(by_model[name]["windows_per_second"])
                for name in worker_sweep_models
            ]
            worker_groups.append(
                {
                    "workers": workers,
                    "geometric_mean_windows_per_second": math.exp(
                        sum(math.log(max(rate, 1e-9)) for rate in rates)
                        / len(rates)
                    ),
                }
            )
    if not worker_groups:
        report = {
            "schema_version": "stage2.boundary-role-profile.v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "status": "worker_sweep_failed",
            "results": results,
        }
        _write_json_atomic(output_path, report)
        raise RuntimeError("all worker-sweep configurations failed; inspect the report")
    selected_workers = int(
        max(
            worker_groups,
            key=lambda item: item["geometric_mean_windows_per_second"],
        )["workers"]
    )

    for model_name in model_names:
        model_config = candidate_model_config(model_name)
        for precision in precisions:
            if precision == "amp" and not device.startswith("cuda"):
                continue
            oom_seen = False
            for batch_size in sorted(set(batch_sizes)):
                if oom_seen:
                    record = {
                        "phase": "model_matrix",
                        "model": model_name,
                        "batch_size": batch_size,
                        "workers": selected_workers,
                        "precision": precision,
                        "status": "skipped_after_oom",
                    }
                else:
                    record = _profile_one(
                        boundary_dataset_dir=boundary_dataset_dir,
                        observation_dataset_dir=observation_dataset_dir,
                        settings=settings,
                        statistics=statistics,
                        role_weights=role_weights,
                        model_name=model_name,
                        model_config=model_config,
                        batch_size=batch_size,
                        workers=selected_workers,
                        precision=precision,
                        batches=batches,
                        warmup_batches=warmup_batches,
                        device=device,
                        phase="model_matrix",
                    )
                    oom_seen = record["status"] == "oom"
                results.append(record)

    total_device_mb = (
        float(torch.cuda.get_device_properties(0).total_memory / (1024**2))
        if device.startswith("cuda")
        else None
    )
    common, per_model = select_profile_recommendations(
        results,
        model_names,
        memory_headroom=memory_headroom,
        total_device_mb=total_device_mb,
    )
    report = {
        "schema_version": "stage2.boundary-role-profile.v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "total_device_mb": total_device_mb,
        "memory_headroom": memory_headroom,
        "batches": batches,
        "warmup_batches": warmup_batches,
        "models": model_names,
        "worker_sweep_models": worker_sweep_models,
        "selected_workers": selected_workers,
        "recommended_common_configuration": common,
        "per_model_fastest_configuration": per_model,
        "results": results,
    }
    _write_json_atomic(output_path, report)
    return report

from __future__ import annotations

import hashlib
import json
import math
import random
import signal
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .losses import BoundaryRoleLoss, BoundaryRoleLossConfig, transition_targets
from .model import BoundaryRoleModelConfig, BoundaryRoleNetV1
from .model_metrics import StreamingBoundaryMetrics, selection_score
from .parallel_data import make_boundary_window_loader
from .pipeline import PipelineLock
from .training_data import (
    GEOMETRY_FEATURE_NAMES,
    SAMPLING_CATEGORY_NAMES,
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
    normalized_sampling_weights,
)


try:
    import torch
except ImportError:
    torch = None


CHECKPOINT_SCHEMA_VERSION = "stage2.boundary-role-checkpoint.v3"
TRAINING_STATE_SCHEMA_VERSION = "stage2.boundary-role-training-state.v3"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BoundaryTrainingConfig:
    boundary_dataset_dir: Path
    observation_dataset_dir: Path
    statistics_path: Path
    output_dir: Path
    seed: int = 20261006
    epochs: int = 20
    samples_per_epoch: int = 50_000
    validation_samples: int = 10_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    checkpoint_every_steps: int = 100
    early_stopping_patience: int = 4
    precision: str = "fp32"
    device: str = "auto"
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    resume: bool = False
    maximum_steps: int | None = None
    maximum_completed_epochs: int | None = None

    def validate(self) -> None:
        if self.epochs < 1 or self.samples_per_epoch < 1:
            raise ValueError("epochs and samples_per_epoch must be positive")
        if self.validation_samples < 1 or self.batch_size < 1:
            raise ValueError("validation_samples and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be positive")
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be 'fp32' or 'amp'")
        if self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("num_workers must be non-negative and prefetch_factor positive")
        if self.maximum_completed_epochs is not None and not (
            1 <= self.maximum_completed_epochs <= self.epochs
        ):
            raise ValueError("maximum_completed_epochs must be between 1 and epochs")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "boundary_dataset_dir",
            "observation_dataset_dir",
            "statistics_path",
            "output_dir",
        ):
            result[key] = str(result[key])
        result.pop("resume", None)
        # This is an execution limiter used for smoke tests and controlled stops,
        # not part of the optimization recipe. Excluding it lets a deliberately
        # shortened run resume with the same reproducible training configuration.
        result.pop("maximum_steps", None)
        result.pop("maximum_completed_epochs", None)
        # Loader concurrency changes throughput, not the counter-seeded sample
        # sequence, and is therefore safe to tune between resumed invocations.
        result.pop("num_workers", None)
        result.pop("prefetch_factor", None)
        result.pop("pin_memory", None)
        return result


class GracefulStop(AbstractContextManager["GracefulStop"]):
    def __init__(self):
        self.requested = False
        self._previous: Any = None

    def __enter__(self) -> "GracefulStop":
        self._previous = signal.getsignal(signal.SIGINT)

        def handler(signum: int, frame: Any) -> None:
            del signum, frame
            if not self.requested:
                print(
                    "\nSafe stop requested; finishing the current batch and saving "
                    "an exact checkpoint...",
                    flush=True,
                )
            self.requested = True

        signal.signal(signal.SIGINT, handler)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.signal(signal.SIGINT, self._previous)


def _resolve_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("boundary-role training requires PyTorch")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte_rng_state(value: Any) -> Any:
    """Return an RNG-state tensor in the format required by PyTorch.

    Loading a checkpoint with ``map_location="cuda"`` also maps the saved CPU
    RNG state to CUDA. Both ``torch.set_rng_state`` and the CUDA RNG restore
    helpers require CPU ByteTensors, so normalize every saved state before it
    is handed back to PyTorch.
    """
    if not torch.is_tensor(value):
        raise TypeError("checkpoint RNG state is not a tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [_cpu_byte_rng_state(value) for value in state["torch_cuda"]]
        )


def _batch_from_sampler(
    sampler: BoundaryWindowSampler,
    rng: np.random.Generator,
    *,
    split: str,
    batch_size: int,
    natural: bool,
    include_raster: bool,
    role_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    samples: list[dict[str, Any]] = []
    for _ in range(batch_size):
        role = None
        if not natural:
            role = SAMPLING_CATEGORY_NAMES[
                int(
                    rng.choice(
                        len(SAMPLING_CATEGORY_NAMES), p=role_probabilities
                    )
                )
            ]
        samples.append(
            sampler.sample(
                rng,
                split=split,
                role=role,
                include_raster=include_raster,
                natural=natural,
            )
        )
    result = {
        "geometry": np.stack([sample["geometry"] for sample in samples]),
        "targets": np.stack([sample["targets"] for sample in samples]),
        "label_confidence": np.stack(
            [sample["label_confidence"] for sample in samples]
        ),
    }
    if include_raster:
        result["raster"] = np.stack([sample["raster"] for sample in samples])
    return result


class BoundaryRoleTrainer:
    def __init__(
        self,
        config: BoundaryTrainingConfig,
        *,
        window_config: BoundaryWindowConfig | None = None,
        model_config: BoundaryRoleModelConfig | None = None,
        loss_config: BoundaryRoleLossConfig | None = None,
        role_sampling_weights: dict[str, float] | None = None,
    ):
        if torch is None:
            raise RuntimeError("BoundaryRoleTrainer requires PyTorch")
        config.validate()
        # Seed before model construction so candidate initialization is itself
        # reproducible, not merely the later minibatch sequence.
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.config = config
        self.window_config = window_config or BoundaryWindowConfig()
        self.window_config.validate()
        self.model_config = model_config or BoundaryRoleModelConfig()
        self.model_config.validate()
        self.loss_config = loss_config or BoundaryRoleLossConfig()
        self.index = BoundaryTrainingIndex(
            config.boundary_dataset_dir, config.observation_dataset_dir
        )
        for required_split in ("train", "validation"):
            if required_split not in self.index.by_split:
                raise ValueError(
                    f"boundary dataset has no {required_split!r} observations"
                )
        weights = normalized_sampling_weights(role_sampling_weights)
        effective_weights = {
            name: float(weights[name])
            if (
                self.index.by_split_transition.get("train")
                if name == "transition"
                else self.index.by_split_role.get(("train", name))
            )
            else 0.0
            for name in SAMPLING_CATEGORY_NAMES
        }
        total_weight = float(sum(effective_weights.values()))
        if total_weight <= 0.0:
            raise ValueError("role sampling weights must be non-negative with positive sum")
        self.role_sampling_weights = {
            name: float(effective_weights[name] / total_weight)
            for name in SAMPLING_CATEGORY_NAMES
        }
        self.role_probabilities = np.asarray(
            [self.role_sampling_weights[name] for name in SAMPLING_CATEGORY_NAMES],
            dtype=np.float64,
        )
        self.sampler = BoundaryWindowSampler(self.index, self.window_config)
        self.statistics = json.loads(config.statistics_path.read_text(encoding="utf-8"))
        if self.statistics.get("status") != "complete":
            raise RuntimeError("training statistics are not complete")
        if self.statistics.get("geometry", {}).get("feature_names") != list(
            GEOMETRY_FEATURE_NAMES
        ):
            raise RuntimeError(
                "training statistics use incompatible geometry features; recompute "
                "them with this package version"
            )
        if (
            self.statistics.get("boundary_dataset_fingerprint")
            != self.index.boundary_dataset.get("boundary_dataset_fingerprint")
        ):
            raise RuntimeError("statistics belong to a different boundary dataset")
        self.device = _resolve_device(config.device)
        if config.precision == "amp" and not self.device.startswith("cuda"):
            raise ValueError("AMP training currently requires a CUDA device")

        self.model = BoundaryRoleNetV1(
            self.model_config,
            geometry_mean=self.statistics["geometry"]["mean"],
            geometry_std=self.statistics["geometry"]["std"],
        ).to(self.device)
        self.loss = BoundaryRoleLoss(self.loss_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.batches_per_epoch = math.ceil(config.samples_per_epoch / config.batch_size)
        total_steps = self.batches_per_epoch * config.epochs
        warmup_steps = max(1, round(total_steps * config.warmup_fraction))

        def learning_rate_factor(step: int) -> float:
            if step < warmup_steps:
                return max((step + 1) / warmup_steps, 1e-3)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, learning_rate_factor
        )
        self.amp_enabled = config.precision == "amp"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.config_payload = {
            "training": config.public_dict(),
            "window": asdict(self.window_config),
            "model": self.model_config.public_dict(),
            "loss": asdict(self.loss_config),
            "role_sampling_weights": self.role_sampling_weights,
            "data_schedule": "counter_seed_transition_v2",
        }
        self.config_hash = _sha256(self.config_payload)
        self.last_path = config.output_dir / "last.pt"
        self.best_path = config.output_dir / "best.pt"
        self.state_path = config.output_dir / "training_state.json"
        self.history_path = config.output_dir / "history.json"

    def _prepare_output(self) -> None:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"training output is not empty: {output}; pass --resume or choose "
                "a new output directory"
            )
        output.mkdir(parents=True, exist_ok=True)

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        next_batch: int,
        global_step: int,
        train_rng: np.random.Generator,
        best_score: float,
        best_epoch: int | None,
        epochs_without_improvement: int,
        history: list[dict[str, Any]],
        epoch_running_loss: float,
        epoch_processed_batches: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "package_version": __version__,
            "created_utc": _utc_now(),
            "config": self.config_payload,
            "config_hash": self.config_hash,
            "boundary_dataset_fingerprint": self.index.boundary_dataset[
                "boundary_dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": self.index.observation_dataset[
                "observation_dataset_fingerprint"
            ],
            "statistics_schema_version": self.statistics["schema_version"],
            "model_config": self.model_config.public_dict(),
            "geometry_normalization": {
                "feature_names": self.statistics["geometry"]["feature_names"],
                "mean": self.statistics["geometry"]["mean"],
                "std": self.statistics["geometry"]["std"],
            },
            "window_config": asdict(self.window_config),
            "loss_config": asdict(self.loss_config),
            "role_sampling_weights": self.role_sampling_weights,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "train_rng_state": train_rng.bit_generator.state,
            "process_rng_state": _rng_state(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "epoch_running_loss": epoch_running_loss,
            "epoch_processed_batches": epoch_processed_batches,
            "calibration": {
                "role_temperature": 1.0,
                "transition_temperature": 1.0,
            },
        }

    def _save_state(
        self,
        payload: dict[str, Any],
        *,
        status: str,
        last_metrics: dict[str, Any] | None = None,
    ) -> None:
        _save_torch_atomic(self.last_path, payload)
        state = {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "updated_utc": _utc_now(),
            "status": status,
            "config_hash": self.config_hash,
            "boundary_dataset_fingerprint": payload[
                "boundary_dataset_fingerprint"
            ],
            "epoch": payload["epoch"],
            "next_batch": payload["next_batch"],
            "global_step": payload["global_step"],
            "best_score": payload["best_score"]
            if math.isfinite(float(payload["best_score"]))
            else None,
            "best_epoch": payload["best_epoch"],
            "last_metrics": last_metrics,
            "last_checkpoint": "last.pt",
            "best_checkpoint": "best.pt" if self.best_path.is_file() else None,
        }
        _write_json_atomic(self.state_path, state)
        _write_json_atomic(self.history_path, {"epochs": payload["history"]})

    def _load_resume(
        self,
    ) -> tuple[
        int,
        int,
        int,
        np.random.Generator,
        float,
        int | None,
        int,
        list[dict[str, Any]],
        float,
        int,
    ]:
        if not self.last_path.is_file():
            if self.config.resume and any(self.config.output_dir.iterdir()):
                allowed = {"training.lock"}
                unexpected = [
                    path.name
                    for path in self.config.output_dir.iterdir()
                    if path.name not in allowed
                ]
                if unexpected:
                    raise RuntimeError(
                        "resume requested but last.pt is missing from a non-empty "
                        f"training directory: {unexpected}"
                    )
            rng = np.random.default_rng(self.config.seed)
            return 0, 0, 0, rng, float("-inf"), None, 0, [], 0.0, 0
        if not self.config.resume:
            raise FileExistsError("last.pt exists; pass --resume to continue")
        try:
            checkpoint = torch.load(
                self.last_path, map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch versions before weights_only was introduced.
            checkpoint = torch.load(self.last_path, map_location=self.device)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            if checkpoint.get("schema_version") in {
                "stage2.boundary-role-checkpoint.v1",
                "stage2.boundary-role-checkpoint.v2",
            }:
                raise RuntimeError(
                    "this is a pre-v0.6.2 checkpoint; finish it with its original "
                    "release or start a separate v0.6.2 output because the sampling "
                    "and model inputs changed"
                )
            raise RuntimeError("checkpoint schema is incompatible")
        if checkpoint.get("config_hash") != self.config_hash:
            raise RuntimeError("training configuration changed; refusing unsafe resume")
        if (
            checkpoint.get("boundary_dataset_fingerprint")
            != self.index.boundary_dataset.get("boundary_dataset_fingerprint")
        ):
            raise RuntimeError("checkpoint belongs to a different dataset")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        _restore_rng_state(checkpoint["process_rng_state"])
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["train_rng_state"]
        return (
            int(checkpoint["epoch"]),
            int(checkpoint["next_batch"]),
            int(checkpoint["global_step"]),
            rng,
            float(checkpoint["best_score"]),
            checkpoint.get("best_epoch"),
            int(checkpoint.get("epochs_without_improvement", 0)),
            list(checkpoint.get("history", [])),
            float(checkpoint.get("epoch_running_loss", 0.0)),
            int(checkpoint.get("epoch_processed_batches", 0)),
        )

    def _tensor_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: torch.as_tensor(value).to(self.device, non_blocking=True)
            for key, value in batch.items()
        }
        return result

    def _loader(
        self,
        *,
        split: str,
        sample_count: int,
        epoch: int,
        natural: bool,
        stream_id: int,
        start_sample: int = 0,
    ) -> Any:
        return make_boundary_window_loader(
            boundary_dataset_dir=self.config.boundary_dataset_dir,
            observation_dataset_dir=self.config.observation_dataset_dir,
            window_config=self.window_config,
            split=split,
            sample_count=sample_count,
            seed=self.config.seed,
            epoch=epoch,
            include_raster=self.model_config.use_raster,
            include_global_context=self.model_config.use_global_context,
            global_context_points=self.model_config.global_context_points,
            global_mask_size=self.model_config.global_mask_size,
            role_sampling_weights=self.role_sampling_weights,
            natural=natural,
            stream_id=stream_id,
            batch_size=self.config.batch_size,
            start_sample=start_sample,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory and self.device.startswith("cuda"),
        )

    def _validate(self, epoch: int) -> dict[str, Any]:
        self.model.eval()
        metrics = StreamingBoundaryMetrics()
        losses: list[float] = []
        loader = self._loader(
            split="validation",
            sample_count=self.config.validation_samples,
            epoch=epoch,
            natural=True,
            stream_id=991,
        )
        with torch.no_grad():
            for cpu_batch in loader:
                target_values = cpu_batch["targets"].numpy()
                batch = self._tensor_batch(cpu_batch)
                outputs = self.model(
                    batch["geometry"],
                    batch.get("raster"),
                    batch.get("global_geometry"),
                    batch.get("global_mask"),
                )
                loss_values = self.loss(
                    outputs, batch["targets"], batch["label_confidence"]
                )
                losses.append(float(loss_values["total"].detach().cpu()))
                role_probabilities = torch.softmax(
                    outputs["role_logits"], dim=-1
                ).cpu().numpy()
                transition_probabilities = torch.sigmoid(
                    outputs["transition_logits"]
                ).cpu().numpy()
                transitions = transition_targets(batch["targets"]).cpu().numpy()
                metrics.update(
                    role_probabilities,
                    target_values,
                    transition_probabilities,
                    transitions,
                )
        values = metrics.compute()
        values["loss"] = float(np.mean(losses))
        values["selection_score"] = selection_score(values)
        return values

    def train(self) -> dict[str, Any]:
        self._prepare_output()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with PipelineLock(self.config.output_dir / "training.lock"), GracefulStop() as stop:
            (
                start_epoch,
                start_batch,
                global_step,
                train_rng,
                best_score,
                best_epoch,
                epochs_without_improvement,
                history,
                resume_running_loss,
                resume_processed_batches,
            ) = self._load_resume()
            if start_epoch >= self.config.epochs:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                self.config.maximum_completed_epochs is not None
                and len(history) >= self.config.maximum_completed_epochs
            ):
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                self.config.maximum_steps is not None
                and global_step >= self.config.maximum_steps
            ):
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            started = time.perf_counter()
            last_metrics: dict[str, Any] | None = None
            for epoch in range(start_epoch, self.config.epochs):
                self.model.train()
                running_loss = resume_running_loss if epoch == start_epoch else 0.0
                processed_batches = (
                    resume_processed_batches if epoch == start_epoch else 0
                )
                batch_begin = start_batch if epoch == start_epoch else 0
                loader = self._loader(
                    split="train",
                    sample_count=self.config.samples_per_epoch,
                    epoch=epoch,
                    natural=False,
                    stream_id=17,
                    start_sample=batch_begin * self.config.batch_size,
                )
                for batch_index, cpu_batch in enumerate(loader, start=batch_begin):
                    batch = self._tensor_batch(cpu_batch)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                        outputs = self.model(
                            batch["geometry"],
                            batch.get("raster"),
                            batch.get("global_geometry"),
                            batch.get("global_mask"),
                        )
                        loss_values = self.loss(
                            outputs, batch["targets"], batch["label_confidence"]
                        )
                        total_loss = loss_values["total"]
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError(
                            f"non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}"
                        )
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    global_step += 1
                    running_loss += float(total_loss.detach().cpu())
                    processed_batches += 1
                    next_epoch = epoch
                    next_batch = batch_index + 1
                    should_checkpoint = (
                        global_step % self.config.checkpoint_every_steps == 0
                        or stop.requested
                        or (
                            self.config.maximum_steps is not None
                            and global_step >= self.config.maximum_steps
                        )
                    )
                    if should_checkpoint:
                        payload = self._checkpoint_payload(
                            epoch=next_epoch,
                            next_batch=next_batch,
                            global_step=global_step,
                            train_rng=train_rng,
                            best_score=best_score,
                            best_epoch=best_epoch,
                            epochs_without_improvement=epochs_without_improvement,
                            history=history,
                            epoch_running_loss=running_loss,
                            epoch_processed_batches=processed_batches,
                        )
                        status = (
                            "stopped"
                            if stop.requested
                            or (
                                self.config.maximum_steps is not None
                                and global_step >= self.config.maximum_steps
                            )
                            else "running"
                        )
                        self._save_state(payload, status=status, last_metrics=last_metrics)
                    if stop.requested or (
                        self.config.maximum_steps is not None
                        and global_step >= self.config.maximum_steps
                    ):
                        return json.loads(self.state_path.read_text(encoding="utf-8"))
                    if global_step % 25 == 0:
                        elapsed = time.perf_counter() - started
                        rate = global_step / max(elapsed, 1e-9)
                        print(
                            f"epoch {epoch + 1}/{self.config.epochs}, batch "
                            f"{batch_index + 1}/{self.batches_per_epoch}, "
                            f"loss={running_loss / max(processed_batches, 1):.5f}, "
                            f"{rate:.2f} steps/s",
                            flush=True,
                        )

                last_metrics = self._validate(epoch)
                score = float(last_metrics["selection_score"])
                epoch_record = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "training_loss": running_loss / max(processed_batches, 1),
                    "validation": last_metrics,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                }
                history.append(epoch_record)
                improved = score > best_score
                if improved:
                    best_score = score
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                payload = self._checkpoint_payload(
                    epoch=epoch + 1,
                    next_batch=0,
                    global_step=global_step,
                    train_rng=train_rng,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                    epoch_running_loss=0.0,
                    epoch_processed_batches=0,
                )
                self._save_state(payload, status="running", last_metrics=last_metrics)
                if improved:
                    _save_torch_atomic(self.best_path, payload)
                    self._save_state(payload, status="running", last_metrics=last_metrics)
                print(
                    f"epoch {epoch + 1}: validation score={score:.6f}, "
                    f"seam_f1={last_metrics['roles']['seam']['f1']:.6f}, "
                    f"border_f1={last_metrics['roles']['page_border']['f1']:.6f}"
                    f"{' NEW BEST' if improved else ''}",
                    flush=True,
                )
                start_batch = 0
                if (
                    self.config.maximum_completed_epochs is not None
                    and len(history) >= self.config.maximum_completed_epochs
                ):
                    self._save_state(
                        payload, status="stopped", last_metrics=last_metrics
                    )
                    return json.loads(self.state_path.read_text(encoding="utf-8"))
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break

            final_payload = self._checkpoint_payload(
                epoch=min(self.config.epochs, len(history)),
                next_batch=0,
                global_step=global_step,
                train_rng=train_rng,
                best_score=best_score,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                epoch_running_loss=0.0,
                epoch_processed_batches=0,
            )
            self._save_state(final_payload, status="complete", last_metrics=last_metrics)
            return json.loads(self.state_path.read_text(encoding="utf-8"))

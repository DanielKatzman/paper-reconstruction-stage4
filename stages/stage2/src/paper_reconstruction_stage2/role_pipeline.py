from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import CalibrationConfig, calibrate_checkpoint
from .evaluation import evaluate_full_contours
from .fragments import FragmentProposalConfig
from .inference import (
    ContourInferenceConfig,
    FullContourPredictor,
    checkpoint_sha256,
    load_boundary_role_model,
)
from .losses import BoundaryRoleLossConfig
from .model import BoundaryRoleModelConfig
from .pipeline import PipelineLock
from .statistics import StatisticsConfig, TrainingStatisticsCompiler
from .training import BoundaryRoleTrainer, BoundaryTrainingConfig
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


ROLE_PIPELINE_SCHEMA_VERSION = "stage2.boundary-role-pipeline.v2"
ROLE_PIPELINE_STAGES = ("statistics", "training", "calibration", "validation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(slots=True)
class BoundaryRolePipelineConfig:
    boundary_dataset_dir: Path
    observation_dataset_dir: Path
    output_dir: Path
    seed: int = 20261006
    epochs: int = 20
    samples_per_epoch: int = 50_000
    validation_samples: int = 10_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    checkpoint_every_steps: int = 100
    early_stopping_patience: int = 4
    precision: str = "fp32"
    device: str = "auto"
    num_workers: int = 2
    prefetch_factor: int = 2
    calibration_maximum_observations: int | None = 512
    evaluation_maximum_observations: int | None = 512
    inference_stride: int = 32
    resume: bool = False
    stop_after: str | None = None
    maximum_steps: int | None = None

    def validate(self) -> None:
        if self.stop_after is not None and self.stop_after not in ROLE_PIPELINE_STAGES:
            raise ValueError(f"stop_after must be one of {ROLE_PIPELINE_STAGES}")

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "boundary_dataset_dir",
            "observation_dataset_dir",
            "output_dir",
        ):
            payload[key] = str(payload[key])
        return payload

    def resume_contract(self) -> dict[str, Any]:
        payload = self.public_dict()
        payload.pop("resume", None)
        payload.pop("stop_after", None)
        payload.pop("maximum_steps", None)
        payload.pop("num_workers", None)
        payload.pop("prefetch_factor", None)
        return payload


class BoundaryRolePipeline:
    def __init__(self, config: BoundaryRolePipelineConfig):
        config.validate()
        self.config = config
        self.state_path = config.output_dir / "boundary_role_pipeline_state.json"
        self.statistics_path = config.output_dir / "training_statistics.json"
        self.training_dir = config.output_dir / "training"
        self.calibrated_path = config.output_dir / "calibrated_best.pt"
        self.calibration_report_path = config.output_dir / "calibration_report.json"
        self.validation_report_path = config.output_dir / "validation_report.json"

    def _state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "schema_version": ROLE_PIPELINE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "status": "running",
            "configuration": self.config.public_dict(),
            "resume_contract": self.config.resume_contract(),
            "stages": {stage: "pending" for stage in ROLE_PIPELINE_STAGES},
        }

    def _write_stage(
        self, state: dict[str, Any], stage: str, stage_status: str, **extra: Any
    ) -> None:
        state["stages"][stage] = stage_status
        state["updated_utc"] = _utc_now()
        state.update(extra)
        _write_json_atomic(self.state_path, state)

    def _stop_after(self, state: dict[str, Any], stage: str) -> bool:
        if self.config.stop_after != stage:
            return False
        state["status"] = "stopped_after_stage"
        state["updated_utc"] = _utc_now()
        _write_json_atomic(self.state_path, state)
        return True

    def _prepare(self) -> dict[str, Any]:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"boundary-role pipeline output is not empty: {output}; pass "
                "--resume or choose a new output directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.config.resume and not self.state_path.is_file():
            unexpected = [
                path.name
                for path in output.iterdir()
                if path.name != "boundary_role_pipeline.lock"
            ]
            if unexpected:
                raise RuntimeError(
                    "resume requested but boundary_role_pipeline_state.json is "
                    f"missing from a non-empty directory: {unexpected}"
                )
        state = self._state()
        if state.get("schema_version") != ROLE_PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("boundary-role pipeline state schema is incompatible")
        if self.state_path.is_file() and not self.config.resume:
            raise FileExistsError("pipeline state exists; pass --resume")
        if self.state_path.is_file() and state.get(
            "resume_contract"
        ) != self.config.resume_contract():
            raise RuntimeError(
                "boundary-role pipeline configuration changed; refusing unsafe resume"
            )
        return state

    @staticmethod
    def _device(requested: str) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("boundary-role pipeline requires PyTorch") from exc
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested

    def run(self) -> dict[str, Any]:
        state = self._prepare()
        with PipelineLock(self.config.output_dir / "boundary_role_pipeline.lock"):
            self._write_stage(state, "statistics", "running", status="running")
            statistics = TrainingStatisticsCompiler(
                StatisticsConfig(
                    boundary_dataset_dir=self.config.boundary_dataset_dir,
                    observation_dataset_dir=self.config.observation_dataset_dir,
                    output_path=self.statistics_path,
                    split="train",
                    resume=self.config.resume and self.statistics_path.exists(),
                )
            ).compile()
            self._write_stage(
                state,
                "statistics",
                "complete",
                statistics_point_count=statistics["accumulators"]["point_count"],
            )
            if self._stop_after(state, "statistics"):
                return state

            self._write_stage(state, "training", "running")
            trainer = BoundaryRoleTrainer(
                BoundaryTrainingConfig(
                    boundary_dataset_dir=self.config.boundary_dataset_dir,
                    observation_dataset_dir=self.config.observation_dataset_dir,
                    statistics_path=self.statistics_path,
                    output_dir=self.training_dir,
                    seed=self.config.seed,
                    epochs=self.config.epochs,
                    samples_per_epoch=self.config.samples_per_epoch,
                    validation_samples=self.config.validation_samples,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    checkpoint_every_steps=self.config.checkpoint_every_steps,
                    early_stopping_patience=self.config.early_stopping_patience,
                    precision=self.config.precision,
                    device=self.config.device,
                    num_workers=self.config.num_workers,
                    prefetch_factor=self.config.prefetch_factor,
                    resume=self.config.resume and self.training_dir.exists(),
                    maximum_steps=self.config.maximum_steps,
                )
            )
            training_state = trainer.train()
            if training_state.get("status") != "complete":
                self._write_stage(
                    state,
                    "training",
                    training_state.get("status", "stopped"),
                    status="stopped",
                    training_state_path="training/training_state.json",
                )
                return state
            self._write_stage(
                state,
                "training",
                "complete",
                training_state_path="training/training_state.json",
                best_epoch=training_state.get("best_epoch"),
            )
            if self._stop_after(state, "training"):
                return state

            best_path = self.training_dir / "best.pt"
            if not best_path.is_file():
                raise RuntimeError("training completed without best.pt")
            self._write_stage(state, "calibration", "running")
            if self.calibrated_path.exists() or self.calibration_report_path.exists():
                if not (
                    self.config.resume
                    and self.calibrated_path.is_file()
                    and self.calibration_report_path.is_file()
                ):
                    raise RuntimeError("incomplete calibration output blocks safe resume")
                calibration_report = json.loads(
                    self.calibration_report_path.read_text(encoding="utf-8")
                )
                if calibration_report.get("input_checkpoint_sha256") != checkpoint_sha256(
                    best_path
                ):
                    raise RuntimeError("calibration belongs to a different best checkpoint")
            else:
                device = self._device(self.config.device)
                model, checkpoint = load_boundary_role_model(best_path, device)
                index = BoundaryTrainingIndex(
                    self.config.boundary_dataset_dir,
                    self.config.observation_dataset_dir,
                )
                window = BoundaryWindowConfig(**checkpoint["window_config"])
                sampler = BoundaryWindowSampler(index, window)
                predictor = FullContourPredictor(
                    model,
                    sampler,
                    ContourInferenceConfig(
                        sequence_length=window.sequence_length,
                        strip_width=window.strip_width,
                        stride=self.config.inference_stride,
                        batch_size=self.config.batch_size,
                    ),
                    FragmentProposalConfig(),
                    device=device,
                )
                calibration_report = calibrate_checkpoint(
                    config=CalibrationConfig(
                        checkpoint_path=best_path,
                        output_checkpoint_path=self.calibrated_path,
                        report_path=self.calibration_report_path,
                        maximum_observations=(
                            self.config.calibration_maximum_observations
                        ),
                        seed=self.config.seed,
                    ),
                    index=index,
                    sampler=sampler,
                    predictor=predictor,
                    checkpoint=checkpoint,
                )
            self._write_stage(
                state,
                "calibration",
                "complete",
                calibrated_checkpoint="calibrated_best.pt",
                calibration_report="calibration_report.json",
            )
            if self._stop_after(state, "calibration"):
                return state

            self._write_stage(state, "validation", "running")
            if self.validation_report_path.exists():
                if not self.config.resume:
                    raise FileExistsError("validation report already exists")
                validation_report = json.loads(
                    self.validation_report_path.read_text(encoding="utf-8")
                )
                if validation_report.get("checkpoint_sha256") != checkpoint_sha256(
                    self.calibrated_path
                ):
                    raise RuntimeError("validation report belongs to another checkpoint")
            else:
                device = self._device(self.config.device)
                model, checkpoint = load_boundary_role_model(
                    self.calibrated_path, device
                )
                index = BoundaryTrainingIndex(
                    self.config.boundary_dataset_dir,
                    self.config.observation_dataset_dir,
                )
                window = BoundaryWindowConfig(**checkpoint["window_config"])
                sampler = BoundaryWindowSampler(index, window)
                calibration = checkpoint["calibration"]
                fragments = FragmentProposalConfig(
                    **checkpoint.get("fragment_config", {})
                )
                predictor = FullContourPredictor(
                    model,
                    sampler,
                    ContourInferenceConfig(
                        sequence_length=window.sequence_length,
                        strip_width=window.strip_width,
                        stride=self.config.inference_stride,
                        batch_size=self.config.batch_size,
                        role_temperature=calibration["role_temperature"],
                        transition_temperature=calibration[
                            "transition_temperature"
                        ],
                    ),
                    fragments,
                    device=device,
                )
                validation_report = evaluate_full_contours(
                    checkpoint_path=self.calibrated_path,
                    split="validation",
                    index=index,
                    sampler=sampler,
                    predictor=predictor,
                    output_path=self.validation_report_path,
                    maximum_observations=self.config.evaluation_maximum_observations,
                    seed=self.config.seed,
                    failure_diagnostics_dir=self.config.output_dir
                    / "validation_failures",
                )
            self._write_stage(
                state,
                "validation",
                "complete",
                status="complete",
                validation_report="validation_report.json",
                validation_selection_score=validation_report["selection_score"],
            )
            return state

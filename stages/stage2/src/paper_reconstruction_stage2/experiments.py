from __future__ import annotations

import hashlib
import json
import shutil
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
from .model_candidates import MODEL_CANDIDATE_NAMES, candidate_model_config
from .pipeline import PipelineLock
from .training import BoundaryRoleTrainer, BoundaryTrainingConfig
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


EXPERIMENT_SCHEMA_VERSION = "stage2.boundary-role-experiments.v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def experiment_selection_score(report: dict[str, Any]) -> float:
    """Combine point accuracy with the fragment recall needed by the solver."""

    fragments = report["fragments"]
    return float(
        0.30 * float(report["selection_score"])
        + 0.20 * float(fragments["recall_at_5_iou_50"])
        + 0.35 * float(fragments["recall_at_5_iou_80"])
        + 0.15 * float(report["metrics"]["transition"]["f1"])
    )


def rank_experiment_reports(
    reports: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ranked = [
        {
            "model": name,
            "experiment_score": experiment_selection_score(report),
            "point_selection_score": float(report["selection_score"]),
            "recall_at_5_iou_50": float(
                report["fragments"]["recall_at_5_iou_50"]
            ),
            "recall_at_5_iou_80": float(
                report["fragments"]["recall_at_5_iou_80"]
            ),
            "transition_f1": float(report["metrics"]["transition"]["f1"]),
        }
        for name, report in reports.items()
    ]
    ranked.sort(key=lambda item: (-item["experiment_score"], item["model"]))
    return ranked


@dataclass(slots=True)
class BoundaryExperimentConfig:
    boundary_dataset_dir: Path
    observation_dataset_dir: Path
    statistics_path: Path
    output_dir: Path
    profile_path: Path | None = None
    candidates: tuple[str, ...] = MODEL_CANDIDATE_NAMES
    seed: int = 20261006
    samples_per_epoch: int = 50_000
    validation_samples: int = 10_000
    screen_epochs: int = 3
    promotion_epochs: int = 10
    final_epochs: int = 20
    promote_count: int = 2
    screen_observations: int = 128
    final_observations: int = 512
    calibration_observations: int = 512
    batch_size: int | None = None
    num_workers: int | None = None
    precision: str | None = None
    prefetch_factor: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    checkpoint_every_steps: int = 100
    inference_stride: int = 32
    device: str = "auto"
    resume: bool = False

    def validate(self) -> None:
        unknown = set(self.candidates) - set(MODEL_CANDIDATE_NAMES)
        if unknown or not self.candidates:
            raise ValueError(f"invalid model candidates: {sorted(unknown)}")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("model candidates must be unique")
        if not 1 <= self.screen_epochs < self.promotion_epochs < self.final_epochs:
            raise ValueError("epochs must satisfy screen < promotion < final")
        if not 1 <= self.promote_count <= len(self.candidates):
            raise ValueError("promote_count must fit inside the candidate set")
        if min(
            self.samples_per_epoch,
            self.validation_samples,
            self.screen_observations,
            self.final_observations,
            self.calibration_observations,
            self.prefetch_factor,
        ) < 1:
            raise ValueError("sample, observation, and prefetch counts must be positive")
        if self.precision is not None and self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be fp32 or amp")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "boundary_dataset_dir",
            "observation_dataset_dir",
            "statistics_path",
            "output_dir",
            "profile_path",
        ):
            if value[key] is not None:
                value[key] = str(value[key])
        value["candidates"] = list(self.candidates)
        return value


class BoundaryRoleExperimentRunner:
    """Resumable successive halving without ever consulting the holdout split."""

    def __init__(self, config: BoundaryExperimentConfig):
        config.validate()
        self.config = config
        self.state_path = config.output_dir / "experiment_state.json"
        self.index = BoundaryTrainingIndex(
            config.boundary_dataset_dir, config.observation_dataset_dir
        )
        self.device = self._resolve_device(config.device)
        self.recipe = self._resolve_recipe()
        self.resume_contract = {
            **config.public_dict(),
            "resume": None,
            "resolved_recipe": self.recipe,
            "boundary_dataset_fingerprint": self.index.boundary_dataset[
                "boundary_dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": self.index.observation_dataset[
                "observation_dataset_fingerprint"
            ],
        }
        self.resume_contract.pop("resume", None)
        self.config_hash = _sha256(self.resume_contract)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("boundary-role experiments require PyTorch") from exc
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return requested

    def _resolve_recipe(self) -> dict[str, Any]:
        profiled: dict[str, Any] = {}
        if self.config.profile_path is not None:
            report = json.loads(self.config.profile_path.read_text(encoding="utf-8"))
            if report.get("schema_version") != "stage2.boundary-role-profile.v3":
                raise ValueError(
                    "profile report is incompatible; run the v0.6.2 profiler"
                )
            recommendation = report.get("recommended_common_configuration")
            if not recommendation:
                raise ValueError(
                    "profile has no common safe configuration for every candidate"
                )
            profiled = recommendation
        batch_size = self.config.batch_size or int(profiled.get("batch_size", 32))
        workers = (
            self.config.num_workers
            if self.config.num_workers is not None
            else int(profiled.get("workers", 2))
        )
        precision = self.config.precision or str(profiled.get("precision", "fp32"))
        if precision == "amp" and not self.device.startswith("cuda"):
            raise ValueError("AMP recipe requires CUDA")
        return {
            "batch_size": batch_size,
            "workers": workers,
            "precision": precision,
            "prefetch_factor": self.config.prefetch_factor,
        }

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "status": "running",
            "config_hash": self.config_hash,
            "resume_contract": self.resume_contract,
            "resolved_recipe": self.recipe,
            "holdout_used": False,
            "stages": {
                "screen": "pending",
                "promotion": "pending",
                "final_training": "pending",
                "calibration": "pending",
                "validation": "pending",
            },
            "candidates": {name: {} for name in self.config.candidates},
        }

    def _prepare(self) -> dict[str, Any]:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"experiment output is not empty: {output}; pass --resume"
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not self.config.resume:
                raise FileExistsError("experiment state exists; pass --resume")
            if state.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
                raise RuntimeError("experiment state schema is incompatible")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError("experiment configuration changed; refusing unsafe resume")
            return state
        unexpected = [
            path.name
            for path in output.iterdir()
            if path.name != "experiment.lock"
        ]
        if unexpected:
            raise RuntimeError(
                "resume state is missing from a non-empty experiment directory: "
                f"{unexpected}"
            )
        state = self._new_state()
        _write_json_atomic(self.state_path, state)
        return state

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_utc"] = _utc_now()
        _write_json_atomic(self.state_path, state)

    def _training_dir(self, name: str) -> Path:
        return self.config.output_dir / "candidates" / name / "training"

    def _snapshot_checkpoint(self, name: str, label: str) -> Path:
        source = self._training_dir(name) / "best.pt"
        target = self.config.output_dir / "candidates" / name / f"{label}_best.pt"
        if target.is_file():
            return target
        if not source.is_file():
            raise RuntimeError(f"candidate {name} has no best checkpoint")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
        return target

    def _train_to(self, name: str, completed_epochs: int | None) -> dict[str, Any]:
        output = self._training_dir(name)
        trainer = BoundaryRoleTrainer(
            BoundaryTrainingConfig(
                boundary_dataset_dir=self.config.boundary_dataset_dir,
                observation_dataset_dir=self.config.observation_dataset_dir,
                statistics_path=self.config.statistics_path,
                output_dir=output,
                seed=self.config.seed,
                epochs=self.config.final_epochs,
                samples_per_epoch=self.config.samples_per_epoch,
                validation_samples=self.config.validation_samples,
                batch_size=self.recipe["batch_size"],
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                checkpoint_every_steps=self.config.checkpoint_every_steps,
                # Successive halving uses equal fixed budgets, not early stopping.
                early_stopping_patience=self.config.final_epochs + 1,
                precision=self.recipe["precision"],
                device=self.config.device,
                num_workers=self.recipe["workers"],
                prefetch_factor=self.recipe["prefetch_factor"],
                resume=output.exists(),
                maximum_completed_epochs=completed_epochs,
            ),
            model_config=candidate_model_config(name),
        )
        return trainer.train()

    def _predictor_for(
        self, checkpoint_path: Path
    ) -> tuple[BoundaryWindowSampler, FullContourPredictor, dict[str, Any]]:
        model, checkpoint = load_boundary_role_model(checkpoint_path, self.device)
        window = BoundaryWindowConfig(**checkpoint["window_config"])
        sampler = BoundaryWindowSampler(self.index, window)
        calibration = checkpoint.get(
            "calibration",
            {"role_temperature": 1.0, "transition_temperature": 1.0},
        )
        fragments = FragmentProposalConfig(**checkpoint.get("fragment_config", {}))
        predictor = FullContourPredictor(
            model,
            sampler,
            ContourInferenceConfig(
                sequence_length=window.sequence_length,
                strip_width=window.strip_width,
                stride=self.config.inference_stride,
                batch_size=self.recipe["batch_size"],
                role_temperature=calibration["role_temperature"],
                transition_temperature=calibration["transition_temperature"],
            ),
            fragments,
            device=self.device,
        )
        return sampler, predictor, checkpoint

    def _evaluate(
        self,
        checkpoint_path: Path,
        output_path: Path,
        maximum_observations: int,
    ) -> dict[str, Any]:
        if output_path.is_file():
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") != checkpoint_sha256(checkpoint_path):
                raise RuntimeError(
                    f"existing evaluation does not match {checkpoint_path}"
                )
            return report
        sampler, predictor, _ = self._predictor_for(checkpoint_path)
        return evaluate_full_contours(
            checkpoint_path=checkpoint_path,
            split="validation",
            index=self.index,
            sampler=sampler,
            predictor=predictor,
            output_path=output_path,
            maximum_observations=maximum_observations,
            seed=self.config.seed,
            failure_diagnostics_dir=output_path.parent
            / f"{output_path.stem}_failures",
        )

    @staticmethod
    def _completed_epochs(training_state: dict[str, Any]) -> int:
        path_value = training_state.get("epoch", 0)
        return int(path_value)

    def run(self) -> dict[str, Any]:
        state = self._prepare()
        with PipelineLock(self.config.output_dir / "experiment.lock"):
            state["status"] = "running"
            state["stages"]["screen"] = "running"
            self._save(state)
            screen_reports: dict[str, dict[str, Any]] = {}
            for name in self.config.candidates:
                report_path = (
                    self.config.output_dir
                    / "candidates"
                    / name
                    / "screen_validation.json"
                )
                snapshot_path = (
                    self.config.output_dir
                    / "candidates"
                    / name
                    / "screen_best.pt"
                )
                if not snapshot_path.is_file():
                    training_state = self._train_to(name, self.config.screen_epochs)
                    if self._completed_epochs(training_state) < self.config.screen_epochs:
                        state["status"] = "stopped"
                        state["candidates"][name]["training_state"] = training_state
                        self._save(state)
                        return state
                    snapshot_path = self._snapshot_checkpoint(name, "screen")
                report = self._evaluate(
                    snapshot_path, report_path, self.config.screen_observations
                )
                screen_reports[name] = report
                state["candidates"][name]["screen"] = {
                    "epochs": self.config.screen_epochs,
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "report": str(report_path.relative_to(self.config.output_dir)),
                    "experiment_score": experiment_selection_score(report),
                }
                self._save(state)
            screen_ranking = rank_experiment_reports(screen_reports)
            promoted = [
                item["model"] for item in screen_ranking[: self.config.promote_count]
            ]
            state["screen_ranking"] = screen_ranking
            state["promoted_models"] = promoted
            state["stages"]["screen"] = "complete"
            state["stages"]["promotion"] = "running"
            self._save(state)

            promotion_reports: dict[str, dict[str, Any]] = {}
            for name in promoted:
                report_path = (
                    self.config.output_dir
                    / "candidates"
                    / name
                    / "promotion_validation.json"
                )
                snapshot_path = (
                    self.config.output_dir
                    / "candidates"
                    / name
                    / "promotion_best.pt"
                )
                if not snapshot_path.is_file():
                    training_state = self._train_to(name, self.config.promotion_epochs)
                    if self._completed_epochs(training_state) < self.config.promotion_epochs:
                        state["status"] = "stopped"
                        state["candidates"][name]["training_state"] = training_state
                        self._save(state)
                        return state
                    snapshot_path = self._snapshot_checkpoint(name, "promotion")
                report = self._evaluate(
                    snapshot_path, report_path, self.config.screen_observations
                )
                promotion_reports[name] = report
                state["candidates"][name]["promotion"] = {
                    "epochs": self.config.promotion_epochs,
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "report": str(report_path.relative_to(self.config.output_dir)),
                    "experiment_score": experiment_selection_score(report),
                }
                self._save(state)
            promotion_ranking = rank_experiment_reports(promotion_reports)
            winner = promotion_ranking[0]["model"]
            state["promotion_ranking"] = promotion_ranking
            state["winner"] = winner
            state["stages"]["promotion"] = "complete"
            state["stages"]["final_training"] = "running"
            self._save(state)

            training_state = self._train_to(winner, None)
            if self._completed_epochs(training_state) < self.config.final_epochs:
                state["status"] = "stopped"
                state["candidates"][winner]["training_state"] = training_state
                self._save(state)
                return state
            best_path = self._snapshot_checkpoint(winner, "final")
            final_uncalibrated_path = (
                self.config.output_dir / "final_uncalibrated_validation.json"
            )
            final_uncalibrated = self._evaluate(
                best_path,
                final_uncalibrated_path,
                self.config.final_observations,
            )
            state["stages"]["final_training"] = "complete"
            state["final_uncalibrated_report"] = final_uncalibrated_path.name
            state["final_uncalibrated_score"] = experiment_selection_score(
                final_uncalibrated
            )
            state["stages"]["calibration"] = "running"
            self._save(state)

            calibrated_path = self.config.output_dir / "calibrated_best.pt"
            calibration_report_path = self.config.output_dir / "calibration_report.json"
            if calibrated_path.is_file() and calibration_report_path.is_file():
                calibration_report = json.loads(
                    calibration_report_path.read_text(encoding="utf-8")
                )
                if calibration_report.get(
                    "input_checkpoint_sha256"
                ) != checkpoint_sha256(best_path):
                    raise RuntimeError("calibration belongs to a different winner")
            elif calibrated_path.exists() or calibration_report_path.exists():
                raise RuntimeError("partial calibration output blocks safe resume")
            else:
                sampler, predictor, checkpoint = self._predictor_for(best_path)
                calibration_report = calibrate_checkpoint(
                    config=CalibrationConfig(
                        checkpoint_path=best_path,
                        output_checkpoint_path=calibrated_path,
                        report_path=calibration_report_path,
                        maximum_observations=self.config.calibration_observations,
                        seed=self.config.seed,
                    ),
                    index=self.index,
                    sampler=sampler,
                    predictor=predictor,
                    checkpoint=checkpoint,
                )
            state["stages"]["calibration"] = "complete"
            state["calibrated_checkpoint"] = calibrated_path.name
            state["calibration_report"] = calibration_report_path.name
            state["stages"]["validation"] = "running"
            self._save(state)

            final_report_path = self.config.output_dir / "final_validation.json"
            final_report = self._evaluate(
                calibrated_path,
                final_report_path,
                self.config.final_observations,
            )
            state["stages"]["validation"] = "complete"
            state["final_validation_report"] = final_report_path.name
            state["final_experiment_score"] = experiment_selection_score(final_report)
            state["status"] = "complete"
            state["holdout_used"] = False
            self._save(state)
            return state

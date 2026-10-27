from __future__ import annotations

import argparse
import hashlib
import json
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
from .pipeline import PipelineLock
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


REFINEMENT_SCHEMA_VERSION = "stage2.fragment-refinement.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _maximum(value: int) -> int | None:
    return None if value == 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate and validate the v0.6.3 hierarchical fragment proposer "
            "using an existing trained boundary-role checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-observations", type=int, default=512)
    parser.add_argument("--validation-observations", type=int, default=512)
    parser.add_argument("--maximum-temperature-points", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--resume", action="store_true")
    return parser


def _predictor(
    *,
    checkpoint_path: Path,
    index: BoundaryTrainingIndex,
    device: str,
    batch_size: int,
    stride: int,
) -> tuple[BoundaryWindowSampler, FullContourPredictor, dict[str, Any]]:
    model, checkpoint = load_boundary_role_model(checkpoint_path, device)
    if (
        checkpoint.get("boundary_dataset_fingerprint")
        != index.boundary_dataset.get("boundary_dataset_fingerprint")
    ):
        raise RuntimeError("checkpoint belongs to a different boundary dataset")
    window = BoundaryWindowConfig(**checkpoint["window_config"])
    sampler = BoundaryWindowSampler(index, window)
    calibration = checkpoint.get("calibration", {})
    fragment_config = FragmentProposalConfig(
        **checkpoint.get("fragment_config", {})
    )
    predictor = FullContourPredictor(
        model,
        sampler,
        ContourInferenceConfig(
            sequence_length=window.sequence_length,
            strip_width=window.strip_width,
            stride=stride,
            batch_size=batch_size,
            role_temperature=float(calibration.get("role_temperature", 1.0)),
            transition_temperature=float(
                calibration.get("transition_temperature", 1.0)
            ),
        ),
        fragment_config,
        device=device,
    )
    return sampler, predictor, checkpoint


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("fragment refinement requires PyTorch") from exc

    for name in ("calibration_observations", "validation_observations"):
        value = int(getattr(args, name))
        if value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size < 1 or args.stride < 1:
        raise SystemExit("--batch-size and --stride must be positive")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    output = args.output
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(
            f"fragment refinement output is not empty: {output}; pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)

    input_sha256 = checkpoint_sha256(args.checkpoint)
    contract = {
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": input_sha256,
        "boundary_dataset_dir": str(args.boundaries.resolve()),
        "observation_dataset_dir": str(args.observations.resolve()),
        "calibration_observations": _maximum(args.calibration_observations),
        "validation_observations": _maximum(args.validation_observations),
        "maximum_temperature_points": args.maximum_temperature_points,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "device": str(device),
        "seed": args.seed,
    }
    contract_hash = _hash_json(contract)
    state_path = output / "fragment_refinement_state.json"
    calibrated_path = output / "calibrated_best_v063.pt"
    calibration_report_path = output / "calibration_report_v063.json"
    validation_report_path = output / "final_validation_v063.json"
    failures_dir = output / "final_validation_v063_failures"

    with PipelineLock(output / "fragment_refinement.lock"):
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != REFINEMENT_SCHEMA_VERSION:
                raise RuntimeError("unsupported fragment refinement state")
            if state.get("contract_hash") != contract_hash:
                raise RuntimeError(
                    "--resume settings differ from the existing refinement run"
                )
        else:
            state = {
                "schema_version": REFINEMENT_SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
                "status": "running",
                "contract_hash": contract_hash,
                "configuration": contract,
                "stages": {"calibration": "pending", "validation": "pending"},
            }
            _write_json_atomic(state_path, state)

        index = BoundaryTrainingIndex(args.boundaries, args.observations)
        if calibrated_path.is_file() and calibration_report_path.is_file():
            calibration_report = json.loads(
                calibration_report_path.read_text(encoding="utf-8")
            )
            if calibration_report.get("input_checkpoint_sha256") != input_sha256:
                raise RuntimeError("saved calibration belongs to another checkpoint")
            print("fragment calibration: reused", flush=True)
        elif calibrated_path.exists() or calibration_report_path.exists():
            raise RuntimeError(
                "partial calibration output blocks safe resume; move the partial "
                "file aside and repeat with --resume"
            )
        else:
            sampler, predictor, checkpoint = _predictor(
                checkpoint_path=args.checkpoint,
                index=index,
                device=device,
                batch_size=args.batch_size,
                stride=args.stride,
            )
            calibration_report = calibrate_checkpoint(
                config=CalibrationConfig(
                    checkpoint_path=args.checkpoint,
                    output_checkpoint_path=calibrated_path,
                    report_path=calibration_report_path,
                    maximum_observations=_maximum(args.calibration_observations),
                    maximum_temperature_points=args.maximum_temperature_points,
                    seed=args.seed,
                ),
                index=index,
                sampler=sampler,
                predictor=predictor,
                checkpoint=checkpoint,
            )
        state["stages"]["calibration"] = "complete"
        state["calibrated_checkpoint"] = calibrated_path.name
        state["calibration_report"] = calibration_report_path.name
        state["updated_utc"] = _utc_now()
        _write_json_atomic(state_path, state)

        calibrated_sha256 = checkpoint_sha256(calibrated_path)
        if validation_report_path.is_file():
            validation_report = json.loads(
                validation_report_path.read_text(encoding="utf-8")
            )
            if validation_report.get("checkpoint_sha256") != calibrated_sha256:
                raise RuntimeError("saved validation belongs to another checkpoint")
            print("fragment validation: reused", flush=True)
        else:
            sampler, predictor, _ = _predictor(
                checkpoint_path=calibrated_path,
                index=index,
                device=device,
                batch_size=args.batch_size,
                stride=args.stride,
            )
            validation_report = evaluate_full_contours(
                checkpoint_path=calibrated_path,
                split="validation",
                index=index,
                sampler=sampler,
                predictor=predictor,
                output_path=validation_report_path,
                maximum_observations=_maximum(args.validation_observations),
                seed=args.seed,
                failure_diagnostics_dir=failures_dir,
            )

        state["stages"]["validation"] = "complete"
        state["validation_report"] = validation_report_path.name
        state["status"] = "complete"
        state["updated_utc"] = _utc_now()
        state["results"] = {
            "selection_score": validation_report["selection_score"],
            "fragments": validation_report["fragments"],
            "failure_count": validation_report["failure_diagnostics"][
                "failure_count"
            ],
            "holdout_used": False,
        }
        _write_json_atomic(state_path, state)
        print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only discovery of local releases and dataset metadata; no inference."""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib
from paper_reconstruction_stage4.outputs import atomic_json, digest, file_hash


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_info(root):
    def command(*args):
        done = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    return {"commit": command("rev-parse", "HEAD"), "tags_at_head": command("tag", "--points-at", "HEAD"),
            "working_changes": command("status", "--short")}


def receipt(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def inventory(desktop):
    stage1 = desktop / "paper-segmentation-stage"
    stage2 = desktop / "paper-reconstruction-stage2"
    stage3 = desktop / "paper-reconstruction-stage3"
    releases = {}
    for name, root in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3)):
        package = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        release = read_json(root / "models" / ("alpha.json" if name == "stage3" else "release.json"))
        models = [receipt(p) for p in sorted((root / "models").glob("*")) if p.suffix in {".onnx", ".pt"}]
        releases[name] = {"root": str(root), "package_version": package["version"], "git": git_info(root),
                          "manifest": receipt(root / "models" / ("alpha.json" if name == "stage3" else "release.json")), "models": models,
                          "release_identity": {k: release[k] for k in ("model_release", "onnx_sha256", "release_version", "manifest_sha256", "software_version", "alpha_version") if k in release}}
    s1_release = read_json(stage1 / "models" / "release.json")
    s2_release = read_json(stage2 / "models" / "release.json")
    stage1_model_ok = file_hash(stage1 / "models" / s1_release["onnx_filename"]) == s1_release["onnx_sha256"]
    stage2_model_hash = file_hash(stage2 / "models" / "stage2_boundary_v1.0.0.onnx")
    # Stage 3's published fixture pins the exact producer identities it accepts.
    fixture = read_json(stage3 / "examples" / "synthetic_page" / "evidence" / "bundle.json")
    release_frozen = {"onnx": "3e80716280375146e2ef66942688e2a262e503df1011918caff262f0bae388a4",
                      "release": "0381683f5c8226974d73915b355bdd5936d80707219e7f083dcdf08f1ecfcfbc"}
    compatibility = {"stage1_model_matches_release": stage1_model_ok,
                     "stage2_model_matches_stage3_frozen_identity": stage2_model_hash == release_frozen["onnx"],
                     "stage2_release_matches_stage3_frozen_identity": file_hash(stage2 / "models" / "release.json") == release_frozen["release"],
                     "stage3_input_contract": fixture["schema_version"]}
    dataset_versions = []
    dataset_root = desktop / "universal-paper-segmenter" / "data"
    for directory in sorted(dataset_root.glob("complete_dataset_v*")):
        freeze = read_json(directory / "dataset_freeze.json")
        dataset_versions.append({"root": str(directory), "fingerprint": freeze.get("fingerprint"),
                                 "sample_count": freeze.get("sample_count"), "split_counts": freeze.get("samples_by_split")})
    latest_dataset = stage1 / "data" / "complete_dataset_v3"
    canonical_root = desktop / "paper_reconstruction_stage3_inputs_v1" / "pilot_canonical"
    canonical = read_json(canonical_root / "dataset_manifest.json")
    split_manifest = read_json(canonical_root / "source_split_manifest.json")
    piece_counts, source_counts = {}, {}
    for puzzle in canonical["puzzles"]:
        key = str(puzzle["actual_piece_count"])
        counts = piece_counts.setdefault(key, {})
        counts[puzzle["split"]] = counts.get(puzzle["split"], 0) + 1
    for source in split_manifest["sources"]:
        source_counts[source["split"]] = source_counts.get(source["split"], 0) + 1
    stage1_manifest = stage1 / "stage1_real_puzzle_output" / "segmentation_manifest.jsonl"
    stage1_rows = [json.loads(line) for line in stage1_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if stage1_manifest.exists() else []
    native_predictions = desktop / "paper_reconstruction_stage3_inputs_v1" / "boundary_role_predictions_v064" / "prediction_manifest.json"
    prediction_metadata = read_json(native_predictions)
    return {"schema_version": "stage4.local-upstream-inventory.v1", "inference_run": False, "reference_maps_opened": False,
            "selected_production_projects": releases, "compatibility": compatibility,
            "stage1_datasets": dataset_versions, "selected_stage1_dataset": {"root": str(latest_dataset), "resolved_root": str(latest_dataset.resolve()), "is_shared_directory": latest_dataset.resolve() != latest_dataset},
            "stage2_canonical_dataset": {"root": str(canonical_root), "fingerprint": canonical["dataset_fingerprint"],
                                         "puzzles": len(canonical["puzzles"]), "piece_counts_by_split": piece_counts,
                                         "source_file_counts_by_split": source_counts},
            "stage2_native_predictions": {"manifest": str(native_predictions), "status": prediction_metadata["status"],
                                           "puzzle_count": prediction_metadata["puzzle_count"], "observation_count": prediction_metadata["observation_count"]},
            "existing_real_stage1_output": {"manifest": str(stage1_manifest), "rows": len(stage1_rows),
                                            "accepted_rows": sum(row.get("accepted") is True for row in stage1_rows)},
            "historical_projects": [str(desktop / name) for name in ("universal-paper-segmenter", "paper-reconstruction-stage2-v0.6.2", "paper-reconstruction-dataset")],
            "limitations": ["Inventory counts do not certify compatible 30-piece production exports.",
                            "Historical native Stage2 predictions need an explicit production-lineage adapter before alpha serving.",
                            "Only one held-out source file exists in the current canonical split."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", type=Path, default=Path.home() / "Desktop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.desktop.resolve())
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "compatibility": report["compatibility"],
                      "real_stage1": report["existing_real_stage1_output"], "dataset_versions": [{"root": d["root"], "samples": d["sample_count"]} for d in report["stage1_datasets"]]}, indent=2))


if __name__ == "__main__":
    main()

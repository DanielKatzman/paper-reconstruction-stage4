"""Fixed metadata-only ten-by-eight confirmation and four-by-twelve scale cohort."""
import argparse
from collections import Counter
from pathlib import Path
from .common import DESKTOP, atomic_json, digest, load_cohort, read_json
from .prepare import export_case

ANCHOR = "puzzle_000174_0f1f6d83b42a"
TIERS = (("simple", 3), ("moderate", 3), ("hard", 2), ("extreme", 2))


def choose(rows, seed):
    """Balance family/page exposure without consulting any reconstruction result."""
    anchor = next(r for r in rows if r["puzzle_id"] == ANCHOR)
    selected = [anchor]
    families, pages = Counter([anchor["source_family"]]), {anchor["source_id"]}
    for tier, count in TIERS:
        count -= int(tier == anchor["difficulty_tier"])
        for _ in range(count):
            available = [r for r in rows if r["difficulty_tier"] == tier and r not in selected]
            if not available:
                raise ValueError("Insufficient distinct cases in " + tier)
            row = min(available, key=lambda r: (r["source_id"] in pages, families[r["source_family"]],
                                                digest([seed, r["puzzle_id"]])))
            selected.append(row)
            families[row["source_family"]] += 1
            pages.add(row["source_id"])
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DESKTOP / "paper_reconstruction_stage3_inputs_v1")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/validation8x10_scale12_v1"))
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        _, old = load_cohort(output)
        if old["selection"]["seed"] != args.seed or old["source_root"] != str(source):
            raise ValueError("Existing cohort parameters differ")
        print("Verified existing cohort:", output)
        return 0
    dataset = read_json(source / "pilot_canonical/dataset_manifest.json")
    rows = []
    for row in dataset["puzzles"]:
        if row["split"] == "train" and row["actual_piece_count"] == 8:
            puzzle = read_json(source / "pilot_canonical" / row["puzzle_path"])
            rows.append({**row, "source_family": Path(puzzle["source"]["origin"]).name})
    selected = choose(rows, args.seed)
    # Reuse the predeclared twelve-piece ladder, without looking at its outcomes.
    _, original = load_cohort(Path(__file__).resolve().parents[1] / "benchmarks/development_v2", verify=False)
    scale_ids = {c["case_id"] for c in original["cases"] if c["pieces"] == 12}
    for row in dataset["puzzles"]:
        if row["puzzle_id"] in scale_ids:
            puzzle = read_json(source / "pilot_canonical" / row["puzzle_path"])
            selected.append({**row, "source_family": Path(puzzle["source"]["origin"]).name})
    if len(scale_ids) != 4 or len(selected) != 14:
        raise ValueError("Incomplete cohort")
    staging = output.with_name(output.name + ".preparing")
    staging.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in selected:
        cases.append(export_case(source, row, staging / "cases" / row["puzzle_id"]))
        print(f"Prepared {len(cases)}/14: {row['puzzle_id']}", flush=True)
    manifest = {"schema_version": "stage4.development-benchmark.v1", "source_root": str(source),
                "input_adapter_version": "observed-crop-with-4px-contour-margin.v2",
                "canonical_dataset_fingerprint": dataset["dataset_fingerprint"],
                "observed_dataset_fingerprint": read_json(source / "pilot_observed/observation_dataset_manifest.json")["observation_dataset_fingerprint"],
                "selection": {"seed": args.seed, "sizes": [8, 12], "puzzles_per_size": {"8": 10, "12": 4},
                    "policy": "Eight: known anchor plus 9 distinct metadata-selected cases; 3 simple, 3 moderate, 2 hard, 2 extreme. Prefer unused pages then least-used source family, seeded tie break. Twelve: original fixed development_v2 ladder. Train, view00, no outcome selection.",
                    "previously_reviewed_case_ids": [ANCHOR]},
                "stage1": {"status": "bypassed", "segmentation_accuracy_measured": False,
                           "inputs": "Generator observed images/masks, cropped with four-pixel background margin"},
                "purpose": "Development confirmation, not independent holdout reliability; source families can overlap upstream training.",
                "cases": cases}
    manifest["benchmark_sha256"] = digest(manifest)
    atomic_json(staging / "benchmark.json", manifest)
    staging.rename(output)
    print("Ready:", output)
    return 0

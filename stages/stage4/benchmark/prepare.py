"""Freeze generated observed cuts as Stage 1 shaped inputs, with isolated truth."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from paper_reconstruction_contracts import map_points
from .common import DESKTOP, SIZES, atomic_json, canonical, digest, inventory, load_cohort, read_json, receipt


def select_cases(candidates, seed=20261021):
    """One per difficulty per size; maximize distinct source files, then seeded order.

    Selection uses generator metadata only. No predicted or measured match quality.
    """
    selected = []
    tiers = ("simple", "moderate", "hard", "extreme")
    for size in SIZES:
        groups = []
        for tier in tiers:
            group = [c for c in candidates if c["actual_piece_count"] == size and c["difficulty_tier"] == tier and c["split"] == "train"]
            if not group:
                raise ValueError(f"Missing training cuts at {size}/{tier}")
            groups.append(sorted(group, key=lambda c: digest([seed, c["puzzle_id"]])))
        choices = itertools.product(*groups)
        best = min(choices, key=lambda rows: (-len({r["source_family"] for r in rows}),
                                             digest([seed, [r["puzzle_id"] for r in rows]])))
        selected.extend(best)
    return selected


def crop_observation(image, mask, observed_to_page):
    mask = np.asarray(mask.convert("L"))
    if image.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("Observation image and mask dimensions differ")
    if not set(np.unique(mask)).issubset({0, 255}) or not mask.any():
        raise ValueError("Generated mask must be nonempty binary 0/255")
    components = int(ndimage.label(mask > 0)[1])
    if components != 1:
        raise ValueError(f"Generated mask has {components} connected components")
    y, x = np.nonzero(mask)
    # Stage 2's marching-squares contour follows half-pixel mask boundaries.
    # Stage 3 requires that complete contour to remain inside the native image.
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Observation foreground touches its canvas edge; cannot preserve an interior contour")
    x0, y0 = max(0, int(x.min())-4), max(0, int(y.min())-4)
    x1, y1 = min(mask.shape[1], int(x.max())+5), min(mask.shape[0], int(y.max())+5)
    transform = np.asarray(observed_to_page, float) @ np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
    return image.convert("RGB").crop((x0, y0, x1, y1)), Image.fromarray(mask[y0:y1, x0:x1]), [x0, y0, x1-x0, y1-y0], transform


def densify(points, spacing=1.):
    points = np.asarray(points, float)
    result = []
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b-a) / spacing)))
        result.extend(a + (b-a) * np.arange(count)[:, None] / count)
    return np.asarray([*result, points[-1]], float)


def contact_sheet(stage1, rows, output):
    width, cell = 960, 240
    canvas = Image.new("RGB", (width, ((len(rows)+3)//4)*cell), "#e8ebee")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        with Image.open(stage1 / row["piece_path"]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((cell-16, cell-38))
        x, y = (i % 4)*cell, (i//4)*cell
        canvas.paste(thumb, (x+(cell-thumb.width)//2, y+25+(cell-38-thumb.height)//2))
        draw.text((x+8, y+6), row["capture_id"], fill="black")
    canvas.save(output)


def export_case(source, row, target):
    canonical_root = source / "pilot_canonical"
    puzzle_path = canonical_root / row["puzzle_path"]
    puzzle = read_json(puzzle_path)
    observed_root = source / "pilot_observed" / "puzzles" / row["puzzle_id"]
    observed = read_json(observed_root / "observation_manifest.json")
    if not puzzle["validation"]["valid"] or not observed["validation"]["valid"]:
        raise ValueError("Invalid generator puzzle")
    if observed["canonical_puzzle_fingerprint"] != puzzle["puzzle_fingerprint"]:
        raise ValueError("Canonical/observed fingerprint mismatch")
    stage1, truth_root = target / "stage1", target / "evaluation_only"
    (stage1 / "pieces").mkdir(parents=True, exist_ok=True)
    (stage1 / "piece_masks").mkdir(exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)
    rows, pieces, sources = [], {}, []
    observations = sorted((o for o in observed["observations"] if o["view_index"] == 0), key=lambda o: o["piece_id"])
    if len(observations) != row["actual_piece_count"] or len({o["piece_id"] for o in observations}) != len(observations):
        raise ValueError("Expected exactly one view-00 per physical piece")
    for obs in observations:
        pid = obs["piece_id"]
        image_path, mask_path = observed_root / obs["observed_piece_path"], observed_root / obs["observed_mask_path"]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            size = [image.height, image.width]
            crop, crop_mask, bbox, transform = crop_observation(image, mask, obs["observed_to_source_page_homography"])
        crop.save(stage1 / "pieces" / f"{pid}.png")
        crop_mask.save(stage1 / "piece_masks" / f"{pid}.png")
        rows.append({"schema_version": "stage1.segmentation-manifest.v1", "capture_id": pid,
                     "accepted": True, "original_size_hw": size, "bbox_xywh": bbox,
                     "piece_path": f"pieces/{pid}.png", "piece_mask_path": f"piece_masks/{pid}.png",
                     "decision_source": "generator_mask_bypass", "model_release": "generator-mask-bypass",
                     "review_status": "generator_validated"})
        pieces[pid] = {"native_to_reference": transform.tolist(), "observation_fingerprint": obs["observation_fingerprint"],
                       "mapping_semantics": obs["mapping_semantics"], "mask_perturbation": obs["mask_perturbation"]}
        sources.extend([{"origin": str(p), **receipt(p, observed_root)} for p in (image_path, mask_path)])
    (stage1 / "segmentation_manifest.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in rows))
    contact_sheet(stage1, rows, stage1 / "input_pieces.jpg")
    contacts, seams, borders = {}, [], []
    with np.load(puzzle_path.parent / puzzle["truth_path"], allow_pickle=False) as arrays:
        def polyline(item):
            ref = item["canonical_polyline_ref"]
            offsets = arrays[ref["offsets_array"]]
            return arrays[ref["points_array"]][offsets[ref["index"]]:offsets[ref["index"]+1]]
        for seam in puzzle["seams"]:
            a, b = sorted((seam["owner_a"], seam["owner_b"]))
            points = densify(polyline(seam))
            # Fixed, equally spaced index probes; multiple seams on a pair share one denominator.
            probes = points[np.linspace(0, len(points)-1, min(65, len(points))).astype(int)]
            contacts.setdefault((a, b), []).extend(probes.tolist())
            seams.append({"seam_id": seam["seam_id"], "piece_a": a, "piece_b": b,
                          "points_reference": points.tolist()})
        for border in puzzle["page_border_segments"]:
            borders.append({"piece_id": border["piece_id"], "points_reference": densify(polyline(border)).tolist()})
    reference = {"schema_version": "stage4.evaluation-reference.v1", "split": "development_train",
                 "pieces": pieces, "contacts": [], "seams": seams, "borders": borders,
                 "reference_size_hw": puzzle["source"]["size_hw"],
                 "probe_semantics": "Generator grid-line seams, mapped through ideal observation homographies; mask damage is not inverted. Multiple seams per pair are concatenated. P95 in native piece-A pixels, threshold 4px."}
    for (a, b), points in sorted(contacts.items()):
        reference["contacts"].append({"piece_a": a, "piece_b": b,
            "points_a": map_points(points, np.linalg.inv(pieces[a]["native_to_reference"])).tolist(),
            "points_b": map_points(points, np.linalg.inv(pieces[b]["native_to_reference"])).tolist()})
    atomic_json(truth_root / "reference_geometry.json", reference)
    with Image.open((puzzle_path.parent / puzzle["source"]["dataset_path"]).resolve()) as page:
        thumb = page.convert("RGB")
        thumb.thumbnail((1000, 1400))
        thumb.save(truth_root / "reference_page.jpg")
    atomic_json(truth_root / "source_receipts.json", {"puzzle": {"origin": str(puzzle_path), **receipt(puzzle_path, canonical_root)},
        "observation_manifest": receipt(observed_root / "observation_manifest.json", observed_root), "images_and_masks": sources,
        "source": puzzle["source"], "generator_version": puzzle["generator_version"], "compiler_version": observed["compiler_version"]})
    return {"case_id": row["puzzle_id"], "pieces": len(rows), "difficulty": row["difficulty_tier"],
            "source_family": row["source_family"], "source_page_id": row["source_id"], "source_page_index": puzzle["source"]["page_index"],
            "split": "development_train", "canonical_puzzle_fingerprint": puzzle["puzzle_fingerprint"],
            "path": f"cases/{row['puzzle_id']}", "files": inventory(target)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DESKTOP / "paper_reconstruction_stage3_inputs_v1")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/development_v2"))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--resume", action="store_true", help="Rebuild an interrupted preparation from the same immutable source")
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        root, manifest = load_cohort(output)
        if (manifest["selection"]["seed"] != args.seed or manifest["source_root"] != str(source)
                or manifest.get("input_adapter_version") != "observed-crop-with-4px-contour-margin.v2"):
            raise ValueError("Existing immutable benchmark has different preparation parameters")
        print(f"Verified existing benchmark: {root}; {len(manifest['cases'])} cases")
        return 0
    dataset = read_json(source / "pilot_canonical/dataset_manifest.json")
    candidates = []
    for row in dataset["puzzles"]:
        if row["actual_piece_count"] in SIZES and row["split"] == "train":
            puzzle = read_json(source / "pilot_canonical" / row["puzzle_path"])
            candidates.append({**row, "source_family": Path(puzzle["source"]["origin"]).name})
    selected = select_cases(candidates, args.seed)
    staging = output.with_name(output.name + ".preparing")
    if staging.exists() and not args.resume:
        raise ValueError(f"Partial preparation exists at {staging}; preserve/inspect it and choose a new output directory")
    staging.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in selected:
        case = export_case(source, row, staging / "cases" / row["puzzle_id"])
        cases.append(case)
        print(f"Prepared {len(cases)}/32: {case['case_id']} ({case['pieces']} pieces, {case['difficulty']})", flush=True)
    manifest = {"schema_version": "stage4.development-benchmark.v1", "source_root": str(source),
        "input_adapter_version": "observed-crop-with-4px-contour-margin.v2",
        "canonical_dataset_fingerprint": dataset["dataset_fingerprint"],
        "observed_dataset_fingerprint": read_json(source / "pilot_observed/observation_dataset_manifest.json")["observation_dataset_fingerprint"],
        "selection": {"seed": args.seed, "sizes": list(SIZES), "puzzles_per_size": 4,
            "policy": "Training split only; one simple/moderate/hard/extreme per size, maximize distinct source files, seeded metadata tie-break. View 00 only. No outcome-based selection."},
        "stage1": {"status": "bypassed", "inputs": "existing generated observed images and masks, cropped with 4px background margin",
                   "segmentation_accuracy_measured": False},
        "purpose": "Development and tuning; generated source pages may have been seen upstream. Not a holdout reliability claim.",
        "cases": cases}
    manifest["benchmark_sha256"] = digest(manifest)
    atomic_json(staging / "benchmark.json", manifest)
    staging.rename(output)
    print(f"Ready: {output / 'benchmark.json'} ({sum(c['pieces'] for c in cases)} pieces)")
    return 0

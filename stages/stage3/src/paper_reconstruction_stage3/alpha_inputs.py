"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from __future__ import annotations
import json
import os
from contextlib import contextmanager
from pathlib import Path
from .contracts import FROZEN, ROLE_NAMES, ContractError, atomic_json, digest, integer, read_json, require, resolve, sha256, utc, validate_candidates
from .data import load_contour, check_images

class Files:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def add(self, path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        require(path.is_file(), f"Missing input file: {path}")
        if str(path) not in self.items:
            stat = path.stat()
            actual = sha256(path)
            self.items[str(path)] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": actual}
        if expected is not None:
            require(self.items[str(path)]["sha256"] == expected, f"SHA-256 mismatch: {path}")
        return path

    def read(self, path: Path) -> dict:
        return read_json(self.add(path))

    def receipt(self) -> list[dict]:
        return list(self.items.values())


def keyed(items: list[dict], field: str, context: str) -> dict:
    require(isinstance(items, list), f"{context}: list required")
    result = {}
    for item in items:
        key = item.get(field)
        require(isinstance(key, str) and bool(key) and key not in result, f"{context}: missing/duplicate {field}")
        result[key] = item
    return result


@contextmanager
def lock(output: Path):
    path = output / "audit.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ContractError(f"Audit lock exists: {path}. If the prior process is stopped, remove that stale lock before resuming.") from exc
    with os.fdopen(fd, "w") as stream:
        json.dump({"pid": os.getpid(), "created_utc": utc()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def production_header(config: dict, files: Files, warnings: list[str]) -> dict:
    root = Path(config["stage2"]).resolve()
    m = files.read(root / "edge_hypotheses_manifest.json")
    require(m.get("schema_version") == "stage2.edge-hypotheses-manifest.v1" and m.get("status") == "complete" and m.get("truth_free") is True, "Incomplete/unsupported Stage 2 production handoff")
    legacy = m.get("model_format") != "onnx"
    if legacy:
        require(config.get("allow_legacy_stage2") is True and m.get("checkpoint_sha256") == FROZEN["checkpoint"], "Legacy Stage 2 input requires --allow-legacy-stage2 and the known checkpoint")
        warnings.append("Legacy PyTorch Stage 2 bundle: integration diagnostic only, not frozen ONNX handoff.")
    else:
        require(m.get("model_sha256") == FROZEN["onnx"] and m.get("source_checkpoint_sha256") == FROZEN["checkpoint"] and m.get("release_manifest_sha256") == FROZEN["release"] and m.get("release_version") == "1.0.0" and m.get("runtime") == "onnxruntime-cpu", "Wrong frozen Stage 2 production release identity")
        require(digest(m["configuration"]) == m.get("config_hash"), "Stage 2 production configuration hash mismatch")
    require(len(keyed(m.get("pieces"), "capture_id", "production")) == m.get("piece_count"), "Production piece total mismatch")
    m["_root"] = str(root)
    m["_stage1_rows"] = {}
    if config.get("geometry_only"):
        warnings.append("Geometry-only audit: Stage 1 RGB/masks and appearance interoperability are not verified.")
    else:
        require(config.get("stage1"), "--stage1 is required unless --geometry-only is explicitly requested")
        sroot = Path(config["stage1"]).resolve()
        manifest = files.add(sroot / "segmentation_manifest.jsonl", m["stage1_manifest_sha256"])
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        keyed(rows, "capture_id", "Stage 1")
        cfg = m["configuration"]
        require(cfg.get("include_unaccepted") is False, "Accepted Stage 1 pieces only are supported")
        selected = [r for r in rows if r.get("accepted") is True]
        if cfg.get("maximum_pieces") is not None:
            selected = selected[:integer(cfg["maximum_pieces"], "maximum_pieces", 1)]
        require({r["capture_id"] for r in selected} == {p["capture_id"] for p in m["pieces"]}, "Selected Stage 1 and Stage 2 capture sets disagree")
        fingerprints = []
        for row in selected:
            image = files.add(resolve(sroot, row["piece_path"]))
            mask = files.add(resolve(sroot, row["piece_mask_path"]))
            fp = digest({"row": row, "piece_sha256": sha256(image), "piece_mask_sha256": sha256(mask)})
            fingerprints.append(fp)
            m["_stage1_rows"][row["capture_id"]] = {"row": row, "fingerprint": fp, "image": str(image), "mask": str(mask)}
        require(digest({"manifest_sha256": m["stage1_manifest_sha256"], "include_unaccepted": False, "maximum_pieces": cfg.get("maximum_pieces"), "capture_fingerprints": fingerprints}) == m["stage1_input_fingerprint"], "Stage 1 input fingerprint mismatch")
    return m


def production_piece(config: dict, manifest: dict, summary: dict, files: Files) -> list[dict]:
    root = Path(manifest["_root"])
    rpath = resolve(root, summary["record_path"])
    r = files.read(rpath)
    require(r.get("schema_version") == "stage2.edge-hypotheses.v1" and r.get("point_order") == "clockwise_circular" and r.get("role_names") == ROLE_NAMES, "Invalid production record schema/orientation/role order")
    for field in ("capture_id", "piece_id", "point_count", "candidate_count", "stage1_record_fingerprint"):
        require(r.get(field) == summary.get(field), f"Production summary {field} mismatch")
    for field in ("config_hash", "stage1_input_fingerprint"):
        require(r.get(field) == manifest.get(field), f"Production lineage {field} mismatch")
    if manifest.get("model_format") == "onnx":
        for field in ("model_sha256", "source_checkpoint_sha256", "release_manifest_sha256", "release_version", "model_format", "runtime"):
            require(r.get(field) == manifest.get(field), f"Production release {field} mismatch")
    else:
        require(r.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"), "Legacy checkpoint lineage mismatch")
    require(r["arrays_path"] == summary["arrays_path"], "Production array path mismatch")
    arrays = files.add(resolve(root, r["arrays_path"]), r["arrays_sha256"])
    if r.get("overlay_path"):
        files.add(resolve(root, r["overlay_path"]), r["overlay_sha256"])
    n = integer(r["point_count"], "point_count", 3)
    candidates = validate_candidates(r["candidate_fragments"], n)
    require(len(candidates) == r["candidate_count"], "Production candidate total mismatch")
    frames = r["coordinate_frames"]
    for field in ("piece_crop", "source_capture"):
        for k, v in (("origin", "top_left"), ("x_axis", "right"), ("y_axis", "down"), ("units", "pixels")):
            require(frames[field].get(k) == v, "Unsupported coordinate frame")
    image, mask = None, None
    provenance = r["stage1_provenance"]
    require(provenance.get("accepted") is True, "Stage 1 piece was not accepted")
    bbox = provenance["bbox_xywh"]
    if not config.get("geometry_only"):
        s = manifest["_stage1_rows"][r["capture_id"]]
        image = files.add(Path(s["image"]), r["piece_sha256"])
        mask = files.add(Path(s["mask"]), r["piece_mask_sha256"])
        require(s["fingerprint"] == r["stage1_record_fingerprint"], "Stage 1 record fingerprint mismatch")
        for key, value in provenance.items():
            require(s["row"].get(key) == value, f"Stage 1 provenance changed: {key}")
    entry = {"mode": "production", "observation_key": r["capture_id"], "puzzle_id": "production", "split": "real", "difficulty_tier": "real", "piece_id": r["piece_id"], "observation_id": r["capture_id"], "view_index": None, "point_count": n,
        "record_path": str(rpath), "arrays_path": str(arrays), "boundaries_path": None, "offset_start": 0, "offset_end": n, "observation_index": 0,
        "image_path": str(image) if image else None, "mask_path": str(mask) if mask else None, "size_hw": [bbox[3], bbox[2]], "crop_to_source_homography": frames["crop_to_source_homography"], "candidates": candidates}
    contour = load_contour(entry)
    if not config.get("geometry_only"):
        check_images(entry, contour)
    return [entry]

def unchanged(files, rehash):
    for item in files:
        path = Path(item['path'])
        if not path.is_file(): return False
        stat = path.stat()
        if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']: return False
        if rehash and sha256(path) != item['sha256']: return False
    return True

import json
from pathlib import Path
from paper_reconstruction_stage4.outputs import atomic_json, canonical, digest, file_hash

WORKSPACE = Path(__file__).resolve().parents[1]
DESKTOP = WORKSPACE.parent
SIZES = (4, 8, 12, 16, 20, 24, 28, 30)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def receipt(path, root):
    path, root = Path(path), Path(root)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def inventory(root):
    root = Path(root)
    return [receipt(p, root) for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")]


def verify_files(root, files):
    root = Path(root).resolve()
    for item in files:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Artifact changed or missing: {path}")


def load_cohort(path, verify=True, serving_only=False):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "benchmark.json"
    value = read_json(path)
    if value.get("schema_version") != "stage4.development-benchmark.v1":
        raise ValueError("Unsupported benchmark")
    if value["benchmark_sha256"] != digest({k: v for k, v in value.items() if k != "benchmark_sha256"}):
        raise ValueError("Benchmark manifest digest mismatch")
    if verify:
        for case in value["cases"]:
            files = [f for f in case["files"] if not serving_only or f["path"].startswith("stage1/")]
            verify_files(path.parent / case["path"], files)
    return path.parent, value

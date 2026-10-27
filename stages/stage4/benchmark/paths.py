"""Explicit per-case artifact mapping for frozen and profiled producers."""
from pathlib import Path


def artifact_paths(case_root, progress):
    root = Path(case_root).resolve()
    defaults = {"stage2": "stage2", "stage3": "stage3", "stage3_work": "stage3.work",
                "stage4": "stage4", "stage4_work": "stage4.work", "logs": "logs"}
    overrides = progress.get("artifact_paths", {})
    if set(overrides) - {*defaults, "stage3_profile"}:
        raise ValueError("Unknown artifact path mapping")
    paths = {}
    for name, relative in {**defaults, **overrides}.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Artifact mappings must be relative to this case")
        path = (root / relative).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError("Artifact mapping escapes its case directory")
        paths[name] = path
    return paths

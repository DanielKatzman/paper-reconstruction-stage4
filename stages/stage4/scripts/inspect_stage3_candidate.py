"""Read-only candidate verification, executed with Stage 3's existing Python.

No producer is imported into the Stage 4 solver. Never builds/repairs snapshots.
"""
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

stage3 = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(stage3 / "scripts"))
from stage3_candidate_runtime import expected_manifest, validate_candidate

expected, _ = expected_manifest()
folder = stage3 / "generated/stage3_candidates" / (expected["variant"] + "_" + expected["candidate_implementation_sha256"][:16] + "_" + expected["manifest_sha256"][:8])
if not folder.is_dir():
    raise ValueError("The current candidate snapshot is absent; prepare it in the Stage 3 workspace first: " + str(folder))
actual = validate_candidate(folder)
print(json.dumps({"status": "candidate_verified", "path": str(folder), "manifest": actual,
                  "environment": {"python": platform.python_version(), "torch_distribution": version("torch"),
                                  "numpy_distribution": version("numpy"), "pillow_distribution": version("Pillow"),
                                  "gpu_execution_checked": False}}, sort_keys=True))

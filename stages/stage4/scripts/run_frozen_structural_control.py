"""Run the preserved structural-v2 source in a fresh process."""
import json
from pathlib import Path
import runpy
import sys
import hashlib

root = Path(__file__).resolve().parents[1] / "controls/stage4_structural_v2_joint_baseline"
receipt = json.loads((root / "receipt.json").read_text())
for name, expected in receipt["files"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen structural control changed: " + name)
sys.path.insert(0, str(root / "src"))
from paper_reconstruction_stage4.outputs import implementation_identity
if implementation_identity() != receipt["implementation"]:
    raise ValueError("Frozen structural control/dependencies differ")
runpy.run_module("paper_reconstruction_stage4", run_name="__main__")

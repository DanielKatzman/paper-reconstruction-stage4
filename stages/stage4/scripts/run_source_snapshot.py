"""Execute a verified Stage 4 source snapshot in a fresh interpreter."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv.pop(1)).resolve()
receipt = json.loads((root / "receipt.json").read_text())
for name, expected in receipt["files"].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Source snapshot changed: " + name)
sys.path.insert(0, str(root / "src"))
from paper_reconstruction_stage4.outputs import implementation_identity
if implementation_identity() != receipt["implementation"]:
    raise ValueError("Snapshot source or installed dependencies changed")
runpy.run_module("paper_reconstruction_stage4", run_name="__main__")

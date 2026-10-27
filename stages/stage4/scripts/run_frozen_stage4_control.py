"""Launch the preserved consumer source in a fresh process, using this venv."""
from pathlib import Path
import runpy
import sys

control = Path(__file__).resolve().parents[1] / "controls/stage4_alpha_0_1_0_review_baseline/src"
if not (control / "paper_reconstruction_stage4/__main__.py").is_file():
    raise SystemExit("Frozen control snapshot is missing")
sys.path.insert(0, str(control))
runpy.run_module("paper_reconstruction_stage4", run_name="__main__", alter_sys=True)

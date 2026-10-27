"""Run the small focused suite and record a machine-readable validation receipt."""
import json
from pathlib import Path
import re
import subprocess
import sys
from paper_reconstruction_stage4.outputs import atomic_json, implementation_identity

root = Path(__file__).resolve().parents[1]
completed = subprocess.run([sys.executable, str(root / "scripts" / "test.py")], cwd=root, capture_output=True, text=True)
log = completed.stdout + completed.stderr
(root / "research" / "TEST_OUTPUT.txt").write_text(log, encoding="utf-8")
match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", log)
report = {"schema_version": "stage4.implementation-validation.v1", "passed": completed.returncode == 0,
          "tests_run": int(match[1]) if match else None, "test_seconds": float(match[2]) if match else None,
          "implementation": implementation_identity(), "log": "research/TEST_OUTPUT.txt",
          "scope": "Known-layout solver fixtures and contract/operational tests; not real assembly capacity",
          "full_pipeline_run": False, "real_assembly_pilot_run": False, "model_training_performed": False}
atomic_json(root / "research" / "IMPLEMENTATION_VALIDATION.json", report)
print(json.dumps({k: report[k] for k in ("passed", "tests_run", "test_seconds", "scope")}, indent=2))
if completed.returncode:
    print(log)
raise SystemExit(completed.returncode)

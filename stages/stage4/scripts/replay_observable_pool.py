"""Generic wrapper for the upstream observable-only pool policy; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage3-root", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-script-sha256", required=True)
args = parser.parse_args()
script = args.stage3_root / "scripts/stage3_pool_replay_tools.py"
if hashlib.sha256(script.read_bytes()).hexdigest() != args.expected_script_sha256:
    raise ValueError("Upstream pool implementation changed")
sys.path.insert(0, str(script.parent))
from stage3_pool_replay_tools import publish_replay, atomic_json
audit = publish_replay(args.source, args.output,
                       {"stage3_pool_replay_tools.py": args.expected_script_sha256,
                        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
atomic_json(args.output.parent / "pool_replay.json", audit)
print(json.dumps(audit, sort_keys=True))

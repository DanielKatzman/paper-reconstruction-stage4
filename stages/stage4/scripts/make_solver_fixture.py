import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from tests.fixtures import build_fixture
from paper_reconstruction_stage4.outputs import atomic_json

parser = argparse.ArgumentParser(description="Write a tiny explicit-layout solver fixture, not a capacity benchmark")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Use a new output directory")
reference = build_fixture(args.output / "evidence", alternatives=True)
atomic_json(args.output / "evaluation_only" / "reference.json", reference)
print(args.output.resolve())

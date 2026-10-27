"""Read-only v3 inspection using Stage 3's environment; never builds a runtime."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.qualified_v3 import inspect_runtime
print(json.dumps(inspect_runtime(),sort_keys=True))

"""Run only Stage 4 on the four authenticated Stage 3 unit-scale bundles."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.native_repair_experiment_v14 import main
if __name__ == '__main__':
    raise SystemExit(main())

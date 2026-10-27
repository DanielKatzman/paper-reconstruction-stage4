"""Fresh 2x4, 2x8, 2x12, 2x16 benchmark with a frozen V14 consumer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.scale_pipeline_v14 import main
if __name__ == '__main__':
    raise SystemExit(main())

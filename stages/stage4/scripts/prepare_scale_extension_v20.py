"""Stage 4 capacity extension through 30 pieces."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.scale_extension_v20 import prepare_main
if __name__ == "__main__":raise SystemExit(prepare_main())

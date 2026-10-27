"""Run one consumer experiment using previously sealed Stage 3 evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.experiment import main
raise SystemExit(main())

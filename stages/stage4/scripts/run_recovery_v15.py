"""Run V15 consumer-only comparisons on the sealed eight-case benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.recovery_experiment_v15 import main
if __name__=='__main__':raise SystemExit(main())

"""Replay all fourteen sealed capacity inputs with V21 Stage 4."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.cut_repair_experiment_v21 import main
if __name__=='__main__':raise SystemExit(main())

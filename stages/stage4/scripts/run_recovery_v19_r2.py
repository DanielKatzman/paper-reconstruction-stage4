"""Run V19 straight-deferred growth on the sealed eight-case Stage 4 inputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.recovery_experiment_v19_r2 import main
if __name__=="__main__":raise SystemExit(main())

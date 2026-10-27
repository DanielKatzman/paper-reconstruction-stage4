from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.quality_experiments import diagnose
raise SystemExit(diagnose())

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.support_aware_experiments import main
raise SystemExit(main())

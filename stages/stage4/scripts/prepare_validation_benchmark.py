import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.validation_cohort import main
raise SystemExit(main())

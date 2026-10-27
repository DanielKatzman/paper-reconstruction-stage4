import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.consumer_replay import main
raise SystemExit(main())

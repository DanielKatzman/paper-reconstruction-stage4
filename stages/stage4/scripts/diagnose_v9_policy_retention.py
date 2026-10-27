from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.snapshot_diagnosis import select_core
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',required=True)
a,_=p.parse_known_args();select_core(a.run)
from benchmark.policy_retention_experiments import diagnose
raise SystemExit(diagnose())

from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    from benchmark.snapshot_diagnosis import select_core
    select_core(args.run)
    from benchmark.retention_experiments import diagnose
    raise SystemExit(diagnose())

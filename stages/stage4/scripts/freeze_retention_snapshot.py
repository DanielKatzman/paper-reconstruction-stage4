from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.snapshots import freeze
from benchmark.common import WORKSPACE
from benchmark.retention_experiments import VARIANTS
parser=argparse.ArgumentParser(description='Freeze opt-in cold retention experiments')
parser.add_argument('--output',type=Path,default=WORKSPACE/'controls/stage4_retention_v6_conservative_development')
args=parser.parse_args()
receipt=freeze(args.output,[WORKSPACE/'configs'/n for n in VARIANTS.values()])
print('Frozen retention source:',receipt['implementation']['sha256'],args.output.resolve())

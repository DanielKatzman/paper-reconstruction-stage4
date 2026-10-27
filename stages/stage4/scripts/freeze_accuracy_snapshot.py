from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE
from benchmark.snapshots import freeze
from benchmark.accuracy_experiments import VARIANTS
parser=argparse.ArgumentParser(description='Freeze independent accuracy ablations')
parser.add_argument('--output',type=Path,default=WORKSPACE/'controls/stage4_continuation_v7_development')
args=parser.parse_args()
receipt=freeze(args.output,[WORKSPACE/'configs'/value[0] for value in VARIANTS.values()])
print('Frozen accuracy source:',receipt['implementation']['sha256'],args.output.resolve())

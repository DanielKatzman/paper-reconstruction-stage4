from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE
from benchmark.snapshots import freeze
from benchmark.support_aware_experiments import VARIANTS
parser=argparse.ArgumentParser(description='Freeze separate v8 score and support-aware search experiments')
parser.add_argument('--output',type=Path,default=WORKSPACE/'controls/stage4_v8_support_aware_development_r2')
a=parser.parse_args()
r=freeze(a.output,[WORKSPACE/'configs'/name for name in VARIANTS.values()])
print('Frozen v8 source:',r['implementation']['sha256'],a.output.resolve())

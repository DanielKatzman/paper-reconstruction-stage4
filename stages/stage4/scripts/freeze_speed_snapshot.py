from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE
from benchmark.snapshots import freeze
parser=argparse.ArgumentParser(description='Freeze execution-only speed experiment')
parser.add_argument('--output',type=Path,default=WORKSPACE/'controls/stage4_speed_reuse_v1')
args=parser.parse_args()
receipt=freeze(args.output,[WORKSPACE/'configs/joint_speed_reuse_development.json'])
print('Frozen speed source:',receipt['implementation']['sha256'],args.output.resolve())

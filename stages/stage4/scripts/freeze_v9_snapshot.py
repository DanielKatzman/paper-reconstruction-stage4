from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE
from benchmark.snapshots import freeze
from benchmark.policy_retention_experiments import VARIANTS
p=argparse.ArgumentParser(description='Freeze the v9 cold-policy retention development source')
p.add_argument('--output',type=Path,default=WORKSPACE/'controls/stage4_policy_retention_v9_development')
a=p.parse_args();receipt=freeze(a.output,[WORKSPACE/'configs'/name for name in VARIANTS.values()])
print('Frozen v9 source:',receipt['implementation']['sha256'],a.output.resolve())

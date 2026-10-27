from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.snapshots import freeze
from benchmark.common import WORKSPACE
p=argparse.ArgumentParser(description="Freeze current consumer source and separate experiment configurations")
p.add_argument("--output",type=Path,default=WORKSPACE/"controls/stage4_quality_v5_combined_development")
a=p.parse_args()
names=["joint_pitch_geometry_development.json","joint_scale_coherence_development.json","joint_component_beam_development.json","joint_scale_component_beam_development.json"]
r=freeze(a.output,[WORKSPACE/"configs"/n for n in names])
print("Frozen development consumer:",r["implementation"]["sha256"],a.output.resolve())

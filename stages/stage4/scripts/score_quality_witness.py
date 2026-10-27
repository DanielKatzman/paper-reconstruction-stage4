from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.common import atomic_json
from evaluation.score_quality_witness import compare
p=argparse.ArgumentParser(description="Offline diagnostic only: rescore an existing reference-guided witness; never creates a serving seed")
p.add_argument("--integration",type=Path,required=True);p.add_argument("--case-id",required=True)
p.add_argument("--coverage",type=Path,required=True);p.add_argument("--config",type=Path,action="append",required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise ValueError("Choose a new diagnostic output")
target=a.output.resolve()
if target.is_relative_to(a.integration.resolve()) or target.is_relative_to(a.coverage.resolve().parent):
    raise ValueError("Offline diagnostic output must be separate from imported predictions and source diagnostics")
r=compare(a.integration,a.case_id,a.coverage,a.config);atomic_json(a.output,r)
for row in r['comparisons']:print(row['objective'],row['primary']['score'],row['reference_guided_existing_witness']['score'],'witness preferred:',row['witness_preferred'])

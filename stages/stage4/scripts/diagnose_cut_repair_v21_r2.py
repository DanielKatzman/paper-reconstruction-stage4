"""Use the exact frozen reader and bound controller for post-seal evaluation."""
import hashlib
import json
from pathlib import Path
import sys
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
request=json.loads((run/'request.json').read_text())
for p,h in request['controllers'].items():
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed V21 controller: '+p)
snapshot=Path(request['snapshot']);receipt=json.loads((snapshot/'receipt.json').read_text())
for name,h in receipt['files'].items():
    p=(snapshot/name).resolve()
    if not p.is_relative_to(snapshot) or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Changed V21 source: '+name)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(snapshot/'src'))
from benchmark.cut_repair_diagnosis_v21_r2 import main
raise SystemExit(main())

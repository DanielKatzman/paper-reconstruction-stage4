"""Authenticate the controller and frozen V16 reader before evaluation."""
import hashlib
import json
from pathlib import Path
import sys

run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
request=json.loads((run/'request.json').read_text())
for name,sha in request['controllers'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed V16 controller: '+name)
snapshot=Path(request['snapshot']);receipt=json.loads((snapshot/'receipt.json').read_text())
for name,sha in receipt['files'].items():
    path=(snapshot/name).resolve()
    if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise ValueError('Changed V16 snapshot: '+name)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(snapshot/'src'))
from benchmark.recovery_diagnosis_v16 import main
raise SystemExit(main())

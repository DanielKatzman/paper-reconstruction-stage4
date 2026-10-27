"""Authenticate the request/controllers before loading its frozen consumer."""
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[sys.argv.index('--run')+1]).resolve()
request = json.loads((root/'request.json').read_text())
payload = json.dumps({k:v for k,v in request.items() if k != 'request_sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
if hashlib.sha256(payload).hexdigest() != request['request_sha256']:
    raise ValueError('Changed experiment request')
for name, sha in request['controllers'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
        raise ValueError('Changed experiment controller: '+name)
snapshot = Path(request['snapshot']); receipt = json.loads((snapshot/'receipt.json').read_text())
for name, sha in receipt['files'].items():
    path = (snapshot/name).resolve()
    if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError('Changed frozen source: '+name)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(snapshot/'src'))
from benchmark.component_closure_experiment_v13 import diagnose
raise SystemExit(diagnose())

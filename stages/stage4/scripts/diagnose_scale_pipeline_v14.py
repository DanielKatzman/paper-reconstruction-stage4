"""Select the authenticated frozen reader before importing evaluation code."""
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
request = json.loads((run / 'request.json').read_text())
for name, sha in request['controllers'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
        raise ValueError('Changed pipeline controller: ' + name)
snapshot = Path(request['snapshot']); receipt = json.loads((snapshot / 'receipt.json').read_text())
for name, sha in receipt['files'].items():
    path = (snapshot / name).resolve()
    if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError('Changed frozen reader: ' + name)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(snapshot / 'src'))
from benchmark.scale_pipeline_diagnosis_v14 import main
raise SystemExit(main())

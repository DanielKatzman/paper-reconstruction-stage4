"""Select the result's frozen consumer before importing any consumer module."""
import hashlib
import json
from pathlib import Path
import sys


def select_core(run):
    run = Path(run).resolve()
    state = json.loads((run/'progress.json').read_text())
    if state['phase'] not in {'diagnostics','finished','diagnostics_failed'}:
        raise ValueError('Snapshot diagnosis serving barrier not reached')
    request = json.loads((run/'request.json').read_text())
    snapshot = Path(request['snapshot']).resolve()
    receipt = json.loads((snapshot/'receipt.json').read_text())
    for name,sha in receipt['files'].items():
        path = (snapshot/name).resolve()
        if not path.is_relative_to(snapshot) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Diagnostic source snapshot changed: '+name)
    if any(name == 'paper_reconstruction_stage4' or name.startswith('paper_reconstruction_stage4.') for name in sys.modules):
        raise ValueError('Frozen diagnosis must start in a fresh interpreter')
    sys.path.insert(0,str(snapshot/'src'))
    from paper_reconstruction_stage4.outputs import implementation_identity
    actual = implementation_identity()
    if actual != receipt['implementation'] or actual != request['implementation']:
        raise ValueError('Diagnostic implementation differs from sealed prediction')
    return actual

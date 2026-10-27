"""Portable orchestration helpers. No inference or evaluation imports."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def environment(stage):
    env = dict(os.environ)
    # Never inherit a source path that can silently import the old Desktop project.
    paths = [ROOT/'contracts/sdk/src', ROOT/'stages'/stage/'src', ROOT/'stages'/stage,
             ROOT/'stages'/stage/'scripts']
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in paths)
    env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONUNBUFFERED'] = '1'
    return env


def python(stage):
    config = ROOT/'local/runtimes.json'
    return str(read(config).get(stage, sys.executable)) if config.exists() else sys.executable


def execute(stage, argv, output, label):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log = output/(label+'.log'); start = time.monotonic()
    command = [python(stage), '-B', *map(str, argv)]
    print(f'{label}: starting; log: {log}', flush=True)
    with log.open('wb') as stream:
        p = subprocess.Popen(command, cwd=ROOT, env=environment(stage), stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = p.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    print(f'{label}: working +{time.monotonic()-start:.0f}s; log: {log}', flush=True)
        except BaseException:
            p.terminate(); p.wait(); raise
    row = {'stage': stage, 'command': command, 'elapsed_seconds': time.monotonic()-start,
           'returncode': code, 'log_sha256': sha(log)}
    save(output/(label+'.receipt.json'), row)
    if code: raise RuntimeError(f'{label} failed ({code}); see {log}')
    print(f'{label}: finished in {row["elapsed_seconds"]:.1f}s', flush=True)
    return row

"""Hydrate private local assets or authenticate a supplied asset directory."""
import argparse
from pathlib import Path
import shutil
from common import ROOT, read, sha


def run(mode, groups):
    manifest = read(ROOT/'provenance/assets.json')['files']
    sources = {x['path']: x for x in read(ROOT/'local/assets-sources.json')['files']} if mode == 'hydrate' else {}
    rows = [x for x in manifest if not groups or x['group'] in groups]
    for i, row in enumerate(rows, 1):
        target = (ROOT/row['path']).resolve()
        if not target.is_relative_to(ROOT): raise ValueError('Asset escapes repository')
        if not target.exists() and mode == 'hydrate':
            source = Path(sources[row['path']]['source'])
            if source.stat().st_size != row['size'] or sha(source) != row['sha256']:
                raise ValueError('Source asset differs: '+str(source))
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if not target.is_file() or target.stat().st_size != row['size'] or sha(target) != row['sha256']:
            raise ValueError('Missing/changed asset: '+row['path'])
        if i % 100 == 0: print(f'Assets verified: {i}/{len(rows)}', flush=True)
    print(f'{mode}: verified {len(rows)} files', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['hydrate', 'verify'])
    p.add_argument('--groups', nargs='+', choices=['models', 'bundles', 'inputs', 'references', 'predictions'])
    a = p.parse_args(); run(a.mode, a.groups)

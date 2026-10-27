"""Verify the pinned release component versions, source inventory and model assets."""
import tomllib
from common import ROOT,read,sha


def verify():
    manifest=read(ROOT/'compatibility.json')
    if manifest['release_version']!=(ROOT/'VERSION').read_text().strip():raise ValueError('Release version differs')
    if sha(ROOT/'provenance/components.json')!=manifest['component_inventory_sha256']:raise ValueError('Component inventory differs')
    for name,stage in manifest['stages'].items():
        package=tomllib.loads((ROOT/stage['package_metadata']).read_text(encoding='utf-8'))
        if package['project']['version']!=stage['software_version']:raise ValueError('Component version differs: '+name)
        for model in stage['models']:
            path=ROOT/model['path']
            if not path.is_file() or path.stat().st_size!=model['size'] or sha(path)!=model['sha256']:
                raise ValueError('Model missing/changed; run git lfs pull: '+model['path'])
    print('Release compatibility verified: four pinned stages and four frozen model files.',flush=True)


if __name__=='__main__':verify()

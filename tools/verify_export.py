"""Seal/check the Git-eligible source inventory, excluding local assets and outputs."""
import argparse
from pathlib import Path
import re
import subprocess
from common import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',action='store_true');a=p.parse_args()
    raw=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z'])
    names=sorted(set(n for n in raw.decode('utf-8').split('\0') if n and n!='provenance/EXPORT_MANIFEST.json'))
    files={};total=0
    for name in names:
        path=ROOT/name
        if name.startswith(('assets/','outputs/','local/')) or path.suffix in {'.pt','.onnx','.pth','.pyc','.pem','.key'}:
            raise ValueError('Private/generated asset is Git-eligible: '+name)
        if path.stat().st_size>10*1024*1024:raise ValueError('Unexpected large source file: '+name)
        if path.suffix in {'.py','.json','.md','.toml','.txt','.yml','.ps1'}:
            text=path.read_text(encoding='utf-8-sig')
            if re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):
                raise ValueError('Possible embedded credential: '+name)
        files[name]=sha(path);total+=path.stat().st_size
    target=ROOT/'provenance/EXPORT_MANIFEST.json'
    if a.seal:save(target,{'release':'2.0.0','files':files,'source_bytes':total})
    elif read(target)['files']!=files:raise ValueError('Git-eligible source inventory differs from export seal')
    print(f'Export verified: {len(files)} Git-eligible files, {total/1e6:.2f} MB; no local assets/weights/outputs included.')


if __name__=='__main__':main()

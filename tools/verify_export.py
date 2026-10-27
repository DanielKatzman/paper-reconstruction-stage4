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
    model_rows={r['path']:r for r in read(ROOT/'provenance/assets.json')['files'] if r['group']=='models'}
    files={};total=0;lfs_bytes=0
    for name in names:
        path=ROOT/name
        is_model=name in model_rows
        if name.startswith(('assets/','outputs/','local/')) or (path.suffix in {'.pt','.onnx','.pth','.pyc','.pem','.key'} and not is_model):
            raise ValueError('Private/generated asset is Git-eligible: '+name)
        if is_model:
            row=model_rows[name]
            if path.stat().st_size!=row['size'] or sha(path)!=row['sha256']:raise ValueError('Frozen model differs: '+name)
            attribute=subprocess.check_output(['git','-C',str(ROOT),'check-attr','filter','--',name],text=True).strip()
            if not attribute.endswith(': lfs'):raise ValueError('Frozen model must use Git LFS: '+name)
            lfs_bytes+=path.stat().st_size
        elif path.stat().st_size>10*1024*1024:raise ValueError('Unexpected large source file: '+name)
        if path.suffix in {'.py','.json','.md','.toml','.txt','.yml','.ps1'}:
            text=path.read_text(encoding='utf-8-sig')
            if re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):
                raise ValueError('Possible embedded credential: '+name)
        files[name]=sha(path);total+=path.stat().st_size
    target=ROOT/'provenance/EXPORT_MANIFEST.json'
    if a.seal:save(target,{'release':'2.0.0','files':files,'source_bytes':total-lfs_bytes,'git_lfs_model_bytes':lfs_bytes})
    elif read(target)['files']!=files:raise ValueError('Git-eligible source inventory differs from export seal')
    print(f'Export verified: {len(files)} files, {(total-lfs_bytes)/1e6:.2f} MB source plus {lfs_bytes/1e6:.2f} MB of hash-verified LFS models; local data/outputs excluded.')


if __name__=='__main__':main()

"""Create optional hash-bound GitHub Release assets, outside the source tree tracked by Git."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile
from common import ROOT,read,sha,save


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'local/release-assets')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);receipts=[]
    for name,groups in [('models-v2.0.0',{'models'}),('benchmark-v2.0.0',{'bundles','inputs','references'})]:
        target=a.output/(name+'.zip')
        if target.exists():raise ValueError('Archive already exists: '+str(target))
        rows=[r for r in read(ROOT/'provenance/assets.json')['files'] if r['group'] in groups]
        with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=3) as z:
            for i,row in enumerate(rows,1):
                source=ROOT/row['path']
                if sha(source)!=row['sha256']:raise ValueError('Asset differs: '+row['path'])
                z.write(source,row['path'])
                if i%100==0:print(name,f'{i}/{len(rows)}',flush=True)
        receipts.append({'name':target.name,'sha256':sha(target),'bytes':target.stat().st_size,'files':len(rows)})
        print(name,'ready',flush=True)
    save(a.output/'checksums.json',{'release':'2.0.0','archives':receipts})

"""Immutable local consumer snapshots for repeatable opt-in experiments."""
from pathlib import Path
import shutil
from uuid import uuid4
from .common import WORKSPACE,atomic_json,file_hash,read_json
from .qualified_v3 import require
from paper_reconstruction_stage4.outputs import implementation_identity


def verify_snapshot(root):
    root=Path(root).resolve();receipt=read_json(root/"receipt.json")
    for name,sha in receipt["files"].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root) and file_hash(path)==sha,"Experiment snapshot changed: "+name)
    return receipt


def freeze(root,configs):
    root=Path(root).resolve();parent=(WORKSPACE/"controls").resolve()
    require(root!=parent and root.is_relative_to(parent),"New source snapshot must be inside this workspace's controls")
    if root.exists():
        receipt=verify_snapshot(root)
        require(receipt["implementation"]==implementation_identity(),"Existing snapshot differs from current source; choose a new name")
        for config in configs:
            require(receipt["files"].get("configs/"+Path(config).name)==file_hash(Path(config)),"Existing snapshot configuration differs")
        return receipt
    paths=sorted(p for p in (WORKSPACE/"src/paper_reconstruction_stage4").rglob("*") if p.is_file() and p.suffix in {".py",".json"})
    paths+=list(map(Path,configs))
    root.parent.mkdir(parents=True,exist_ok=True)
    staging=root.with_name(root.name+".building_"+uuid4().hex[:10]);staging.mkdir()
    files={}
    for path in paths:
        relative=path.relative_to(WORKSPACE)
        target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        files[relative.as_posix()]=file_hash(target)
    receipt={"files":files,"implementation":implementation_identity(),"status":"development_experiments_not_qualified",
             "purpose":"Separate geometry, scale-coherence and component-beam ablations; historical controls preserved"}
    atomic_json(staging/"receipt.json",receipt)
    require(staging.resolve().parent==root.parent and root.is_relative_to(parent),"Snapshot rename escaped controls")
    staging.rename(root)
    return receipt

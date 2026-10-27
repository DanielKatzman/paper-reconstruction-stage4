"""Recover targeted rotation modes from authenticated, already supplied evidence."""
import time
from collections import defaultdict
import numpy as np
from paper_reconstruction_contracts import load_bundle
from .constraints import require_native_rotation
from .seam_poses_rigid import angle_modes


def modes(problem, pairs, deadline):
    started=time.perf_counter();wanted={tuple(sorted(p)) for p in pairs}
    existing=angle_modes(problem);extra=defaultdict(list);count=0;complete=False
    bundle=load_bundle(problem.root,full=False)
    if bundle.manifest['manifest_sha256']!=problem.manifest['manifest_sha256']:
        raise ValueError('Archive rotation recovery input identity changed')
    for row in bundle.hypotheses():
        if time.perf_counter()>=deadline:break
        count+=1;a,b=row['piece_a'],row['piece_b'];h=np.asarray(row['transform_b_to_a'],float)
        if a>b:a,b=b,a;h=np.linalg.inv(h)
        if (a,b) not in wanted:continue
        require_native_rotation(h,problem.capture_metadata)
        angle=float(np.arctan2(h[1,0],h[0,0]));prior=[*existing[a,b],*extra[a,b]]
        if any(abs(np.arctan2(np.sin(angle-v['angle']),np.cos(angle-v['angle'])))<np.deg2rad(.25) for v in prior):continue
        extra[a,b].append({'angle':angle,'source_geometry_id':row['hypothesis_id'],'source':'authenticated_archive'})
    else:complete=True
    # Never consume an unexhausted/unvalidated replacement archive.
    if not complete:extra.clear()
    result=dict(extra)
    return result,{'policy':'targeted-existing-archive-rotation-recovery.v1',
        'input_manifest_sha256':problem.manifest['manifest_sha256'],'target_pairs':[list(p) for p in sorted(wanted)],
        'archive_rows_scanned':count,'archive_validation_exhausted':complete,
        'additional_angle_modes':sum(map(len,result.values())),'angular_bin_degrees':.25,
        'translation_inherited':False,'upstream_archive_modified':False,'reference_coordinates_used':False,
        'budget_abstained':not complete,'elapsed_seconds':time.perf_counter()-started}

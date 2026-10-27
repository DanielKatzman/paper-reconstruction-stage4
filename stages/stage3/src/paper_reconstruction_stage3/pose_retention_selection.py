"""Observable portfolios with explicit budget/duplicate reasons; no label inputs."""
from collections import defaultdict,deque
import numpy as np
from .alignment import apply_transform
from .pose_correction_geometry import diversity_key

REASONS=('not_applicable','retained','transform_duplicate','budget_limit','warp_rejected','collision_rejected')


def portfolio(values,limit,probes,*,score,secondary,distance=.25):
    """Alternate fit/support orderings across overlap and coarse/fine buckets."""
    unique={}
    for p in values:
        token=p['proposal_id'];old=unique.get(token)
        if old is None or (-score(p),-secondary(p),str(p.get('_parameters')))<(-score(old),-secondary(old),str(old.get('_parameters'))):unique[token]=p
    buckets=defaultdict(list)
    for p in unique.values():buckets[(p.get('_selection_origin','coarse'),*diversity_key(p))].append(p)
    queues={}
    for key,rows in buckets.items():
        queues[key]=[deque(sorted(rows,key=lambda p:(-fn(p),p['proposal_id']))) for fn in (score,secondary)]
    selected=[];seen=set();mapped=[];reasons={p:'budget_limit' for p in unique};turn=0
    while queues and len(selected)<limit:
        progressed=False
        # Every bucket gets a turn before another item from the same bucket.
        for key in sorted(list(queues)):
            alternatives=queues[key];queue=alternatives[turn%2]
            while queue and queue[0]['proposal_id'] in seen:queue.popleft()
            if not queue:
                other=alternatives[(turn+1)%2]
                while other and other[0]['proposal_id'] in seen:other.popleft()
                if not other:del queues[key];continue
                queue=other
            p=queue.popleft();token=p['proposal_id'];seen.add(token);progressed=True
            xy=apply_transform(probes,p['alignment']['transform'])
            duplicate=bool(mapped) and bool(np.any(np.max(np.linalg.norm(np.asarray(mapped)-xy,axis=2),axis=1)<distance))
            if duplicate:reasons[token]='transform_duplicate';continue
            selected.append(p);mapped.append(xy);reasons[token]='retained'
            if len(selected)>=limit:break
        if not progressed:break
        turn+=1
    return selected,reasons


def checked_survivors(values,warp,collision):
    """Check the complete bounded prefilter; a survivor count never ends the scan."""
    out=[];reasons={}
    for p in values:
        token=p['proposal_id']
        if not warp(p):reasons[token]='warp_rejected';continue
        if not collision(p):reasons[token]='collision_rejected';continue
        out.append(p);reasons[token]='retained'
    return out,reasons

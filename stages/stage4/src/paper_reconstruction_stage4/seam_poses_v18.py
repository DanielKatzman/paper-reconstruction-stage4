"""Budgeted pair/rotation coverage before deeper native-contour modes.

Measurements, support checks and per-pair retention are inherited unchanged.
Only work ordering changes. Source partitions guide priority without providing
new pose seeds; no reference coordinates or confidence labels are accessed.
"""
from collections import deque
from itertools import combinations
import time
import numpy as np
from .outputs import digest
from .native_bridge import pair_options,arc_value
from .seam_poses import translation_modes,refine_translation,spread

POLICY='native-contour-quarter-votes-geometry.v3'
MEASUREMENT_POLICY='native-contour-quarter-votes.v1'


def prepare_modes(data,a,b,peaks,rotation_cache=None):
    """Preserve the V14 modes, including separately retrieved curved basins."""
    _,_,sa,na=data[a];xb,nb,sb,snb=data[b];rotations=[]
    for quarter in range(4):
        key=b,quarter
        cached=rotation_cache.get(key) if rotation_cache is not None else None
        if cached is None:
            theta=quarter*np.pi/2.;c,s=np.cos(theta),np.sin(theta);r=np.array([[c,-s],[s,c]])
            cached=r,xb@r.T,nb@r.T,sb@r.T,snb@r.T
            if rotation_cache is not None:rotation_cache[key]=cached
        r,rb,rn,rsb,rsnb=cached
        general=translation_modes(sa,rsb,na,rsnb,peaks)
        curved=[(t,v) for t,v in translation_modes(sa,rsb,na,rsnb,2,curved_only=True)
                if not any(np.linalg.norm(t-old[0])<6. for old in general)]
        # Curved de-duplication must also compare earlier curved modes, as V14 does.
        distinct=[]
        for t,v in curved:
            if not any(np.linalg.norm(t-old[0])<6. for old in distinct):distinct.append((t,v))
        rotations.append({'quarter':quarter,'r':r,'rb':rb,'rn':rn,
                          'general':general,'curved':distinct})
    return rotations


def task(rotation,channel,index):
    t,votes=rotation[channel][index]
    return (rotation,channel,index,t,votes)


def queues(rotations,peaks):
    foundation=[task(r,'general',0) for r in rotations if r['general']]
    foundation.sort(key=lambda t:(-t[4],t[0]['quarter']))
    expansion=[]
    strata=[('curved',0),('general',1),('curved',1),*[('general',i) for i in range(2,peaks)]]
    for channel,index in strata:
        rows=[task(r,channel,index) for r in rotations if len(r[channel])>index]
        rows.sort(key=lambda t:(-t[4],t[0]['quarter']))
        expansion.extend(rows)
    return deque(foundation),deque(expansion)


def fit_task(problem,config,physical,data,a,b,value):
    rotation,_,_,translation,votes=value;xa,na,_,_=data[a]
    h=np.eye(3);h[:2,:2]=rotation['r']
    h[:2,2]=refine_translation(xa,rotation['rb'],na,rotation['rn'],translation)
    options=pair_options(problem,a,b,{a:np.eye(3),b:h},config,[])
    rows=[]
    for arcs,metrics in options:
        length=min(v.length for v in arcs);condition=spread(arcs)
        if length<64. or not physical.pair(a,b,h,final=False)['valid']:continue
        score=length*(.1+min(condition,.35))/(1.+metrics['bidirectional_contact_p95_native_px']/4.)
        # Scheduling does not change the measurement or its stable identity.
        row={'policy':MEASUREMENT_POLICY,'pieces':[a,b],'quarter_turn':rotation['quarter'],
            'transform_b_to_a':h.tolist(),'arcs':[arc_value(v) for v in arcs],
            'native_length_px':length,'shape_spread_ratio':condition,'conditioned':condition>=.05,
            'native_metrics':metrics,'score':float(score),'vote_strength':votes}
        row['proposal_id']=digest(row);rows.append(row)
    return rows


def retain(rows):
    kept=[]
    for row in sorted(rows,key=lambda v:(not v['conditioned'],-v['score'],v['proposal_id'])):
        h=np.asarray(row['transform_b_to_a'])
        if any(np.max(np.abs(h-np.asarray(old['transform_b_to_a'])))<4. for old in kept):continue
        kept.append(row)
        # Every distinct fitted mode remains available to geometry search.
    return kept


def frontier_priority(pairs,frontiers,ids):
    maps=[]
    for groups in frontiers:
        mapping={p:i for i,group in enumerate(groups) for p in group}
        if set(mapping)!=set(ids) or sum(map(len,groups))!=len(ids):
            raise ValueError('Native frontier must partition the observed pieces')
        maps.append(mapping)
    return {pair:sum(m[a]!=m[b] for m in maps) for pair in pairs for a,b in [pair]}


def generate(problem,config,deadline,maximum_checks=None,samples=256,peaks=4,frontiers=(),
             _rotation_provider=None,_fit_provider=None,_retain_provider=None):
    family=(problem.capture_metadata or {}).get('rotation_family')
    if family=='continuous_rigid' and _rotation_provider is None:
        from .seam_poses_rigid import generate as rigid_generate
        return rigid_generate(problem,config,deadline,maximum_checks,samples,peaks,frontiers)
    if family!='quarter_turn' and not (family=='continuous_rigid' and _rotation_provider is not None):
        raise ValueError('Quarter-turn native alignment requires authenticated capture metadata')
    prepare=_rotation_provider or prepare_modes;fit=_fit_provider or fit_task;retainer=_retain_provider or retain
    if maximum_checks is not None and (type(maximum_checks) is not int or maximum_checks<0):raise ValueError('Invalid native batch size')
    started=time.perf_counter();data={};ids=sorted(problem.pieces);pairs=list(combinations(ids,2))
    priority=frontier_priority(pairs,frontiers,ids)
    for pid in ids:
        piece=problem.pieces[pid];xy=np.asarray(piece.contour['points_xy'],float)
        norm=np.asarray(piece.contour['inward_normal_xy'],float)
        norm=norm/np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-12)
        picks=np.linspace(0,len(xy)-1,min(samples,len(xy))).astype(int)
        data[pid]=(xy,norm,xy[picks],norm[picks])
    from .feasibility import Feasibility
    physical=Feasibility(problem,config);blocks={};checks=0;reasons=set();rotation_cache={}
    audit={'policy':POLICY,'measurement_policy':MEASUREMENT_POLICY,'samples_per_piece':samples,
        'peaks_per_rotation':peaks,'additional_curved_peaks_per_rotation':2,
        'maximum_native_checks':maximum_checks,'evaluation_truth_opened':False,'learned_confidence_claimed':False,
        'schedule':'first-general-mode-each-rotation-then-pair-round-robin-curved-and-general.v1',
        'proposal_count_is_batch_size_only': True, 'retained_modes_per_pair_limit': None,
        'frontier_source_count':len(frontiers),'native_fit_measurement_changed':False}

    def available():
        if maximum_checks is not None and checks>=maximum_checks:reasons.add('native_check_limit');return False
        if time.perf_counter()>=deadline:reasons.add('generation_deadline');return False
        return True

    def consume(pair,value):
        nonlocal checks
        block=blocks[pair];checks+=1;block['checks']+=1
        block['rotations'].add(value[0]['quarter'])
        block['channels'][value[1]]=block['channels'].get(value[1],0)+1
        block['candidates'].extend(fit(problem,config,physical,data,*pair,value))

    ordered=sorted(pairs,key=lambda p:(-priority[p],p))
    # Prepare one pair at a time and fit its first rotation before preparing
    # another pair. Expensive vote work is never repeated in later rounds.
    for pair in ordered:
        if not available():break
        rotations=prepare(data,*pair,peaks,rotation_cache);foundation,expansion=queues(rotations,peaks)
        blocks[pair]={'foundation':foundation,'expansion':expansion,'candidates':[],
                      'checks':0,'rotations':set(),'channels':{},'configured':len(foundation)+len(expansion),'rotation_count':len(rotations)}
        if foundation:
            if not available():break
            consume(pair,foundation.popleft())

    def round_robin(channel):
        while True:
            active=[p for p in ordered if p in blocks and blocks[p][channel]]
            if not active:return True
            # One fit per active pair per round; geometry guides only priority.
            active.sort(key=lambda p:(-priority[p],-blocks[p][channel][0][4],p))
            for pair in active:
                if not available():return False
                consume(pair,blocks[pair][channel].popleft())

    prepared=len(blocks)==len(pairs)
    if prepared and round_robin('foundation'):round_robin('expansion')
    remaining=sum(len(b['foundation'])+len(b['expansion']) for b in blocks.values())
    complete=prepared and remaining==0
    if complete:reasons.clear()
    results=[];pair_rows=[]
    for pair in pairs:
        block=blocks.get(pair);kept=retainer(block['candidates']) if block else []
        results.extend(kept)
        pair_rows.append({'pieces':list(pair),'retained':len(kept),'conditioned_retained':sum(v['conditioned'] for v in kept),
            'native_checks':block['checks'] if block else 0,'tested_rotations':len(block['rotations']) if block else 0,
            'fit_channels':block['channels'] if block else {},'vote_work_prepared':block is not None,
            'configured_fit_count':block['configured'] if block else None,
            'all_configured_work_completed':bool(block is not None and not block['foundation'] and not block['expansion']),
            'frontier_source_count':priority[pair]})
    per_piece={p:{'possible_pairs':len(ids)-1,'fitted_pairs':sum(row['native_checks']>0 and p in row['pieces'] for row in pair_rows),
        'native_checks':sum(row['native_checks'] for row in pair_rows if p in row['pieces'])} for p in ids}
    frontier_rows=[]
    for index,groups in enumerate(frontiers):
        for group in groups:
            members=set(group);rows=[r for r in pair_rows if (r['pieces'][0] in members)!=(r['pieces'][1] in members)]
            frontier_rows.append({'source_index':index,'component':list(group),'possible_cross_pairs':len(rows),
                'fitted_cross_pairs':sum(r['native_checks']>0 for r in rows),
                'native_checks':sum(r['native_checks'] for r in rows)})
    audit.update(native_checks=checks,pair_rotations=sum(r['tested_rotations'] for r in pair_rows),
        vote_rotations_prepared=sum(b['rotation_count'] for b in blocks.values()),pairs=pair_rows,possible_pair_count=len(pairs),
        untested_pair_count=len(pairs)-len(blocks),
        pairs_without_fitted_checks=sum(r['native_checks']==0 for r in pair_rows),
        pairs_without_modes=sum(b['configured']==0 for b in blocks.values()),
        pairs_with_all_rotations_fitted=sum(r['tested_rotations']==blocks.get(tuple(r['pieces']),{}).get('rotation_count',4) for r in pair_rows),
        preparation_completed=prepared,all_configured_work_completed=complete,
        unperformed_known_fits=remaining,budget_abstained=not complete,budget_reason_codes=sorted(reasons),
        per_piece_coverage=per_piece,frontier_coverage=frontier_rows,
        retained_poses=len(results),rotation_buffers=len(rotation_cache),
        rotation_buffer_bytes=sum(a.nbytes for arrays in rotation_cache.values() for a in arrays),
        elapsed_seconds=time.perf_counter()-started)
    return results,audit

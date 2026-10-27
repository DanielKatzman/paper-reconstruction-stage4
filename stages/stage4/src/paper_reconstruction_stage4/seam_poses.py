"""Bounded full-native-contour translation votes and rigid seam verification.

The quarter-turn route is opt-in through authenticated capture metadata. Scores
are observed geometric heuristics, never learned confidence or truth labels.
"""
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import map_points
from .outputs import digest
from .native_bridge import pair_options,arc_value


def spread(arcs):
    values=[]
    for arc in arcs:
        singular=np.linalg.svd(arc.points-arc.points.mean(axis=0),compute_uv=False)
        values.append(float(singular[-1]/max(singular[0],1e-9)))
    return min(values)


def translation_modes(first,second,normal_a,normal_b,peaks=4,step=4.,curved_only=False):
    opposed=(normal_a@normal_b.T)<-.8
    if curved_only:
        va=np.linalg.norm(normal_a-np.roll(normal_a,4,axis=0),axis=1)+np.linalg.norm(normal_a-np.roll(normal_a,-4,axis=0),axis=1)
        vb=np.linalg.norm(normal_b-np.roll(normal_b,4,axis=0),axis=1)+np.linalg.norm(normal_b-np.roll(normal_b,-4,axis=0),axis=1)
        opposed &= (va[:,None]>.15)&(vb[None,:]>.15)
    ia,ib=np.nonzero(opposed)
    if len(ia)<8:return []
    delta=first[ia]-second[ib];lo=np.floor(delta.min(axis=0)/step)*step-step
    bins=np.floor((delta-lo)/step).astype(int)
    size=bins.max(axis=0)+2
    votes=np.bincount(bins[:,1]*size[0]+bins[:,0],minlength=int(np.prod(size))).reshape(size[::-1])
    score=gaussian_filter(votes.astype(float),.8);rows=[]
    for _ in range(peaks):
        index=int(np.argmax(score));strength=float(score.flat[index])
        if strength<=0:break
        y,x=np.unravel_index(index,score.shape)
        rows.append((lo+(np.array([x,y])+.5)*step,strength))
        score[max(0,y-2):y+3,max(0,x-2):x+3]=0.
    return rows


def refine_translation(xa,xb,na,nb,translation):
    tree=cKDTree(xa);t=translation.copy()
    for _ in range(8):
        dist,nearest=tree.query(xb+t)
        good=(dist<=16.)&(np.sum(nb*na[nearest],axis=1)<-.8)
        indices=np.flatnonzero(good)
        if len(indices)<12:break
        keep=indices[dist[indices]<=np.quantile(dist[indices],.8)+.5]
        move=np.median(xa[nearest[keep]]-xb[keep]-t,axis=0)
        move=np.clip(move,-4.,4.);t+=move
        if np.linalg.norm(move)<.03:break
    return t


def generate(problem,config,deadline,maximum_checks=512,samples=256,peaks=4):
    """All physical pairs, bounded votes/fits, per-pair diverse pose retention."""
    if not problem.capture_metadata or problem.capture_metadata.get('rotation_family')!='quarter_turn':
        raise ValueError('Quarter-turn native alignment requires authenticated capture metadata')
    started=time.perf_counter();data={};audit={'policy':'native-contour-quarter-votes.v1',
        'samples_per_piece':samples,'peaks_per_rotation':peaks,'additional_curved_peaks_per_rotation':2,'maximum_native_checks':maximum_checks,
        'native_checks':0,'pair_rotations':0,'pairs':[],'budget_abstained':False,
        'evaluation_truth_opened':False,'learned_confidence_claimed':False}
    for pid,piece in problem.pieces.items():
        xy=np.asarray(piece.contour['points_xy'],float);norm=np.asarray(piece.contour['inward_normal_xy'],float)
        norm=norm/np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-12)
        picks=np.linspace(0,len(xy)-1,min(samples,len(xy))).astype(int)
        data[pid]=(xy,norm,xy[picks],norm[picks])
    from .feasibility import Feasibility
    physical=Feasibility(problem,config);results=[]
    ids=sorted(data)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if time.perf_counter()>=deadline or audit['native_checks']>=maximum_checks:
                audit['budget_abstained']=True;break
            xa,na,sa,sna=data[a];xb,nb,sb,snb=data[b];candidates=[]
            pair_checks=audit['native_checks'];pair_rotations=audit['pair_rotations']
            for quarter in range(4):
                if time.perf_counter()>=deadline or audit['native_checks']>=maximum_checks:
                    audit['budget_abstained']=True;break
                theta=quarter*np.pi/2.;c,s=np.cos(theta),np.sin(theta)
                r=np.array([[c,-s],[s,c]]);rb=xb@r.T;rn=nb@r.T
                modes=translation_modes(sa,sb@r.T,sna,snb@r.T,peaks)
                # A separate curved vote channel prevents long flat outer
                # borders consuming every translation peak.
                for t,votes in translation_modes(sa,sb@r.T,sna,snb@r.T,2,curved_only=True):
                    if not any(np.linalg.norm(t-old[0])<6. for old in modes):modes.append((t,votes))
                audit['pair_rotations']+=1
                for t,votes in modes:
                    if time.perf_counter()>=deadline or audit['native_checks']>=maximum_checks:
                        audit['budget_abstained']=True;break
                    audit['native_checks']+=1
                    h=np.eye(3);h[:2,:2]=r;h[:2,2]=refine_translation(xa,rb,na,rn,t)
                    poses={a:np.eye(3),b:h}
                    options=pair_options(problem,a,b,poses,config,[])
                    if not options:continue
                    for arcs,metrics in options:
                        length=min(v.length for v in arcs);condition=spread(arcs)
                        if length<64.:continue
                        check=physical.pair(a,b,h,final=False)
                        if not check['valid']:continue
                        # Length and curvature conditioning rank proposals;
                        # straight contacts remain explicit alternatives.
                        score=length*(.1+min(condition,.35))/(1.+metrics['bidirectional_contact_p95_native_px']/4.)
                        record={'policy':audit['policy'],'pieces':[a,b],'quarter_turn':quarter,
                            'transform_b_to_a':h.tolist(),'arcs':[arc_value(v) for v in arcs],
                            'native_length_px':length,'shape_spread_ratio':condition,
                            'conditioned':condition>=.05,'native_metrics':metrics,
                            'score':float(score),'vote_strength':votes}
                        record['proposal_id']=digest(record)
                        candidates.append(record)
            # Retain several basins; duplicate arc fits are not independent.
            candidates.sort(key=lambda v:(not v['conditioned'],-v['score'],v['proposal_id']))
            kept=[]
            for row in candidates:
                h=np.asarray(row['transform_b_to_a'])
                if any(np.max(np.abs(h-np.asarray(old['transform_b_to_a'])))<4. for old in kept):continue
                kept.append(row)
                if len(kept)==4:break
            results.extend(kept);audit['pairs'].append({'pieces':[a,b],'retained':len(kept),
                'conditioned_retained':sum(v['conditioned'] for v in kept),
                'native_checks':audit['native_checks']-pair_checks,
                'tested_rotations':audit['pair_rotations']-pair_rotations,
                'all_configured_work_completed':not audit['budget_abstained']})
        if audit['budget_abstained']:break
    audit.update(retained_poses=len(results),elapsed_seconds=time.perf_counter()-started,
                 untested_pair_count=len(ids)*(len(ids)-1)//2-len(audit['pairs']))
    return results,audit

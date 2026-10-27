"""Bounded, observed-only two-component placement and native seam closure.

Extent proposals are hypotheses, not a proof that other placements do not
exist. Shared scale and complete exact masks are required by the bound profile.
"""
import time
from dataclasses import replace
import numpy as np
from paper_reconstruction_contracts import map_points
from .ambiguity import AmbiguityContext
from .config import CLOSURE_POLICY
from .constraints import require_bound, require_rigid
from .geometry import regauge
from .outputs import digest
from .page import enclosing_rectangle, THRESHOLDS
from .page_geometry import fingerprint
from .search.execution import clone
from .native_bridge import pair_options, make_record, edge_from_record, MINIMUM_PAIRS
from .support import overlaps


def normalized_group(context,state,group,quarter):
    cloud=np.vstack([map_points(context.hulls[p],state.poses[p]) for p in group])
    axes=enclosing_rectangle(cloud)[1]
    theta=quarter*np.pi/2.;c,s=np.cos(theta),np.sin(theta)
    h=np.eye(3);h[:2,:2]=np.array([[c,-s],[s,c]])@axes.T
    xy=map_points(cloud,h);lo=xy.min(axis=0)-.5;hi=xy.max(axis=0)+.5
    h[:2,2]=-lo
    return h,hi-lo


def placement_proposals(context, source):
    """Finite boundary-placement hypotheses; internal relative poses preserved."""
    groups=sorted(source.components(),key=lambda g:(-len(g),g))
    if len(groups)!=2:return
    total=sum(context.areas.values());main,wh_main=normalized_group(context,source,groups[0],0)
    seen=set()
    for quarter in range(4):
        other,wh_other=normalized_group(context,source,groups[1],quarter)
        width=max(wh_main[0],wh_other[0]);height=max(wh_main[1],wh_other[1])
        for wh in (np.array([width,total/width]),np.array([total/height,height])):
            rm,ro=wh-wh_main,wh-wh_other
            if min(*rm,*ro)<-1e-6:continue
            for xm in np.unique([0.,max(0.,rm[0])]):
                for ym in np.unique([0.,max(0.,rm[1])]):
                    for xo in np.unique([0.,max(0.,ro[0])]):
                        for yo in np.unique([0.,max(0.,ro[1])]):
                            poses={}
                            for group,h,x,y in ((groups[0],main,xm,ym),(groups[1],other,xo,yo)):
                                t=np.eye(3);t[:2,2]=[x,y]
                                poses.update({pid:t@h@source.poses[pid] for pid in group})
                            # A single gauge makes identity and pair evidence
                            # independent of the arbitrary diagnostic frame.
                            poses=regauge(poses,set(poses))
                            key=digest({pid:h.tolist() for pid,h in sorted(poses.items())})
                            if key in seen:continue
                            seen.add(key)
                            state=clone(source);state.poses=poses
                            yield key,state,{'quarter_turn':quarter,'extent_wh_native_px':wh.tolist(),
                                             'source_groups':groups}


def shape_measurement(context,state,side):
    f=context.raster(state,side);ratio=f['rectangle_area_px2']/sum(context.areas.values())
    passed=(f['fill_fraction']>=THRESHOLDS['minimum_fill'] and
            f['overlap_fraction']<=THRESHOLDS['maximum_overlap'] and
            f['interior_hole_fraction']<=THRESHOLDS['maximum_holes'] and
            ratio<=1.+context.config.page_component_area_slack)
    return {**f,'area_ratio':ratio,'shape_gate_pass':bool(passed),
        'score':float(1.-.65*f['unfilled_fraction']-.20*f['overlap_fraction']-.15*f['interior_hole_fraction'])}


class ClosureContext(AmbiguityContext):
    def __init__(self,problem,config,support):
        require_bound(problem,config)
        super().__init__(problem,config,support)
        self.closure_audit={'policy':CLOSURE_POLICY,'source_limit':config.closure_source_limit,
            'proposal_limit':config.closure_proposal_limit,'fine_limit':config.closure_fine_limit,
            'seconds_limit':config.closure_seconds,'sources':[],'proposals':[],
            'generated_proposals':0,'fine_checks':0,'verified_closures':0,
            'minimum_distinct_cross_pairs':MINIMUM_PAIRS,
            'source_component_internal_poses_preserved':True,
            'new_relative_poses_generated_from_observed_shapes':True,
            'upstream_bundle_or_occurrences_modified':False,'evaluation_truth_opened':False,
            'search_is_exhaustive':False,'budget_abstained':False}
        self.candidate_audit['component_closure_policy']=CLOSURE_POLICY

    def verify_placement(self,source,proposal,proposal_id,feasibility):
        """A page outline is never itself a connecting edge."""
        groups=source.components();occupied=[]
        for eid in source.selected:occupied.extend(self.support.arcs_for(source,eid))
        options=[]
        for a in groups[0]:
            for b in groups[1]:
                first,second=sorted((a,b))
                for arcs,metrics in pair_options(self.problem,first,second,proposal.poses,self.config,occupied):
                    options.append((arcs,metrics))
        options.sort(key=lambda v:(-min(a.length for a in v[0]),digest([a.intervals for a in v[0]]),[a.piece for a in v[0]]))
        chosen=[];pairs=set();used=list(occupied)
        for arcs,metrics in options:
            pair=tuple(a.piece for a in arcs)
            if pair in pairs or any(overlaps(a,b,self.config) for a in arcs for b in used):continue
            chosen.append((arcs,metrics));pairs.add(pair);used.extend(arcs)
            if len(chosen)==3:break
        if len(chosen)<MINIMUM_PAIRS:
            return None,{'disposition':'rejected_native_bridge','distinct_pairs':len(chosen),
                         'reason':'insufficient_independent_unoccupied_contiguous_seams'}
        candidate=clone(proposal)
        # Group membership changes only after distinct native seams exist.
        candidate.groups={p:min(candidate.poses) for p in candidate.poses}
        physical=feasibility.state(candidate,final=True)
        if not physical['valid']:
            return None,{'disposition':'rejected_native_collision','reason':physical['reason']}
        records=[];edges=[]
        for arcs,metrics in chosen:
            record=make_record(self.problem,arcs,candidate.poses,fingerprint(source),proposal_id,groups)
            edge,rebuilt=edge_from_record(self.problem,record,self.config)
            edges.append((edge,rebuilt));records.append({'bridge_id':edge.id,'pieces':[edge.a,edge.b],
                'native_lengths_px':[a.length for a in rebuilt],**metrics})
        for edge,arcs in edges:
            oid=edge.occurrences[0]['occurrence_id']
            candidate.selected=(*candidate.selected,edge.id);candidate.assignments[edge.id]=oid
        candidate.tree=(*candidate.tree,edges[0][0].id)
        from .active_support import verify_result_support
        from .outputs import serialize_layout
        temporary=replace(self.problem,edges=[*self.problem.edges,*[e for e,_ in edges if e.id not in self.edges]])
        try:
            verify_result_support(temporary,{'layouts':[serialize_layout(temporary,candidate,0,physical,{'attempted':False})]},self.config)
        except ValueError as exc:
            return None,{'disposition':'rejected_native_bridge','reason':str(exc)}
        # Only independently verified proposals add local evidence. The
        # immutable input pool and its Stage 3 scores remain untouched.
        for edge,arcs in edges:
            if edge.id not in self.edges:
                self.problem.edges.append(edge);self.edges[edge.id]=edge
            self.support.arcs[edge.occurrences[0]['occurrence_id']]=arcs
        # New selected cycles corroborate straight bridges without a fake vote
        # from duplicated fits of one physical pair.
        self.cache.clear();self.ambiguity_cache.clear()
        scorer=self.scorer();scorer.edges.update(self.edges)
        candidate.score=float(scorer.evaluate(candidate)['score'])
        return candidate,{'disposition':'verified_closure','bridges':records,
                          'candidate_id':fingerprint(candidate),'native_physical_valid':True}

    def augment_finalists(self,states,deadline):
        states=super().augment_finalists(states,deadline)
        started=time.perf_counter();stop=min(started+self.config.closure_seconds,deadline-5.)
        allowance=max(0,getattr(self,'finalization_proposal_allowance',self.config.closure_proposal_limit)-len(self.deletion_audit['checks']))
        limit=min(allowance,self.config.closure_proposal_limit)
        self.closure_audit['effective_proposal_limit']=limit
        sources=[];families=set()
        for state in sorted(states,key=lambda s:(-s.score,fingerprint(s))):
            if len(state.components())!=2:continue
            family=self.pose_family(state)
            if family in families:continue
            families.add(family);sources.append(state)
        self.closure_audit['eligible_source_families']=len(sources)
        self.closure_audit['sources_omitted_by_limit']=max(0,len(sources)-self.config.closure_source_limit)
        sources=sources[:self.config.closure_source_limit]
        # The ordinary search and already complete pages do not depend on this
        # additional route. Two-component sources only in this first version.
        proposals=[]
        for source in sources:
            source_row={'source_candidate_id':fingerprint(source),'component_sizes':list(map(len,source.components())),
                        'generated':0};self.closure_audit['sources'].append(source_row)
            for key,state,metadata in placement_proposals(self,source):
                if len(proposals)>=limit or time.perf_counter()>=stop:
                    self.closure_audit['budget_abstained']=True;break
                measurement=shape_measurement(self,state,256)
                row={'proposal_id':key,'source_candidate_id':fingerprint(source),**metadata,
                     'coarse_geometry':measurement,'disposition':'coarse_only',
                     'native_to_common_frame':{p:h.tolist() for p,h in state.poses.items()}}
                proposals.append((row,source,state));source_row['generated']+=1
            if self.closure_audit['budget_abstained']:break
        proposals.sort(key=lambda v:(not v[0]['coarse_geometry']['shape_gate_pass'],-v[0]['coarse_geometry']['score'],v[0]['proposal_id']))
        from .feasibility import Feasibility
        feasibility=Feasibility(self.problem,self.config) if proposals else None
        extra=[]
        for row,source,state in proposals[:self.config.closure_fine_limit]:
            if time.perf_counter()>=stop:
                row['disposition']='unverified_budget';self.closure_audit['budget_abstained']=True;continue
            self.closure_audit['fine_checks']+=1
            row['fine_geometry']=shape_measurement(self,state,768)
            if not row['fine_geometry']['shape_gate_pass']:
                row['disposition']='rejected_geometry';continue
            candidate,verdict=self.verify_placement(source,state,row['proposal_id'],feasibility)
            row.update(verdict)
            if candidate is not None:extra.append(candidate)
        self.closure_audit.update(proposals=[v[0] for v in proposals],generated_proposals=len(proposals),
            verified_closures=len(extra),elapsed_seconds=time.perf_counter()-started,
            unexamined_after_coarse=max(0,len(proposals)-self.closure_audit['fine_checks']),
            fine_limit_reached=len(proposals)>self.config.closure_fine_limit)
        return [*states,*extra]

    def report(self):
        value=super().report();value['component_closure']=self.closure_audit
        return value

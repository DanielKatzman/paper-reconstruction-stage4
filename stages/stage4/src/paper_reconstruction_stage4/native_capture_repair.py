"""Common-scale capture recovery after every ordinary cold route gets a turn."""
from collections import OrderedDict,Counter
from itertools import combinations
import time
import numpy as np
from .native_repair_v21 import RepairContext as PreviousRepairContext
from .native_repair_v18 import RepairContext as SupportContext
from .native_graph_v17 import Growth
from .native_partial import PartialRepair,evaluate_contacts
from .page_geometry import fingerprint
from .outputs import digest


class RepairContext(PreviousRepairContext):
    def __init__(self,problem,config,support):
        super().__init__(problem,config,support)
        self.partial_jobs=OrderedDict();self.partial_support=OrderedDict();self.partial_repairs=OrderedDict()
        self.partial_seen=set();self.partial_candidates=OrderedDict();self.capture_counts=Counter()
        self.partial_repair_counts=Counter()
        self.repair_audit.update(capture_recovery_policy='cold-routes-then-persistent-capture-repair.v1',
            native_first=False,ordinary_routes_yield_before_native=True,
            partial_selection_policy='verified-component-progress.v1',reference_geometry_used=False)

    def solve_geometry(self,feasibility):
        # The ordinary portfolio already shares its allowance fairly. Let it
        # execute before persistent native DFS can consume the shared deadline.
        self.repair_audit['native_search_deferred_for_cold_routes']=True
        return None

    def observe_growth(self,growth):
        depth=max(map(len,growth.node.state.components()));n=len(growth.node.state.poses)
        if depth<max(2,n-2) or depth==n:return
        key=digest([fingerprint(growth.node.state),[r['proposal_id'] for r in growth.node.contacts]])
        if key in self.partial_seen:return
        self.partial_seen.add(key);self.partial_jobs[key]=growth;self.partial_repairs[key]=PartialRepair(growth)
        self.capture_counts['partial_hypotheses_observed']+=1
        # Bounded diagnostic/repair ordering cache; native DFS keeps its own
        # live branches. An eviction is not an impossibility/exhaustion claim.
        limit=self.config.repair_beam
        for values in (self.partial_jobs,self.partial_repairs):
            if len(values)>limit:
                def priority(item):
                    candidate=item[1] if values is self.partial_jobs else item[1].growth
                    return max(map(len,candidate.node.state.components())),candidate.node.value,item[0]
                values.pop(min(values.items(),key=priority)[0])
        if len(self.partial_seen)>self.config.maximum_cache_entries:self.partial_seen={key}

    def extra_tasks(self):return ('capture_partial_support','capture_partial_repair')

    def extra_task_ready(self,name):
        if name=='capture_partial_support':return bool(self.partial_jobs)
        return any(job.revision!=self.proposal_revision or not job.exhausted for job in self.partial_repairs.values())

    def evaluate_contacts(self,source,node,contacts,physical):
        if len(node.state.components())>1:return evaluate_contacts(self,source,node,contacts,physical)
        return super().evaluate_contacts(source,node,contacts,physical)

    def extra_task_step(self,name):
        if name=='capture_partial_support':
            key,growth=next(iter(self.partial_jobs.items()));original=self.support_jobs
            self.support_jobs=self.partial_support
            try:
                candidate,verdict=SupportContext.finalize(self,self.native_basis,growth.node,self.native_physical)
            finally:self.support_jobs=original
            self.capture_counts['partial_support_turns']+=1
            if candidate is not None:
                self.partial_candidates[fingerprint(candidate)]=candidate
                self.capture_counts['verified_partial_candidates']+=1
            if candidate is not None or not verdict.get('support_selection',{}).get('continuation_pending',False):
                self.partial_jobs.pop(key,None)
            else:self.partial_jobs.move_to_end(key)
            if len(self.partial_candidates)>self.config.ambiguity_partial_slots:
                worst=min(self.partial_candidates,key=lambda k:(max(map(len,self.partial_candidates[k].components())),
                    self.utility(self.partial_candidates[k],self.features(self.partial_candidates[k])),k))
                self.partial_candidates.pop(worst)
            if len(self.partial_support)>self.config.maximum_cache_entries:self.partial_support.popitem(last=False)
            return None,True
        for key,job in list(self.partial_repairs.items()):
            if job.revision==self.proposal_revision and job.exhausted:continue
            before=job.counts.copy()
            node,worked=job.step(self,self.native_physical);self.partial_repairs.move_to_end(key)
            self.partial_repair_counts.update(job.counts-before)
            if node is not None:
                frontier=self.frontiers.get('all_modes_fallback')
                growth=Growth(node,job.growth.anchor,job.growth.seed)
                if frontier is not None and frontier.inject(growth):self.capture_counts['repaired_partial_roots']+=1
            return None,worked
        return None,False

    def closure_priority(self,state,features):
        value=super().closure_priority(state,features)
        if not features['complete'] and features['rigid'] and features.get('component_bounds',{}).get('valid',True):
            if not features.get('contacts',{}).get('unresolved_weak_geometry_ids'):
                return .5*(max(map(len,state.components()))-1)/max(1,len(state.poses)-1)
        return value

    def report(self):
        self.repair_audit['capture_partial_recovery']={**dict(self.capture_counts),
            'repair_counts':dict(self.partial_repair_counts),'pending_support_jobs':len(self.partial_jobs),
            'retained_verified_partials':len(self.partial_candidates),'raw_growth_published_as_verified':False}
        return super().report()

    def _merge_generation(self,rows,audit,label):
        self.generated=list({r['proposal_id']:r for r in [*self.generated,*rows]}.values())
        self.repair_audit[label]=audit;self.repair_audit['generated_pose_proposals']=self.generated
        self.repair_audit['proposal_checks']+=audit.get('native_checks',0)
        self.repair_audit['shared_native_generation_passes']=self.repair_audit.get('shared_native_generation_passes',0)+1

    def archive_pairs(self,states):
        """Use unresolved groups and outline inflation to prioritize angle recovery."""
        from .page import enclosing_rectangle
        from paper_reconstruction_contracts import map_points
        partial=[s for s in states if 1<max(map(len,s.components()))<len(s.poses)]
        best=max((max(map(len,s.components())) for s in partial),default=1);pairs=set()
        for state in partial:
            if max(map(len,state.components()))!=best:continue
            pairs.update((a,b) for a,b in combinations(sorted(state.poses),2) if state.groups[a]!=state.groups[b])
        complete=sorted((s for s in states if len(s.components())==1),key=lambda s:-s.score)
        for state in complete[:self.config.repair_source_limit]:
            points={pid:map_points(self.hulls[pid],state.poses[pid]) for pid in state.poses}
            if len(points)<2:continue
            full=enclosing_rectangle(np.vstack(list(points.values())))[0];rank=[]
            for pid in sorted(points):
                rest=np.vstack([v for other,v in points.items() if other!=pid])
                area=enclosing_rectangle(rest)[0]
                rank.append(((full-area)/max(1.,self.areas[pid]),pid))
            suspects={pid for _,pid in sorted(rank,reverse=True)[:self.config.repair_source_limit]}
            pairs.update((a,b) for a,b in combinations(sorted(state.poses),2) if a in suspects or b in suspects)
        if not pairs:pairs=set(combinations(sorted(self.problem.pieces),2))
        return pairs

    def augment_finalists(self,states,deadline):
        if self.augmentation_done:return [*states,*self.partial_candidates.values()]
        self.augmentation_done=True
        from .active_support import verify_result_support
        from .outputs import serialize_layout
        from .feasibility import Feasibility
        from .component_closure import shape_measurement
        physical=Feasibility(self.problem,self.config)
        for state in states:
            if len(state.components())!=1 or not shape_measurement(self,state,768)['shape_gate_pass']:continue
            check=physical.state(state,final=True)
            if not check['valid']:continue
            try:verify_result_support(self.problem,{'layouts':[serialize_layout(self.problem,state,0,check,{'attempted':False})]},self.config)
            except ValueError:continue
            self.geometry_finished=True;self.repair_audit['route_stop_reason']='protected_verified_construction'
            return states
        from .seam_poses_v18 import generate
        if time.perf_counter()<deadline:
            # Prepare the finite working/targeted archive modes before a
            # marginal incumbent can hold persistent search to the deadline.
            rows,audit=generate(self.problem,self.config,deadline,maximum_checks=None,peaks=8)
            self._merge_generation(rows,audit,'native_generation')
            self.expanded_generation_done=bool(audit.get('all_configured_work_completed',True))
        if time.perf_counter()<deadline:
            from .archive_rotations import modes
            pairs=self.archive_pairs(states)
            if pairs:
                angles,scan=modes(self.problem,pairs,deadline);self.repair_audit['archive_rotation_recovery']=scan
                if angles and time.perf_counter()<deadline:
                    from .seam_poses_rigid import generate as fit_angles
                    rows,audit=fit_angles(self.problem,self.config,deadline,maximum_checks=None,peaks=8,angle_seeds=angles)
                    self._merge_generation(rows,audit,'targeted_archive_generation')
        candidate=self.run_geometry(deadline)
        self.repair_audit['capture_partial_recovery']={**dict(self.capture_counts),
            'repair_counts':dict(self.partial_repair_counts),'pending_support_jobs':len(self.partial_jobs),
            'retained_verified_partials':len(self.partial_candidates),'raw_growth_published_as_verified':False}
        return [*([candidate] if candidate is not None else []),*states,*self.partial_candidates.values()]

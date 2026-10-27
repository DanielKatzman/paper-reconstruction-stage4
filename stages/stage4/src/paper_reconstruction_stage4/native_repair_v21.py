"""Observed support-cut repair interleaved with persistent native frontiers."""
from collections import Counter, OrderedDict
import time
from .native_repair_v18 import RepairContext as PreviousRepairContext
from .native_graph_v19 import informative
from .native_graph_v21 import NativeFrontier
from .cut_repair_v21 import CutJob, smaller_side
from .model import singleton_state
from .feasibility import Feasibility
from .outputs import digest
from .page_geometry import fingerprint, state_record
from .native_repair_v17 import satisfactory


COUNTERS=('merge_checks','native_proposal_replay_checks','final_candidates_checked',
    'complete_shape_candidates','backtracks','backtracking_seed_sessions_started','duplicate_final_histories_avoided')


class RepairContext(PreviousRepairContext):
    def __init__(self,problem,config,support):
        super().__init__(problem,config,support)
        self.frontiers={}; self.cut_jobs=OrderedDict(); self.proposal_revision=0; self.known_proposals=set()
        self.graph_final_histories=set(); self.native_basis=None; self.native_physical=None
        self.last_growth_snapshot={}; self.last_rejections={}; self.last_growth_seconds={}
        self.expand_for_unsupported_cut=False; self.schedule_cursor=0; self.native_fully_attempted=False
        self.expand_for_geometry=False; self.verified_incumbent=None; self.verified_verdict=None
        self.task_seconds=Counter()
        self.repair_audit.update(cut_repair_policy='observed-unsupported-cut-existing-modes.v1',
            native_scheduler_policy='persistent-beam-and-backtracking.time-fair.v2',
            incumbent_policy='preserve-verified-page-continue-to-existing-0.98-geometry-target.v1',
            acceptance_thresholds_changed=False,reference_used_for_cut_repair=False)

    def finalize(self,source,node,physical):
        candidate,verdict=super().finalize(source,node,physical)
        if candidate is not None:
            score=verdict.get('geometry',{}).get('score',float('-inf'))
            if self.verified_incumbent is None or score>self.verified_verdict.get('geometry',{}).get('score',float('-inf')):
                self.verified_incumbent=candidate; self.verified_verdict=verdict
            if not satisfactory(candidate,verdict) and not self.expanded_generation_done:
                self.expand_for_geometry=True
        shape=verdict.get('geometry',{})
        if candidate is None and shape.get('shape_gate_pass') and verdict.get('unsupported_bridges') and not source.selected:
            for cut in verdict['unsupported_bridges']:
                moving=smaller_side(cut)
                # Only a single bridge of the independent growth tree can be
                # replaced while retaining both sides' internal connectivity.
                crossing=[r for r in node.contacts if (r['pieces'][0] in moving)!=(r['pieces'][1] in moving)]
                if len(crossing)!=1 or crossing[0]['role']!='merge':
                    continue
                key=digest([fingerprint(node.state),list(moving),sorted(r['proposal_id'] for r in node.contacts)])
                if key not in self.cut_jobs:
                    self.cut_jobs[key]=CutJob(source,node,moving,cut,self.proposal_revision)
                if not self.expanded_generation_done:
                    self.expand_for_unsupported_cut=True
        return candidate,verdict

    def _complete(self,node,shape):
        key=digest([fingerprint(node.state),sorted(r['proposal_id'] for r in node.contacts)])
        if key in self.graph_final_histories:
            self.repair_audit['duplicate_final_histories_avoided']=self.repair_audit.get('duplicate_final_histories_avoided',0)+1
            return None
        self.graph_final_histories.add(key)
        candidate,verdict=self.finalize(self.native_basis,node,self.native_physical)
        verdict.update(coarse_candidate_id=fingerprint(node.state),lane='persistent_native_growth')
        self.repair_audit['final_checks'].append(verdict)
        if candidate is None and 'diagnostic_state' in verdict:
            self.repair_audit['rejected_complete_candidates'].append(verdict)
        return candidate if satisfactory(candidate,verdict) else None

    def _update_frontiers(self):
        new={p['proposal_id'] for p in self.generated}-self.known_proposals
        if not new:
            return
        self.known_proposals.update(new); self.proposal_revision+=1
        if self.native_basis is None:
            self.native_basis=singleton_state(self.problem); self.native_physical=Feasibility(self.problem,self.config)
        lanes={'nonstraight_first':[p for p in self.generated if informative(p)],'all_modes_fallback':self.generated}
        for name,rows in lanes.items():
            if name not in self.frontiers:
                self.frontiers[name]=NativeFrontier(self,self.native_basis,self.native_physical,self._complete,name)
            self.frontiers[name].add(rows)

    def _cut_turn(self):
        for key,job in list(self.cut_jobs.items()):
            job.update(self)
            if not job.pending:
                continue
            candidate,worked=job.step(self,self.native_physical)
            self.cut_jobs.move_to_end(key)
            verdict=job.records[-1].get('verdict',{}) if worked and job.records else {}
            return (candidate if satisfactory(candidate,verdict) else None),worked
        return None,False

    def _support_turn(self):
        pending=next(((key,job) for key,job in self.support_jobs.items() if not job['exhausted']),None)
        if pending is None:
            return None,False
        key,job=pending
        candidate,verdict=self.finalize(job['source'],job['node'],self.native_physical)
        self.support_jobs.move_to_end(key)
        verdict.update(lane='deferred_native_support',continuation_resume=True)
        self.repair_audit['final_checks'].append(verdict)
        return (candidate if satisfactory(candidate,verdict) else None),True

    def _save_cut_audit(self):
        counts=Counter()
        for job in self.cut_jobs.values():
            counts.update(job.counts)
        self.repair_audit['unsupported_cut_repair']={
            'policy':'observed-unsupported-cut-existing-modes.v1','jobs':len(self.cut_jobs),
            **dict(counts),'proposal_bank_revision':self.proposal_revision,
            'new_pair_fits_generated':False,'reference_poses_used':False,'acceptance_thresholds_changed':False,
            'local_ordering_radius_native_px':4.*self.config.cycle_tolerance_px,
            'wider_modes_permanently_discarded':False,
            'records':[r for job in self.cut_jobs.values() for r in job.records]}

    def run_geometry(self,deadline,fast_only=False):
        started=time.perf_counter(); self.repair_deadline=deadline
        self._update_frontiers(); candidate=None; reason=None
        for frontier in self.frontiers.values():
            frontier.answer=None  # A verified incumbent is held by the caller.
        # The expanded bank first revisits previously identified failed cuts.
        # Other lanes then take turns; no lane gets the entire remaining clock.
        if self.expanded_generation_done and time.perf_counter()<deadline:
            cut_started=time.perf_counter()
            candidate,_=self._cut_turn()
            self.task_seconds['cut']+=time.perf_counter()-cut_started
        names=list(self.frontiers)
        tasks=[(name,kind) for kind in ('beam','dfs') for name in names]+[('cut',None),('support',None)]
        weights={(name,kind):(8. if (name,kind)==('nonstraight_first','beam') else 1.) for name,kind in tasks}
        while candidate is None and time.perf_counter()<deadline:
            available=[]
            for name,kind in tasks:
                if name=='cut':
                    ready=any(j.revision!=self.proposal_revision or j.pending for j in self.cut_jobs.values())
                elif name=='support':
                    ready=any(not j['exhausted'] for j in self.support_jobs.values())
                else:
                    ready=not (self.frontiers[name].fast_done if kind=='beam' else self.frontiers[name].dfs_done)
                if ready:available.append((name,kind))
            if not available:
                reason='configured_native_graph_exhausted';break
            def spent(task):
                name,kind=task;key=name if kind is None else name+'/'+kind
                return self.task_seconds[key]/weights[task]
            name,kind=min(available,key=spent)
            task_key=name if kind is None else name+'/'+kind
            turn_started=time.perf_counter()
            worked=False
            if name=='cut':
                candidate,worked=self._cut_turn()
            elif name=='support':
                candidate,worked=self._support_turn()
            else:
                frontier=self.frontiers[name]
                # Scheduling quantum only: the same generator resumes next
                # turn. It is not a proposal, candidate or search stop budget.
                turn=min(deadline,time.perf_counter()+.10)
                while time.perf_counter()<turn:
                    advanced=frontier.step(kind); worked|=advanced
                    if frontier.answer is not None:
                        candidate=frontier.answer; break
                    if not advanced or (self.expand_for_unsupported_cut or self.expand_for_geometry) and not self.expanded_generation_done:
                        break
            # Charge expensive atomic support/pool work to its actual lane.
            # A costly callback must not receive another turn merely because
            # cheap growth steps and that callback each counted as one turn.
            self.task_seconds[task_key]+=time.perf_counter()-turn_started
            if (self.expand_for_unsupported_cut or self.expand_for_geometry) and not self.expanded_generation_done and candidate is None:
                reason='unsupported_cut_requests_more_observed_modes' if self.expand_for_unsupported_cut else 'verified_geometry_requests_more_observed_modes'
                break
            if fast_only and all(f.fast_done for f in self.frontiers.values()) and (
                    self.verified_incumbent is None or satisfactory(self.verified_incumbent,self.verified_verdict)):
                reason='persistent_fast_pass_yielded'; break
            if all(f.fast_done and f.dfs_done for f in self.frontiers.values()):
                has_cut=any(job.pending for job in self.cut_jobs.values())
                has_support=any(not job['exhausted'] for job in self.support_jobs.values())
                if not has_cut and not has_support:
                    reason='configured_native_graph_exhausted'; break
        if self.verified_incumbent is not None:
            candidate=self.verified_incumbent
        if candidate is not None:
            self.geometry_finished=True
            self.repair_audit.update(verified_completions=1,route_stop_reason='verified_construction',
                acceptance_requires_pixel_perfection=False,
                verified_incumbent={'candidate_id':fingerprint(candidate),
                    'geometry':self.verified_verdict['geometry'],
                    'existing_geometry_stop_target_met':satisfactory(candidate,self.verified_verdict),
                    'preserved_after_deadline':time.perf_counter()>=deadline})
            reason='verified_construction'
        elif time.perf_counter()>=deadline:
            reason='overall_deadline'
        else:
            reason=reason or 'configured_native_graph_exhausted'
        self._save_cut_audit()
        lanes=[]; totals=Counter(); rejects=Counter()
        for name,frontier in self.frontiers.items():
            full=frontier.audit(); previous=self.last_growth_snapshot.get(name,Counter())
            delta={k:full.get(k,0)-previous.get(k,0) for k in COUNTERS}
            before=self.last_rejections.get(name,Counter()); after=Counter(full['rejections'])
            rejected={k:v-before[k] for k,v in after.items() if v!=before[k]}
            seconds=full['elapsed_seconds']-self.last_growth_seconds.get(name,0.)
            lanes.append({**full,**delta,'elapsed_seconds':seconds,'rejections':rejected,
                'counters_are_per_resume_delta':True})
            totals.update(delta); rejects.update(rejected)
            self.last_growth_snapshot[name]=Counter({k:full.get(k,0) for k in COUNTERS})
            self.last_rejections[name]=after; self.last_growth_seconds[name]=full['elapsed_seconds']
        audit={**dict(totals),'policy':'persistent-native-cut-repair-scheduler.v1','growth_lanes':lanes,
            'completed_depth':max((f.deepest for f in self.frontiers.values()),default=1),
            'required_piece_depth':len(self.problem.pieces),'rejections':dict(rejects),
            'stop_reason':reason,'budget_abstained':reason=='overall_deadline',
            'elapsed_seconds':time.perf_counter()-started,'lane':'persistent_native_fast' if fast_only else 'persistent_native_backtracking',
            'source_groups':self.native_basis.components() if self.native_basis else [],
            'old_piece_poses_inherited':False,'evaluation_truth_opened':False,'legacy_selected_joins_inherited':0,
            'frontiers_restarted':False,'fast_pass_pruning_is_reopened':True,
            'lower_modes_permanently_discarded':False,'maximum_merge_checks':None,'final_candidate_count_limit':None,
            'proposal_count_is_telemetry':True,'growth_choices_exhaustive':False,
            'proposal_graph_exhausted':reason=='configured_native_graph_exhausted',
            'observed_growth_states':[r for lane in lanes for r in lane['observed_growth_states']]}
        self.repair_audit['merge_searches'].append(audit)
        self.repair_audit['proposal_checks']+=audit.get('merge_checks',0)
        self.repair_audit['final_candidates_checked']=len(self.repair_audit['final_checks'])
        self.repair_audit['pending_support_searches']=sum(not j['exhausted'] for j in self.support_jobs.values())
        self.repair_audit['route_stop_reason']=reason
        self.repair_audit['scheduler_work']={'seconds_by_task':dict(self.task_seconds),
            'weights':{(name if kind is None else name+'/'+kind):w for (name,kind),w in weights.items()},
            'policy':'least-consumed-normalized-wall-time.v1','work_counts_are_not_stop_limits':True}
        self.repair_audit['budget_abstained']=bool(candidate is None and time.perf_counter()>=deadline)
        self.repair_audit['deadline_reached']=time.perf_counter()>=deadline
        return candidate

    def solve_geometry(self,feasibility):
        from .search.common import SearchResult
        result=super().solve_geometry(feasibility)
        if result is not None:
            return result  # Includes the unchanged verified-incumbent protection.
        self.native_fully_attempted=True
        candidate=self.run_geometry(self.problem.deadline-self.config.joint_finalize_seconds)
        if candidate is not None:
            return SearchResult([candidate],{'proposals':0,'native_first_completion':True},'verified_construction')
        return None

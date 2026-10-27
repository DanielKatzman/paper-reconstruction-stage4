"""Versioned native-seam repair after the protected V13 completion route."""
from dataclasses import replace
import time
import numpy as np
from .component_closure import ClosureContext,shape_measurement
from .attachments import without_join
from .page_geometry import fingerprint
from .search.execution import clone
from .geometry import relative
from .outputs import digest,serialize_layout
from .native_bridge import pair_options,edge_from_record,check_arcs
from .native_bridge_v2 import make_record
from .seam_poses import generate,spread
from .repair_search import search
from .active_support import verify_result_support
from .feasibility import Feasibility


class RepairContext(ClosureContext):
    def generate_proposals(self,deadline,maximum_checks,sources):
        return generate(self.problem,self.config,deadline,maximum_checks=maximum_checks)

    def merge_search(self,source,proposals,deadline,maximum_checks):
        return search(self,source,proposals,deadline,maximum_checks=maximum_checks,beam=self.config.repair_beam)

    def __init__(self,problem,config,support):
        super().__init__(problem,config,support)
        self.repair_audit={'policy':config.native_repair_policy,'sources':[],'native_generation':None,
            'merge_searches':[],'final_checks':[],'proposal_checks':0,'verified_completions':0,
            'evaluation_truth_opened':False,'protected_supported_completion':False,
            'maximum_released_joins':4,'maximum_components':6,
            'capture_binding':problem.capture_metadata,'budget_abstained':False}

    def sources(self,states):
        sources=[];families=set()
        ranked=sorted(states,key=lambda s:(-s.score,fingerprint(s)))
        # First protect a well-conditioned core recovery, chosen entirely from
        # observed support. Straight true joins stay available in other seeds.
        for state in ranked[:4]:
            if len(state.components())==1:continue
            suspect=[j for j in self.contacts(state)['joins'] if j['selected_graph_bridge'] and
                j['shape_spread_ratio']<.05 and not j['independent_witness_pairs']]
            seed=clone(state);removed=[]
            for join in sorted(suspect,key=lambda j:(j['native_length_px'],j['geometry_id']))[:4]:
                seed=without_join(seed,self.edges,join['geometry_id'],canonical=True);removed.append(join['geometry_id'])
            if 2<=len(seed.components())<=6:
                family=self.pose_family(seed)
                if family not in families:
                    families.add(family);sources.append((seed,removed,'conditioned_core'))
        # The already-retained five-plus-singletons style alternatives are
        # explicitly eligible, without relying on a fixed number of groups.
        for state in ranked:
            if not 2<=len(state.components())<=6:continue
            family=self.pose_family(state)
            if family in families:continue
            families.add(family);sources.append((state,[],'retained_partition'))
        return sources[:self.config.repair_source_limit],max(0,len(sources)-self.config.repair_source_limit)

    def finalize(self,source,node,physical):
        state=clone(node.state);used=[]
        for eid in source.selected:used.extend(self.support.arcs_for(source,eid))
        contacts=[];selected_pairs=set()
        for row in node.contacts:
            check_arcs(self.problem,row['arcs'],state.poses,self.config,used)
            used.extend(row['arcs']);contacts.append(row);selected_pairs.add(tuple(row['pieces']))
        candidates=[];ids=sorted(state.poses)
        # Additional independently measured native contacts corroborate sliding
        # joins; they are not duplicated votes from an old pair fit.
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                if (a,b) in selected_pairs or source.groups[a]==source.groups[b]:continue
                for arcs,metrics in pair_options(self.problem,a,b,state.poses,self.config,used):
                    if min(v.length for v in arcs)<96.:continue
                    candidates.append((arcs,metrics))
        candidates.sort(key=lambda v:(-min(a.length for a in v[0]),[a.piece for a in v[0]]))
        from .support import overlaps
        for arcs,metrics in candidates:
            a,b=(v.piece for v in arcs)
            if (a,b) in selected_pairs or any(overlaps(v,w,self.config) for v in arcs for w in used):continue
            h=relative(state.poses,a,b);proposed=h.copy();proposed[:2,:2]=np.rint(h[:2,:2])
            contacts.append({'pieces':[a,b],'arcs':arcs,'role':'cycle','metrics':metrics,
                'proposal_id':digest(['native-contact-at-final-pose.v1',a,b,h.tolist()]),
                'proposed_transform_b_to_a':proposed.tolist(),'source_groups':source.components(),
                'merged_groups':[sorted(p for p in source.poses if source.groups[p]==source.groups[a]),
                                 sorted(p for p in source.poses if source.groups[p]==source.groups[b])]})
            used.extend(arcs);selected_pairs.add((a,b))
        edges=[]
        for contact in contacts:
            record=make_record(self.problem,state,contact,fingerprint(source));edge,arcs=edge_from_record(self.problem,record,self.config)
            state.selected=(*state.selected,edge.id);state.assignments[edge.id]=edge.occurrences[0]['occurrence_id']
            if contact['role']=='merge':state.tree=(*state.tree,edge.id)
            edges.append((edge,arcs))
        shape=shape_measurement(self,state,768)
        if not shape['shape_gate_pass']:return None,{'disposition':'rejected_geometry','geometry':shape}
        check=physical.state(state,final=True)
        if not check['valid']:return None,{'disposition':'rejected_native','reason':check['reason'],'geometry':shape}
        temporary=replace(self.problem,edges=[*self.problem.edges,*[e for e,_ in edges if e.id not in self.edges]])
        try:
            verify_result_support(temporary,{'layouts':[serialize_layout(temporary,state,0,check,{'attempted':False})]},self.config)
        except ValueError as exc:return None,{'disposition':'rejected_native_support','reason':str(exc),'geometry':shape}
        for edge,arcs in edges:
            if edge.id not in self.edges:self.problem.edges.append(edge);self.edges[edge.id]=edge
            self.support.arcs[edge.occurrences[0]['occurrence_id']]=arcs
        self.cache.clear();self.ambiguity_cache.clear()
        scorer=self.scorer();scorer.edges.update(self.edges);state.score=float(scorer.evaluate(state)['score'])
        return state,{'disposition':'verified_completion','candidate_id':fingerprint(state),'geometry':shape,
            'native_bridges':len(edges),'new_pairs':[list(v) for v in sorted(selected_pairs)]}

    def augment_finalists(self,states,deadline):
        states=super().augment_finalists(states,deadline)
        started=time.perf_counter();stop=min(started+self.config.repair_seconds,deadline-8.)
        physical=Feasibility(self.problem,self.config)
        # Preserve successful controls and the proven V13 completion route.
        for state in states:
            if len(state.components())==1 and shape_measurement(self,state,256)['shape_gate_pass'] and self.closure_priority(state,self.features(state))>0:
                if physical.state(state,final=True)['valid']:
                    try:verify_result_support(self.problem,{'layouts':[serialize_layout(self.problem,state,0,{'valid':True},{'attempted':False})]},self.config)
                    except ValueError:continue
                    self.repair_audit['protected_supported_completion']=True
                    return states
        metadata=self.problem.capture_metadata
        if not metadata or metadata.get('rotation_family')!='quarter_turn':
            self.repair_audit['unavailable_reason']='capture_does_not_declare_quarter_turns';return states
        sources,omitted=self.sources(states);self.repair_audit['sources_omitted_by_limit']=omitted
        self.repair_audit['scheduled_source_count']=len(sources)
        allowance=max(0,getattr(self,'finalization_proposal_allowance',0)-len(self.deletion_audit['checks'])-self.closure_audit['generated_proposals'])
        if not sources or allowance==0 or time.perf_counter()>=stop:
            self.repair_audit['budget_abstained']=True;return states
        proposals,generation=self.generate_proposals(min(stop,started+self.config.repair_seconds*.4),
            min(allowance,self.config.repair_native_checks),sources)
        self.repair_audit['native_generation']=generation
        self.repair_audit['proposal_checks']=generation['native_checks'];allowance-=generation['native_checks']
        self.repair_audit['generated_pose_proposals']=proposals
        extra=[];checked=0
        for source,removed,kind in sources:
            if allowance<=0 or time.perf_counter()>=stop:break
            self.repair_audit['sources'].append({'candidate_id':fingerprint(source),'groups':source.components(),
                'released_geometry_ids':removed,'kind':kind})
            complete,trace=self.merge_search(source,proposals,stop,min(allowance,self.config.repair_merge_checks))
            self.repair_audit['merge_searches'].append(trace)
            self.repair_audit['proposal_checks']+=trace['merge_checks'];allowance-=trace['merge_checks']
            for node,shape in complete:
                if checked>=self.config.repair_complete_checks or time.perf_counter()>=stop:break
                checked+=1
                try:candidate,verdict=self.finalize(source,node,physical)
                except ValueError as exc:candidate=None;verdict={'disposition':'rejected_native_support','reason':str(exc)}
                self.repair_audit['final_checks'].append(verdict)
                if candidate is not None:extra.append(candidate)
            if extra:break
        self.repair_audit.update(verified_completions=len(extra),elapsed_seconds=time.perf_counter()-started,
            budget_abstained=time.perf_counter()>=stop or allowance<=0,
            scheduled_sources_not_attempted=len(sources)-len(self.repair_audit['sources']))
        return [*states,*extra]

    def report(self):
        value=super().report();value['native_repair']=self.repair_audit;return value

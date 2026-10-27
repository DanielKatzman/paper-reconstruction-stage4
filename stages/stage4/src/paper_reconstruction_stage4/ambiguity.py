"""Observed ambiguity, bounded deletion alternatives and partial diversity.

No new relative pose is fitted. Image variation reduces a soft uncertainty cost;
it is never positive proof of a unique placement or a hard straight-seam veto.
"""
from collections import Counter, OrderedDict
from itertools import combinations
import time
from .page_geometry import GeometryContext, fingerprint
from .search.execution import state_key
from .attachments import without_join

POLICY = 'native-sliding-ambiguity.v1'


class AmbiguityContext(GeometryContext):
    def __init__(self, problem, config, support):
        super().__init__(problem, config, support)
        self.ambiguity_cache = OrderedDict()
        self.ambiguity_seconds = Counter()
        self.deletion_audit = {'policy':POLICY, 'checks':[], 'omitted_by_budget':0,
            'maximum_removed_joins':2, 'poses_refitted':False}
        self.candidate_audit['ambiguity_policy'] = POLICY

    def component_bounds(self, state):
        started=time.perf_counter()
        value=super().component_bounds(state)
        self.ambiguity_seconds['component_bounds_calls']+=time.perf_counter()-started
        return value

    def scorer(self):
        if not hasattr(self,'ambiguity_scorer'):
            from .joint_score import JointScorer
            existing=self.problem.structural
            self.ambiguity_scorer=(existing if existing is not None and existing.config==self.config
                                   else JointScorer(self.problem,self.config,self.support))
        return self.ambiguity_scorer

    def ambiguity(self,state,contacts):
        key=state_key(state)
        if key in self.ambiguity_cache:
            self.counts['ambiguity_cache_hits']+=1
            self.ambiguity_cache.move_to_end(key);return self.ambiguity_cache[key]
        started=time.perf_counter();rows=[];measurements=0;omitted=[]
        for join in sorted(contacts['joins'],key=lambda j:(j['native_length_px'],j['geometry_id'])):
            eid=join['geometry_id'];edge=self.edges[eid]
            occurrence=next((o for o in edge.occurrences if o['occurrence_id']==state.assignments.get(eid)),edge.raw)
            alignment=occurrence.get('pose',{}).get('alignment',{})
            conditioned=bool(alignment.get('underconstrained',False) or join['shape_spread_ratio']<.05)
            if not conditioned or not join['selected_graph_bridge']:continue
            # A distinct physical pair crossing the cut supplies third-piece
            # support. Alternative fits of the very same pair are not a vote.
            witnesses=[p for p in join['independent_witness_pairs'] if set(p)!=set(join['pieces'])]
            corroborated=bool(witnesses)
            info=None;appearance_status='not_measured_independently_corroborated'
            measurement=None
            if not corroborated:
                if measurements<self.config.ambiguity_measurement_limit:
                    from .continuation import Continuation
                    scorer=self.scorer()
                    if not hasattr(scorer,'continuation_context'):scorer.continuation_context=Continuation(scorer)
                    measurement=scorer.continuation_context.measurement(state,eid)
                    info=measurement.get('appearance_information',0.)
                    appearance_status=measurement.get('appearance_evidence_status','uninformative_for_placement')
                    measurements+=1
                else:
                    omitted.append(eid);appearance_status='unverified_measurement_budget'
            # Informative local bands still do not establish displacement
            # uniqueness. Unknown or blank agreement retains the full cost.
            risk=0. if corroborated else self.config.ambiguity_risk_weight*(1.-.75*(info or 0.))
            rows.append({'geometry_id':eid,'occurrence_id':state.assignments.get(eid),
                'pieces':join['pieces'],'native_spans':join['native_spans'],
                'native_length_px':join['native_length_px'],'shape_spread_ratio':join['shape_spread_ratio'],
                'producer_underconstrained':bool(alignment.get('underconstrained',False)),
                'independent_witness_pairs':witnesses,'corroborated':corroborated,
                'appearance_information':info,'appearance_evidence_status':appearance_status,
                'local_mismatch':measurement.get('mismatch') if measurement else None,
                'risk_cost':float(risk),'unresolved_sliding':not corroborated,
                'reason':'independent_native_cut_support' if corroborated else 'unresolved_straight_or_underconstrained_bridge',
                'hard_rejection':False})
        value={'policy':POLICY,'joins':rows,'risk_cost':float(sum(r['risk_cost'] for r in rows)),
            'unresolved_geometry_ids':[r['geometry_id'] for r in rows if r['unresolved_sliding']],
            'unverified_measurement_geometry_ids':omitted,'measurements':measurements,
            'maximum_measurements':self.config.ambiguity_measurement_limit,
            'low_image_mismatch_is_positive_placement_proof':False,
            'straight_seams_automatically_rejected':False,'evaluation_truth_opened':False}
        if len(self.ambiguity_cache)>=512:self.ambiguity_cache.popitem(last=False)
        self.ambiguity_cache[key]=value;self.counts['ambiguity_extractions']+=1
        self.ambiguity_seconds['ambiguity_including_native_bands']+=time.perf_counter()-started
        return value

    def features(self,state,fine=True):
        value=super().features(state,fine)
        if value['rigid'] and 'ambiguity' not in value:
            value['ambiguity']=self.ambiguity(state,value['contacts'])
        return value

    def utility(self,state,features):
        return float(super().utility(state,features)-features.get('ambiguity',{}).get('risk_cost',0.))

    def assess(self,state,alternatives=()):
        result=super().assess(state,alternatives);f=self.features(state);a=f.get('ambiguity',{})
        result['ambiguity']=a
        # Complete-page shape plus retained-alternative disagreement remains
        # the declared full-page assessment. Partial shape cannot resolve slide.
        result['complete_page_constraints_available']=bool(f['complete'] and f['geometry']['status']=='page_like')
        if not f['complete'] and a.get('unresolved_geometry_ids'):
            reasons=set(result['reason_codes'])|{'unresolved_straight_or_underconstrained_bridge'}
            if any(j['appearance_evidence_status']=='uninformative_for_placement' and not j['corroborated'] for j in a['joins']):
                reasons.add('uninformative_local_appearance')
            result['reason_codes']=sorted(reasons)
        return result

    def augment_finalists(self,states,deadline):
        started=time.perf_counter();scorer=self.scorer();extra=[];seen={fingerprint(s) for s in states}
        limit=min(self.config.ambiguity_deletion_checks,getattr(self,'finalization_proposal_allowance',self.config.ambiguity_deletion_checks))
        self.deletion_audit['effective_proposal_limit']=limit
        # Keep at least half the finalization allowance for final native checks.
        stop=min(deadline-self.config.joint_finalize_seconds/2.,started+self.config.joint_finalize_seconds/3.)
        sources=sorted((s for s in states if len(s.components())>1),key=lambda s:(-s.score,fingerprint(s)))[:4]
        for source in sources:
            if time.perf_counter()>=stop:break
            a=self.features(source)['ambiguity']
            candidates=sorted((j for j in a['joins'] if j['unresolved_sliding']),
                key=lambda j:(-j['risk_cost'],j['native_length_px'],j['geometry_id']))[:4]
            for size in (1,2):
                for removed in combinations(candidates,size):
                    if len(self.deletion_audit['checks'])>=limit or time.perf_counter()>=stop:
                        self.deletion_audit['omitted_by_budget']+=1;continue
                    candidate=source
                    for join in removed:candidate=without_join(candidate,self.edges,join['geometry_id'],canonical=True)
                    key=fingerprint(candidate)
                    if key in seen:continue
                    seen.add(key);candidate.score=float(scorer.evaluate(candidate)['score'])
                    extra.append(candidate)
                    self.deletion_audit['checks'].append({'candidate_id':key,
                        'removed_geometry_ids':[j['geometry_id'] for j in removed],
                        'source_score':float(source.score),'candidate_score':candidate.score,
                        'independent_component_gauges_restored':True,'poses_refitted':False})
        self.ambiguity_seconds['deletion_augmentation']+=time.perf_counter()-started
        return [*states,*extra]

    def select_partials(self,states):
        ranked=sorted(states,key=lambda s:(-self.utility(s,self.features(s)),fingerprint(s)))
        half=self.config.ambiguity_partial_slots//2;selected=[];families=set()
        def add(state):
            family=self.pose_family(state)
            if family not in families:selected.append(state);families.add(family)
        for state in ranked:
            add(state)
            if len(selected)>=half:break
        # Protect safer topology alternatives alongside objective winners.
        safety=sorted(states,key=lambda s:(self.features(s)['ambiguity']['risk_cost'],
            -max(map(len,s.components())),-self.utility(s,self.features(s)),fingerprint(s)))
        topologies=set()
        for state in safety:
            topology=tuple(sorted((self.edges[e].a,self.edges[e].b) for e in state.selected))
            if topology in topologies:continue
            topologies.add(topology);add(state)
            if len(selected)>=self.config.ambiguity_partial_slots:break
        for state in ranked:
            if len(selected)>=self.config.ambiguity_partial_slots:break
            add(state)
        return selected

    def report(self):
        value=super().report()
        value.update(ambiguity_policy=POLICY,ambiguity_risk_weight=self.config.ambiguity_risk_weight,
            ambiguity_partial_slots=self.config.ambiguity_partial_slots,
            ambiguity_deletion_alternatives=self.deletion_audit,
            ambiguity_phase_seconds=dict(self.ambiguity_seconds),
            ambiguity_timer_semantics='Component-bound calls are measured separately; ambiguity and deletion timers include their nested measurements and must not be summed')
        return value

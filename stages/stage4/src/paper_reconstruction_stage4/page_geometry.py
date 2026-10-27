"""Versioned geometry gate, diverse top-k retention, and partial-page bounds.

Every score uses observed masks/contours and rigid poses. Geometry plausibility
is independent of reference labels and cannot certify correct content order.
"""
from collections import OrderedDict, Counter
import hashlib
import time
import numpy as np
from paper_reconstruction_contracts import map_points
from .geometry import relative
from .page import PageContext, enclosing_rectangle, LIMITS, THRESHOLDS
from .outputs import digest, canonical, serialize_layout
from .search.execution import clone, state_key

POLICY = 'page-geometry-topk.v1'
PARTIAL_SLOTS = 8
POSE_BUCKET_FRACTION = .003


def fingerprint(state):
    """Bounded-size exact-state receipt, avoiding large support JSON in sets."""
    h = hashlib.sha256()
    for pid, matrix in sorted(state.poses.items()):
        h.update(canonical([pid,state.groups[pid],matrix.dtype.str,list(matrix.shape)]))
        h.update(matrix.tobytes())
    h.update(canonical([list(state.selected),list(state.tree),state.assignments,state.active_support]))
    return h.hexdigest()


def state_record(state):
    return {'native_to_component':{p:h.tolist() for p,h in state.poses.items()},
        'groups':dict(state.groups),'selected_geometry_ids':list(state.selected),
        'tree_geometry_ids':list(state.tree),'occurrence_ids':dict(state.assignments),
        'active_support':state.active_support,'source_score':float(state.score)}


def record_state(record):
    from .model import State
    return State({p:np.asarray(h,float) for p,h in record['native_to_component'].items()},
        dict(record['groups']),tuple(record['selected_geometry_ids']),tuple(record['tree_geometry_ids']),
        dict(record['occurrence_ids']),record['source_score'],record.get('active_support',{}))


class GeometryContext(PageContext):
    def __init__(self, problem, config, support):
        super().__init__(problem,config,support)
        self.bound_cache = OrderedDict()
        self.family_cache = OrderedDict()
        self.coarse_geometry_cache = OrderedDict()
        self.candidate_audit = {'schema_version':'stage4.geometry-candidate-audit.v1','policy':POLICY,
            'candidates':[],'is_certified_global_k_best':False,'evaluation_truth_opened':False}
        self.pose_step = max(1.,POSE_BUCKET_FRACTION*np.sqrt(sum(self.areas.values())))

    def pose_family(self,state):
        key=self.pose_key(state)
        if key in self.family_cache:
            self.family_cache.move_to_end(key);return self.family_cache[key]
        rows=[]
        for group in state.components():
            root=min(group);rows.append(tuple(group))
            for pid in group:
                points=self.hulls[pid]
                # Four extrema of native foreground geometry, not crop padding.
                picks=points[np.unique([np.argmin(points[:,0]),np.argmax(points[:,0]),
                                        np.argmin(points[:,1]),np.argmax(points[:,1])])]
                xy=map_points(picks,relative(state.poses,root,pid))
                rows.append((pid,tuple(np.rint(xy.ravel()/self.pose_step).astype(np.int64))))
        value=tuple(rows)
        if len(self.family_cache)>=4096:self.family_cache.popitem(last=False)
        self.family_cache[key]=value
        return value

    def pose_key(self,state):
        return tuple((p,state.groups[p],h.dtype.str,h.shape,h.tobytes()) for p,h in sorted(state.poses.items()))

    def component_bounds(self,state):
        key=self.pose_key(state)
        if key in self.bound_cache:
            self.counts['component_bound_cache_hits']+=1;self.bound_cache.move_to_end(key)
            return self.bound_cache[key]
        total=sum(self.areas.values());limit=1.+self.config.page_component_area_slack;rows=[]
        for group in state.components():
            points=np.vstack([map_points(self.hulls[p],state.poses[p]) for p in group])
            area=enclosing_rectangle(points)[0];ratio=float(area/total)
            rows.append({'pieces':group,'minimum_rectangle_area_px2':area,
                'ratio_to_total_observed_area':ratio,'valid':ratio<=limit})
        value={'valid':all(v['valid'] for v in rows),'components':rows,
            'total_observed_area_px2':total,'maximum_area_ratio':limit,
            'orientation_freely_fitted':True,'partial_fill_required':False,
            'independent_component_rectangles_summed':False}
        if len(self.bound_cache)>=1024:self.bound_cache.popitem(last=False)
        self.bound_cache[key]=value;self.counts['component_bound_extractions']+=1
        return value

    def quick(self,state):
        f=super().quick(state)
        if f['rigid']:
            f['component_bounds']=self.component_bounds(state)
        return f

    def coarse_geometry(self,state):
        key=self.pose_key(state)
        if key in self.coarse_geometry_cache:
            self.counts['coarse_geometry_cache_hits']+=1
            self.coarse_geometry_cache.move_to_end(key)
            return self.coarse_geometry_cache[key]
        started=time.perf_counter();features=self.quick(state)
        if features['complete'] and features['rigid']:
            features.update(self.raster(state,LIMITS['coarse_raster_side']))
        value=self.geometry(features)
        if len(self.coarse_geometry_cache)>=1024:self.coarse_geometry_cache.popitem(last=False)
        self.coarse_geometry_cache[key]=value
        self.counts['coarse_geometry_extractions']+=1;self.seconds+=time.perf_counter()-started
        return value

    def geometry(self,features):
        reasons=[]
        if not features['rigid']:
            return {'schema_version':'stage4.page-geometry.v1','score':0.,'status':'profile_violation',
                    'reason_codes':['profile_nonrigid_transform'],'calibrated_probability':False}
        if not features['component_bounds']['valid']:
            reasons.append('component_exceeds_page_area_budget')
        if not features['complete']:
            return {'schema_version':'stage4.page-geometry.v1','score':None,
                    'status':'rejected' if reasons else 'partial','reason_codes':reasons,
                    'calibrated_probability':False,'complete_page_score_available':False}
        gap=features.get('unfilled_fraction',features['gap_proxy'])
        overlap=features.get('overlap_fraction',features['overlap_proxy'])
        holes=features.get('interior_hole_fraction',0.)
        if gap>1.-THRESHOLDS['minimum_fill']:reasons.append('large_unfilled_page_area')
        if overlap>THRESHOLDS['maximum_overlap']:reasons.append('excessive_mask_overlap')
        if holes>THRESHOLDS['maximum_holes']:reasons.append('large_interior_hole')
        score=float(np.clip(1.-(.65*gap+.20*overlap+.15*holes),0.,1.))
        return {'schema_version':'stage4.page-geometry.v1','score':score,
                'status':'rejected' if reasons else 'page_like','reason_codes':reasons,
                'calibrated_probability':False,'complete_page_score_available':True,
                'score_formula':'clip(1 - 0.65*unfilled - 0.20*overlap - 0.15*holes, 0, 1)',
                'format_or_aspect_ratio_prior_used':False,
                'content_order_verified_by_geometry':False}

    def features(self,state,fine=True):
        f=super().features(state,fine)
        if 'geometry' not in f:
            started=time.perf_counter()
            if f['rigid'] and 'contacts' not in f:
                f['contacts']=self.contacts(state)
            f['geometry']=self.geometry(f)
            self.seconds+=time.perf_counter()-started
        return f

    def utility(self,state,features):
        if not features['rigid']:return -1e12
        if not features.get('component_bounds',{}).get('valid',True):return -1e12
        if not features['complete']:
            weak=features.get('contacts',{}).get('unresolved_weak_bridge_fraction',0.)
            return float(state.score-.15*max(1,len(state.poses)-1)*weak)
        return super().utility(state,features)

    def closure_priority(self,state,features):
        g=features.get('geometry') or self.geometry(features)
        if g['status'] in {'rejected','profile_violation'}:return -1
        if not features['complete']:return 0
        return int(not features.get('contacts',{}).get('unresolved_weak_geometry_ids') and
                   len(state.selected)>=len(state.poses)-1)

    def assess(self,state,alternatives=()):
        value=super().assess(state,alternatives);f=self.features(state)
        reasons=set(value['reason_codes'])|set(f['geometry']['reason_codes'])
        if f.get('contacts',{}).get('unresolved_weak_geometry_ids'):
            reasons.add('short_straight_ambiguous_attachment')
        if reasons and value['status']!='profile_violation':value['status']='unresolved'
        value.update(reason_codes=sorted(reasons),features=f,geometry=f['geometry'],
                     geometry_policy=POLICY)
        return value

    def make_bank(self,deadline):return GeometryBank(self,deadline)

    def report(self):
        value=super().report()
        value.update(geometry_policy=POLICY,partial_page_bounds_enabled=True,
            component_area_slack=self.config.page_component_area_slack,
            solver_top_k=self.config.page_top_k,geometry_top_k=self.config.page_geometry_k,
            diversity_slots=self.config.page_diversity_k,
            maximum_final_candidate_checks=self.config.page_final_candidate_limit,
            pose_diversity_bucket_native_px=self.pose_step,
            geometry_rejection_is_not_correct_content_certification=True)
        return value

    def prioritize_finalists(self,problem,config,states,feasibility):
        from .active_support import verify_result_support
        started=time.perf_counter();deadline=min(started+config.joint_finalize_seconds,problem.deadline or float('inf'))
        if hasattr(self,'augment_finalists'):
            states=self.augment_finalists(states,deadline)
        # Rebuild a union of both top-k orders. This also protects cold-route
        # finalists that were never admitted to the streaming completion bank.
        bank=GeometryBank(self,float('inf'))
        for state in states:
            if len(state.components())==1:bank.consider(state,'final_source_union')
        complete=bank.finalists()
        partial=[s for s in states if len(s.components())>1 and self.component_bounds(s)['valid']]
        if hasattr(self,'select_partials'):
            diverse=self.select_partials(partial)
        else:
            partial.sort(key=lambda s:(-self.utility(s,self.features(s)),fingerprint(s)))
            diverse=[];families=set()
            for s in partial:
                family=self.pose_family(s)
                if family not in families:
                    diverse.append(s);families.add(family)
                if len(diverse)==PARTIAL_SLOTS:break
        pool=[*complete,*diverse]
        source_order={fingerprint(s):i+1 for i,s in enumerate(sorted(states,key=lambda s:(-s.score,fingerprint(s))))}
        # Geometry contenders precede old-score-only contenders; all are bounded
        # and checked. No skipped candidate is silently called rejected/valid.
        pool.sort(key=lambda s:(-self.closure_priority(s,self.quick(s)),
            -(self.coarse_geometry(s)['score'] or 0.),-s.score,fingerprint(s)))
        accepted=[];records=[];rejected=Counter();seen=set()
        for state in pool[:config.page_final_candidate_limit]:
            key=fingerprint(state)
            if key in seen:continue
            seen.add(key)
            row={'candidate_id':key,'source_rank_in_final_union':source_order.get(key),
                 'source_score':float(state.score),'complete':len(state.components())==1,
                 'origin':bank.origins.get(state_key(state),'protected_partial'),**state_record(state)}
            if records and time.perf_counter()>=deadline:
                row.update(disposition='unverified_budget',native_checked=False)
                records.append(row);continue
            features=self.features(state)
            row.update(geometry=features['geometry'],component_bounds=features.get('component_bounds'),
                       selection_utility=self.utility(state,features))
            if 'ambiguity' in features:row['ambiguity']=features['ambiguity']
            if features['geometry']['status'] in {'rejected','profile_violation'}:
                row.update(disposition='rejected_geometry',native_checked=False)
                rejected.update(features['geometry']['reason_codes']);records.append(row);continue
            check=feasibility.state(state,final=True)
            row.update(native_checked=True,native_valid=bool(check['valid']),native_reason=check['reason'])
            if check['valid']:
                try:verify_result_support(problem,{'layouts':[serialize_layout(problem,state,0,check,{'attempted':False})]},config)
                except ValueError:
                    row.update(native_valid=False,native_reason='conflicting_native_support')
            if not row['native_valid']:
                row['disposition']='rejected_native';rejected[row['native_reason']]+=1
            else:
                row['disposition']='eligible';accepted.append(clone(state))
            records.append(row)
        accepted.sort(key=lambda s:(-self.closure_priority(s,self.features(s)),
            -self.utility(s,self.features(s)),fingerprint(s)))
        self.candidate_audit.update(candidates=records,solver_top_k=config.page_top_k,
            geometry_top_k=config.page_geometry_k,diversity_slots=config.page_diversity_k,
            candidate_limit=config.page_final_candidate_limit,available_source_states=len(states),
            measured_complete_pool=len(complete),protected_partial_candidates=len(diverse),
            geometry_rejections=sum(r['disposition']=='rejected_geometry' for r in records),
            unverified_budget=sum(r['disposition']=='unverified_budget' for r in records),
            all_retained_candidates_checked=not any(r['disposition']=='unverified_budget' for r in records),
            selected_candidate_id=fingerprint(accepted[0]) if accepted else None,
            ranking_semantics='Union of top-k distinct pose families by solver score and page geometry, plus topology diversity; generated-state pool, not global k-best')
        return accepted,{'checked':sum(r.get('native_checked',False) for r in records),
            'geometry_evaluated':sum('geometry' in r for r in records),'available':len(pool),
            'rejections':dict(rejected),'limit':config.page_final_candidate_limit,
            'elapsed_seconds':time.perf_counter()-started,'cooperative_finalization_seconds':config.joint_finalize_seconds,
            'unverified_states_published':False,'page_candidates_rechecked':True}


class GeometryBank:
    """Protect both old-score and geometric winners, before old beam pruning."""
    def __init__(self,context,deadline):
        self.context,self.deadline=context,deadline;self.states={};self.origins={}
        self.seen=set();self.counts=Counter();self.geometry_scores={}
        self.ids={};self.families={};self.topologies={}
        self.source_scores=[]

    def consider(self,state,origin):
        if len(state.components())!=1:return
        fingerprint_key=fingerprint(state)
        if fingerprint_key in self.seen:
            self.counts['duplicate_completions']+=1;return
        if len(self.seen)>=LIMITS['completed_feature_checks'] or time.perf_counter()>=self.deadline:
            self.counts['feature_budget_abstentions']+=1;return
        self.seen.add(fingerprint_key);self.counts['completed_states_measured']+=1
        f=self.context.quick(state)
        if not f['rigid'] or not f['component_bounds']['valid']:
            self.counts['impossible_component_rejections']+=1;return
        candidate=clone(state)
        candidate.score=float(self.context.problem.structural.evaluate(candidate)['score'])
        self.source_scores.append(float(candidate.score))
        key=state_key(candidate)
        self.states[key]=(candidate.score,candidate);self.origins[key]=origin
        self.geometry_scores[key]=self.context.coarse_geometry(candidate)['score']
        self.ids[key]=fingerprint_key;self.families[key]=self.context.pose_family(candidate)
        self.topologies[key]=tuple(sorted((self.context.edges[e].a,self.context.edges[e].b) for e in candidate.selected))
        self._trim()

    def _trim(self):
        c=self.context.config
        by_source=sorted(self.states,key=lambda k:(-self.states[k][0],self.ids[k]))
        by_geometry=sorted(self.states,key=lambda k:(-self.geometry_scores[k],-self.states[k][0],self.ids[k]))
        keep=set()
        for ranked,limit in ((by_source,c.page_top_k),(by_geometry,c.page_geometry_k)):
            families=set()
            for key in ranked:
                family=self.families[key]
                if family in families:continue
                families.add(family);keep.add(key)
                if len(families)>=limit:break
        topology=set();extra=0
        for key in by_source:
            signature=self.topologies[key]
            if key not in keep and signature not in topology and extra<c.page_diversity_k:
                keep.add(key);topology.add(signature);extra+=1
        self.states={k:self.states[k] for k in keep};self.origins={k:self.origins[k] for k in keep}
        self.geometry_scores={k:self.geometry_scores[k] for k in keep}
        self.ids={k:self.ids[k] for k in keep};self.families={k:self.families[k] for k in keep}
        self.topologies={k:self.topologies[k] for k in keep}

    def finalists(self):
        return [self.states[k][1] for k in sorted(self.states,key=lambda k:(-self.geometry_scores[k],-self.states[k][0],self.ids[k]))]

    def report(self):
        c=self.context.config
        return {'policy':POLICY,'counts':dict(self.counts),'retained':len(self.states),
            'solver_top_k':c.page_top_k,'geometry_top_k':c.page_geometry_k,'diversity_slots':c.page_diversity_k,
            'retention_before_old_score_pruning':True,'baseline_finalists_protected_separately':True,
            'approximate_pose_clustering_used_for_diversity':True,'pose_bucket_native_px':self.context.pose_step,
            'seen_fingerprints_stored':len(self.seen),'maximum_complete_measurements':LIMITS['completed_feature_checks'],
            'retained_candidates':[{'candidate_id':self.ids[k],
                'old_score_rank_among_measured_completions':int(1+sum(v>self.states[k][0] for v in self.source_scores)),
                'coarse_geometry_score':self.geometry_scores[k],'origin':self.origins[k]}
                for k in sorted(self.states,key=lambda k:(-self.geometry_scores[k],self.ids[k]))],
            'is_certified_global_k_best':False,'evaluation_truth_opened':False}

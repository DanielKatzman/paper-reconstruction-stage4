"""Version 3 recovery: occupied ambiguous seams and fair region allocation."""
from itertools import product
from .outputs import digest
from .page import POLICY


def build_request(problem,config,result,assessment,binding):
    layout=result['layouts'][0];features=assessment['features'];contacts=features.get('contacts',{})
    groups=sorted(layout['components'],key=lambda g:(-len(g),g))
    owners={p:g for g in groups for p in g};uncovered=contacts.get('uncovered_boundary_intervals',{})
    inventory=sorted(owners);buckets={k:[] for k in ('check_join','connect_components','uncovered_piece','rebuild_component')}

    def region(kind,components,reason,join=None,pairs=None):
        pieces=sorted({p for group in components for p in group})
        spans=(join['native_spans'] if join else [{'piece_id':p,'intervals':uncovered[p]} for p in pieces if uncovered.get(p)])
        pairs=sorted({tuple(sorted(p)) for p in (pairs or []) if p[0]!=p[1]})
        eid=join['geometry_id'] if join else None
        return {'region_id':digest([kind,components,eid,spans]),'region_kind':kind,
            'piece_ids':pieces,'component_piece_ids':components,'native_spans':spans,
            'suspect_geometry_id':eid,'suspect_occurrence_id':join['occurrence_id'] if join else None,
            'available_geometry_ids':sorted(e.id for e in problem.edges if (e.a,e.b) in pairs),
            'reason':reason,'published_alternative_tests':[],
            'candidate_piece_pairs':[list(p) for p in pairs],
            'pair_possibilities_exceed_budget':len(pairs)>8,
            'untested_pairs_must_be_reported':True,
            'occupied_spans_may_be_reconsidered':kind=='check_join',
            'request':'Investigate distinct unit-scale rigid placement basins using observed native contour. Component memberships and current poses are revisable alternatives, not fixed true relationships. Report untested pairs/windows and exclusions; outer page border is not automatically a tear.'}

    if assessment['status']!='supported':
        joins={j['geometry_id']:j for j in contacts.get('joins',[])}
        uncertain=features.get('ambiguity',{}).get('joins',[])
        for row in sorted((r for r in uncertain if r['unresolved_sliding']),
                          key=lambda r:(-r['risk_cost'],r['native_length_px'],r['geometry_id'])):
            join=joins[row['geometry_id']];whole=set(owners[join['pieces'][0]])
            first=sorted(whole&set(join['cut']));second=sorted(whole-set(first))
            if first and second:
                buckets['check_join'].append(region('check_join',[first,second],
                    'unresolved_straight_or_underconstrained_bridge',join,list(product(first,second))))
        if len(groups)>1:
            for other in groups[1:]:
                buckets['connect_components'].append(region('connect_components',[groups[0],other],
                    'disconnected_components',pairs=list(product(groups[0],other))))
        for group in groups:
            if len(group)==1:
                pid=group[0]
                buckets['uncovered_piece'].append(region('uncovered_piece',[group],
                    'insufficient_pose_coverage_observed',pairs=[(pid,p) for p in inventory if p!=pid]))
        for bound in features.get('component_bounds',{}).get('components',[]):
            if not bound['valid']:
                pieces=bound['pieces']
                buckets['rebuild_component'].append(region('rebuild_component',[pieces],
                    'component_exceeds_page_area_budget',pairs=[(a,b) for a in pieces for b in pieces if a<b]))
    regions=[];ordered=[]
    # A bounded fair schedule reserves both occupied-join and coverage requests.
    # The largest observed group is the hub, not an arbitrary small singleton.
    while any(buckets.values()):
        for values in buckets.values():
            if values:ordered.append(values.pop(0))
    if assessment['status']!='supported' and not ordered:
        ordered=[region('uncovered_component',[groups[0]],'insufficient_pose_coverage_observed')]
    regions=ordered[:config.recovery_region_limit]
    omitted=[{'region_kind':r['region_kind'],'piece_ids':r['piece_ids'],'reason':'region_budget'}
             for r in ordered[config.recovery_region_limit:]]
    return {'schema_version':'stage4.pose-recovery-request.v3','policy':POLICY,'binding':binding,
        'request_policy':'occupied-and-unoccupied-native-recovery.v1',
        'assessment_status':assessment['status'],'reason_codes':assessment['reason_codes'],
        'motion_model':'proper_rigid_unit_scale','regions':regions,'regions_omitted_by_budget':omitted,
        'observed_components':layout['components'],'uncovered_boundary_intervals':uncovered,
        'uncovered_regions_may_be_outer_page_border':True,
        'current_component_poses_are_revisable':True,
        'budget':{'rounds':1,'regions':config.recovery_region_limit,'poses_per_region':config.recovery_poses_per_region,
            'maximum_new_rigid_poses':config.recovery_region_limit*config.recovery_poses_per_region,
            'maximum_pair_investigations_per_region':8},
        'budget_is_maximum_not_sufficiency_claim':True,'evaluation_truth_opened':False,
        'producer_response_required':['input/profile identities','unit-scale rigid pose/window provenance',
            'pair/window work and untested possibilities','exclusions and budget reasons',
            'useful existing poses preserved as alternatives','occupied suspect seams reconsidered when requested']}

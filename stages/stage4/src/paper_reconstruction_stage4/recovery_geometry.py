"""Observed disconnected-component recovery requests, version 2."""
from .outputs import digest
from .page import POLICY


def build_request(problem,config,result,assessment,binding):
    layout=result['layouts'][0];features=assessment['features'];contacts=features.get('contacts',{})
    groups=sorted(layout['components'],key=lambda g:(len(g),g))
    uncovered=contacts.get('uncovered_boundary_intervals',{})
    bounds=features.get('component_bounds',{}).get('components',[])
    regions=[];omitted=[];limit=config.recovery_region_limit

    def append(kind,components,reason,geometry=None,occurrence=None,spans=None):
        pieces=sorted({p for group in components for p in group})
        spans=spans if spans is not None else [{'piece_id':p,'intervals':uncovered.get(p,[])} for p in pieces if uncovered.get(p)]
        value={'region_id':digest([kind,components,geometry,spans]),'region_kind':kind,
            'piece_ids':pieces,'component_piece_ids':components,
            'suspect_geometry_id':geometry,'suspect_occurrence_id':occurrence,
            'native_spans':spans,'reason':reason,
            'available_geometry_ids':sorted(e.id for e in problem.edges if e.a in pieces and e.b in pieces),
            'published_alternative_tests':[],
            'request':'Inspect observed unoccupied native contour and distinct proper-rigid displacement basins; preserve accurate existing poses. Outer page borders are not tear matches.'}
        if len(regions)<limit:regions.append(value)
        else:omitted.append({'region_kind':kind,'piece_ids':pieces,'reason':'region_budget'})

    if assessment['status']!='supported':
        # Isolated pieces get an explicit slot even if selected joins exist.
        for group in groups:
            if len(group)==1:append('uncovered_piece',[group],'insufficient_pose_coverage_observed')
        # One request covers both disconnected components and their observed
        # boundary regions, without guessing a true neighboring piece pair.
        if len(groups)>1:
            for index in range(1,len(groups)):
                append('connect_components',[groups[0],groups[index]],'disconnected_components')
        for bound in bounds:
            if not bound['valid']:
                append('rebuild_component',[bound['pieces']],'component_exceeds_page_area_budget')
        weak=set(contacts.get('unresolved_weak_geometry_ids',[]))
        for join in sorted(contacts.get('joins',[]),key=lambda j:(j['native_length_over_sqrt_smaller_area'],j['geometry_id'])):
            if join['geometry_id'] not in weak:continue
            append('check_join',[join['pieces']],'short_straight_ambiguous_attachment',
                   join['geometry_id'],join['occurrence_id'],join['native_spans'])
        if not regions:
            append('uncovered_component',[groups[0]],'insufficient_pose_coverage_observed')
    return {'schema_version':'stage4.pose-recovery-request.v2','policy':POLICY,'binding':binding,
        'assessment_status':assessment['status'],'reason_codes':assessment['reason_codes'],
        'motion_model':'proper_rigid_unit_scale','regions':regions,
        'observed_components':layout['components'],'regions_omitted_by_budget':omitted,
        'uncovered_boundary_intervals':uncovered,'uncovered_regions_may_be_outer_page_border':True,
        'budget':{'rounds':1,'regions':limit,'poses_per_region':config.recovery_poses_per_region,
            'maximum_new_rigid_poses':limit*config.recovery_poses_per_region,
            'maximum_pair_investigations_per_region':8},
        'budget_is_maximum_not_sufficiency_claim':True,'evaluation_truth_opened':False,
        'producer_response_required':['input/profile identities','rigid pose provenance','exclusions and budget reasons',
                                      'preservation of existing accurate proposals','pair investigations and correspondence work per region']}

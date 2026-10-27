"""Sealed, reference-free assessment and bounded upstream recovery requests."""
import json
import os
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator
from .model import State
from .outputs import atomic_json, digest, file_hash, implementation_identity, read_result
from .page import make_page_context, POLICY
from .active_support import ActiveSupportIndex, verify_result_support
from .feasibility import Feasibility


def state_from_layout(layout):
    joins = layout['selected_joins']
    return State({p['piece_id']: np.asarray(p['native_to_component'], float) for p in layout['pieces']},
        {p['piece_id']: p['component_id'] for p in layout['pieces']},
        tuple(j['geometry_id'] for j in joins), tuple(j['geometry_id'] for j in joins if j['tree_join']),
        {j['geometry_id']: j['occurrence_id'] for j in joins}, layout['objective_score'],
        {j['geometry_id']: j['active_support'] for j in joins if j.get('active_support')})


def validate_recovery(value):
    names={'stage4.pose-recovery-request.v1':'pose-recovery-request.schema.json',
           'stage4.pose-recovery-request.v2':'pose-recovery-request-v2.schema.json',
           'stage4.pose-recovery-request.v3':'pose-recovery-request-v3.schema.json'}
    name=names.get(value.get('schema_version'))
    if name is None:raise ValueError('Unknown recovery request version')
    schema = json.loads((Path(__file__).parent/'schemas'/name).read_text())
    Draft202012Validator(schema).validate(value)
    envelope = value['budget']
    if len(value['regions']) > envelope['regions'] or envelope['maximum_new_rigid_poses'] != envelope['regions']*envelope['poses_per_region']:
        raise ValueError('Recovery request budget/inventory differs')
    if value['schema_version']=='stage4.pose-recovery-request.v3':
        inventory={p for group in value['observed_components'] for p in group}
        for region in value['regions']:
            pairs=region['candidate_piece_pairs'];unique={tuple(p) for p in pairs}
            if len(unique)!=len(pairs) or any(a==b or a not in inventory or b not in inventory or a>b for a,b in pairs):
                raise ValueError('Invalid observed recovery pair inventory')
            if region['pair_possibilities_exceed_budget'] != (len(pairs)>envelope['maximum_pair_investigations_per_region']):
                raise ValueError('Recovery pair budget disclosure differs')


def recovery_request(problem, config, result, assessment, binding):
    from .config import GEOMETRY_SELECTION, AMBIGUITY_SELECTION
    if config.page_selection_policy == AMBIGUITY_SELECTION:
        from .recovery_ambiguity import build_request
        value=build_request(problem,config,result,assessment,binding)
        validate_recovery(value)
        return value
    if config.page_selection_policy == GEOMETRY_SELECTION:
        from .recovery_geometry import build_request
        value = build_request(problem,config,result,assessment,binding)
        validate_recovery(value)
        return value
    f = assessment['features']; regions = []
    contacts = f.get('contacts', {})
    if assessment['status'] != 'supported':
        weak = set(contacts.get('unresolved_weak_geometry_ids', []))
        joins = contacts.get('joins', [])
        joins = sorted(joins, key=lambda j: (j['geometry_id'] not in weak,
                         j['native_length_over_sqrt_smaller_area'], j['shape_spread_ratio'], j['geometry_id']))
        for join in joins[:config.recovery_region_limit]:
            alternatives = [e for e in problem.edges if {e.a, e.b} == set(join['pieces'])]
            selected = join['geometry_id']
            regions.append({'region_id': digest(join['native_spans']), 'piece_ids': join['pieces'],
                'suspect_geometry_id': selected, 'suspect_occurrence_id': join['occurrence_id'],
                'native_spans': join['native_spans'],
                'reason': 'short_straight_ambiguous_attachment' if selected in weak else 'page_or_alternative_disagreement',
                'available_geometry_ids': sorted(e.id for e in alternatives),
                'published_alternative_tests': [{'layout_index': l['layout_index'],
                    'geometry_ids': [j['geometry_id'] for j in l['selected_joins'] if set((j['piece_a'],j['piece_b'])) == set(join['pieces'])]}
                    for l in result['layouts']],
                'request': 'broader native contour correspondence and alternative rigid displacement basins; preserve accurate existing poses'})
        # Partial candidates may have no uncertain joins. Request only observed
        # uncovered regions, never infer a hidden neighbor identity.
        if not regions:
            for pid, intervals in sorted(contacts.get('uncovered_boundary_intervals', {}).items()):
                if intervals:
                    regions.append({'region_id': digest([pid, intervals]), 'piece_ids': [pid],
                        'suspect_geometry_id': None, 'suspect_occurrence_id': None,
                        'native_spans': [{'piece_id': pid, 'intervals': intervals}],
                        'reason': 'insufficient_pose_coverage_observed', 'available_geometry_ids': [],
                        'published_alternative_tests': [],
                        'request': 'search unoccupied observed contour regions for rigid correspondences; exclude outer page border where identifiable'})
                if len(regions) == config.recovery_region_limit:
                    break
    if assessment['status'] != 'supported' and not regions:
        for pid in sorted(problem.pieces):
            xy = problem.pieces[pid].contour['points_xy']
            circumference = float(np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1).sum())
            regions.append({'region_id': digest([pid,circumference]),'piece_ids':[pid],
                'suspect_geometry_id':None,'suspect_occurrence_id':None,
                'native_spans':[{'piece_id':pid,'intervals':[[0.,circumference]]}],
                'reason':'insufficient_pose_coverage_observed','available_geometry_ids':[],
                'published_alternative_tests':[],
                'request':'search observed unoccupied boundary; exclude outer page border and occupied spans'})
            if len(regions) == config.recovery_region_limit:
                break
    request = {'schema_version': 'stage4.pose-recovery-request.v1', 'policy': POLICY, 'binding': binding,
        'assessment_status': assessment['status'], 'reason_codes': assessment['reason_codes'],
        'motion_model': 'proper_rigid_unit_scale', 'regions': regions,
        'uncovered_boundary_intervals': contacts.get('uncovered_boundary_intervals', {}),
        'uncovered_regions_may_be_outer_page_border': True,
        'budget': {'rounds': 1, 'regions': config.recovery_region_limit, 'poses_per_region': config.recovery_poses_per_region,
                   'maximum_new_rigid_poses': config.recovery_region_limit*config.recovery_poses_per_region},
        'budget_is_maximum_not_sufficiency_claim': True, 'evaluation_truth_opened': False,
        'producer_response_required': ['input/profile identities', 'rigid pose provenance', 'exclusions and budget reasons', 'preservation of existing accurate proposals']}
    validate_recovery(request)
    return request


def verify_package(root, binding=None):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version') != 'stage4.assessment-manifest.v1' or manifest.get('manifest_sha256') != digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Invalid assessment package seal')
    if binding is not None and manifest['binding'] != binding:
        raise ValueError('Assessment package belongs to a different result/profile/policy')
    if set(manifest['files']) != {'assessment.json', 'pose_recovery_request.json', 'report.md'}:
        raise ValueError('Unexpected assessment package inventory')
    for name, sha in manifest['files'].items():
        if file_hash(root/name) != sha:
            raise ValueError('Assessment package file changed: '+name)
    value = json.loads((root/'pose_recovery_request.json').read_text()); validate_recovery(value)
    if value['binding'] != manifest['binding']:
        raise ValueError('Recovery request binding differs')
    return manifest


def write_package(problem, config, result_path, output=None):
    """Adopt an authenticated published result; never solve or read references."""
    from .constraints import require_bound
    require_bound(problem, config)
    if not problem.constraint_profile:
        raise ValueError('Assessment requires an explicit complete-page profile')
    result_path = Path(result_path).resolve(); result = read_result(result_path)
    if result['input']['manifest_sha256'] != problem.manifest['manifest_sha256']:
        raise ValueError('Assessment input/result bundle mismatch')
    from .native_bridge import install_result_bridges
    install_result_bridges(problem,result,config)
    output = Path(output).resolve() if output else result_path.with_name(result_path.name+'.assessment')
    if output == result_path or output.is_relative_to(result_path) or result_path.is_relative_to(output) or output.is_relative_to(problem.root):
        raise ValueError('Assessment must be outside immutable input and prediction trees')
    binding = {'input_manifest_sha256': problem.manifest['manifest_sha256'],
        'result_sha256': file_hash(result_path/'result.json'), 'source_run_id': result['run_id'],
        'source_implementation_sha256': result['implementation']['sha256'],
        'assessment_implementation_sha256': implementation_identity()['sha256'],
        'configuration_sha256': digest(config.to_dict()), 'constraint_profile': problem.constraint_profile}
    if (output/'manifest.json').exists():
        return verify_package(output, binding)
    if output.exists():
        raise ValueError('Nonempty or unpublished assessment output; choose a separate path')
    work = output.with_name(output.name+'.work'); work.mkdir(parents=True, exist_ok=True)
    prior = work/'binding.json'
    if prior.exists() and json.loads(prior.read_text()) != binding:
        raise ValueError('Unpublished assessment work identity differs')
    lock = work/'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        atomic_json(prior, binding)
        support = ActiveSupportIndex(problem, config)
        context = make_page_context(problem, config, support)
        states = [state_from_layout(l) for l in result['layouts']]
        # Source result scores may already be page utilities. Restore v8 bases
        # explicitly for prospective assessment/alternative comparison.
        from .joint_score import JointScorer
        scorer = JointScorer(problem, config, support)
        from .config import AMBIGUITY_SELECTION
        if config.page_selection_policy == AMBIGUITY_SELECTION:context.ambiguity_scorer=scorer
        for state in states:
            state.score = scorer.evaluate(state)['score']
        feasibility = Feasibility(problem, config)
        assessments = []
        for i, state in enumerate(states):
            assessment = context.assess(state, [s for s in states if s is not state])
            physical = feasibility.state(state, final=True)
            try:
                verify_result_support(problem, {'layouts': [result['layouts'][i]]}, config)
                support_valid = True
            except ValueError:
                support_valid = False
            assessment['native_final_validation'] = physical
            assessment['native_support_valid'] = support_valid
            if not physical['valid'] or not support_valid:
                assessment['status'] = 'unresolved' if assessment['status'] != 'profile_violation' else 'profile_violation'
                assessment['reason_codes'] = sorted(set(assessment['reason_codes']+['conflicting_native_support']))
            assessments.append(assessment)
        primary = assessments[0]
        request = recovery_request(problem, config, result, primary, binding)
        atomic_json(work/'assessment.json', {'binding': binding, 'layouts': assessments, 'feature_accounting': context.report(),
                                           'serving_prediction_modified': False, 'evaluation_truth_opened': False})
        atomic_json(work/'pose_recovery_request.json', request)
        report = ('# Observed reconstruction assessment\n\n'+
            f"Status: **{primary['status']}**. Candidate connectivity: {result['status']}.\n\n"+
            'Reasons: '+(', '.join(primary['reason_codes']) or 'page and measured support checks passed')+'.\n\n'+
            f"Recovery regions: {len(request['regions'])}; maximum one round, {request['budget']['maximum_new_rigid_poses']} new rigid poses.\n\n"+
            'No reference geometry, true neighbor labels or raw-fit diagnostic traces were read. This assessment is uncalibrated; a rectangular silhouette does not guarantee correct content order.\n')
        if request['schema_version'] in {'stage4.pose-recovery-request.v2','stage4.pose-recovery-request.v3'}:
            report += '\nPage geometry: '+primary['geometry']['status']+'.\n\n'
            report += '\n'.join('- '+region['region_kind']+': '+', '.join(region['piece_ids'])+' — '+region['reason'] for region in request['regions'])+'\n'
        (work/'report.md').write_text(report, encoding='utf-8')
        manifest = {'schema_version': 'stage4.assessment-manifest.v1', 'binding': binding,
                    'files': {n: file_hash(work/n) for n in ('assessment.json', 'pose_recovery_request.json', 'report.md')}}
        manifest['manifest_sha256'] = digest(manifest)
        staging = work/'publication'; staging.mkdir(exist_ok=True)
        for name in manifest['files']:
            os.replace(work/name, staging/name)
        atomic_json(staging/'manifest.json', manifest)
        os.replace(staging, output)
        return manifest
    finally:
        lock.unlink(missing_ok=True)

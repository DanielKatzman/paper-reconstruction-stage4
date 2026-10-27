"""Authenticate and independently replay the completed fourteen-case V21 run.

Post-seal review only: no search, pose fitting, prediction replacement, or
upstream inference is performed. The exact frozen implementation is loaded.
"""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit,unquote

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/v21_cut_repair_full_r4')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'outputs') or out==ROOT/'outputs' or out.exists():
        raise ValueError('Choose a new review folder inside Stage 4 outputs')
    request=json.loads((run/'request.json').read_text(encoding='utf8'))
    snapshot=Path(request['snapshot']).resolve()
    receipt=json.loads((snapshot/'receipt.json').read_text())
    for name,sha in receipt['files'].items():
        p=(snapshot/name).resolve()
        if not p.is_relative_to(snapshot) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('Changed frozen source: '+name)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(snapshot/'src'))
    import numpy as np
    from benchmark.common import read_json,file_hash,digest,atomic_json,verify_files
    from benchmark.cut_repair_experiment_v21_r4 import authenticate,checked,verify_predictions,command_for
    from paper_reconstruction_stage4.outputs import implementation_identity,read_result
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.config import SolverConfig
    from paper_reconstruction_stage4.constraints import bind_profile
    from paper_reconstruction_stage4.active_support import ActiveSupportIndex,verify_result_support
    from paper_reconstruction_stage4.native_bridge_v2 import verify_layout as verify_native
    from paper_reconstruction_stage4.native_bridge_v3 import verify_layout as verify_boundary
    from paper_reconstruction_stage4.feasibility import Feasibility
    from paper_reconstruction_stage4.model import State
    from paper_reconstruction_stage4.page import make_page_context
    from paper_reconstruction_stage4.component_closure import shape_measurement
    from evaluation.evaluate import evaluate
    from benchmark.quality import assembly_quality
    assert implementation_identity()==request['implementation']
    authenticate(request)
    progress=read_json(run/'progress.json');assert progress['phase']=='finished'
    barrier=checked(read_json(run/'predictions_sealed.json'),'barrier_sha256')
    assert barrier['references_opened'] is False and barrier['request_sha256']==request['request_sha256']
    verified=verify_predictions(run,request,progress)
    assert all(barrier[k]==v for k,v in verified.items())
    report_receipt=read_json(run/'experiment_receipt.json')
    assert report_receipt['barrier_sha256']==barrier['barrier_sha256'] and report_receipt['request_sha256']==request['request_sha256']
    verify_files(run,report_receipt['files'])
    summary=read_json(run/'report/summary.json')
    assert summary['cold_stage4_cases']==14 and summary['references_opened_after_seal'] is True
    assert summary['old_layouts_supplied_to_solver'] is False and summary['diagnostic_witnesses_count_as_blind_success'] is False
    assert Counter(c['pieces'] for c in request['cases'])=={4:2,8:2,12:2,16:2,20:2,24:2,30:2}
    reported={c['case_id']:c for c in summary['cases']};rows=[];out.mkdir()
    for i,case in enumerate(request['cases'],1):
        cid=case['case_id'];print(f'[{i}/14] Independent geometry, collision, native-support and reference replay: {cid}',flush=True)
        started=time.perf_counter();entry=progress['cases'][cid]
        assert entry['status']=='predicted' and len(entry['attempts'])==1
        attempt=entry['attempts'][0]
        assert attempt['returncode']==0 and attempt['existing_result_before_invocation'] is False
        assert attempt['command']==list(map(str,command_for(run,request,case)))
        assert '--initial-result' not in attempt['command'] and case['evaluation_reference'] not in attempt['command']
        result=read_result(run/'cases'/cid/'stage4');layout=result['layouts'][0]
        config=SolverConfig.from_dict(result['configuration'])
        problem=load_problem(case['bundle'],config);bind_profile(problem,config,request['profile'],case['capture_profile'])
        assert len(problem.pieces)==case['pieces'] and problem.constraint_profile==case['constraint_binding']
        support=verify_result_support(problem,result,config)
        native_count=verify_native(problem,layout,config);boundary_count=verify_boundary(problem,layout,config)
        poses={p['piece_id']:np.asarray(p['native_to_component']) for p in layout['pieces']}
        groups={p['piece_id']:p['component_id'] for p in layout['pieces']}
        assert set(poses)==set(problem.pieces) and len(set(groups.values()))==1
        selected=tuple(j['geometry_id'] for j in layout['selected_joins'])
        state=State(poses,groups,selected,(),{j['geometry_id']:j['occurrence_id'] for j in layout['selected_joins']})
        context=make_page_context(problem,config,ActiveSupportIndex(problem,config));problem.page_context=context
        physical=Feasibility(problem,config).state(state,final=True)
        geometry=shape_measurement(context,state,768)
        assert physical['valid'] and geometry['shape_gate_pass'] and support['valid']
        assert file_hash(case['evaluation_reference'])==case['evaluation_reference_sha256']
        ref={**read_json(case['evaluation_reference']),'input_manifest_sha256':case['bundle_sha256']}
        evaluation=evaluate(result,ref);quality=assembly_quality(result,ref,evaluation)
        q=quality['common_gauge'];pieces=q['per_piece']
        assert q['whole_page_error_available'] and not evaluation['false_neighbor_joins']
        assert q['non_anchor_probe_errors_reference_px']['maximum']<=30.000001
        assert max(abs(p.get('local_orientation_error_degrees') or 0.) for p in pieces)<1e-6
        assert all(abs(p.get('local_scale_ratio',1.)-1.)<1e-6 for p in pieces)
        old=reported[cid];new=old['candidate'];baseline=old['baseline']
        # The evaluator returns some pair tuples, while saved JSON contains
        # equivalent lists. Compare their canonical JSON representation.
        assert digest(new['quality'])==digest(quality) and digest(new['evaluation'])==digest(evaluation)
        assert abs(geometry['score']-new['assessment']['geometry']['score'])<1e-8
        audit=result['search']['native_repair'];cut=audit['unsupported_cut_repair']
        row={'case_id':cid,'pieces':case['pieces'],'cold_completed_attempts':1,
            'result_sha256':file_hash(run/'cases'/cid/'stage4/result.json'),
            'input_manifest_sha256':case['bundle_sha256'],'reference_sha256':case['evaluation_reference_sha256'],
            'connected':True,'native_joins_replayed':native_count,'boundary_joins_replayed':boundary_count,
            'selected_joins':len(selected),'independent_support':support,
            'final_physical':{'valid':physical['valid'],'pairs_checked':len(physical['pairs']),
                'maximum_sampled_overlap':physical['maximum_overlap'],'fidelity':physical['fidelity']},
            'independent_geometry':geometry,'whole_page_p95_px':q['non_anchor_probe_errors_reference_px']['p95'],
            'maximum_probe_error_px':q['non_anchor_probe_errors_reference_px']['maximum'],
            'max_per_piece_p95_px':max(p.get('p95_reference_px') or 0. for p in pieces),
            'false_selected_neighbor_pairs':len(evaluation['false_neighbor_joins']),
            'worker_seconds':new['stage4_worker_seconds'],'solver_seconds':result['search']['elapsed_solver_seconds'],
            'baseline_worker_seconds':baseline['stage4_worker_seconds'],
            'baseline_page_p95_px':baseline['quality']['common_gauge']['non_anchor_probe_errors_reference_px']['p95']
                if baseline['quality']['common_gauge']['whole_page_error_available'] else None,
            'cut_verified_completions':cut.get('verified_completions',0),
            'cut_alternatives_examined':cut.get('alternatives_examined',0),
            'native_fitting_seconds':sum((audit.get(k) or {}).get('elapsed_seconds',0.) for k in ('native_generation','expanded_native_generation')),
            'scheduler_seconds':audit.get('scheduler_work',{}).get('seconds_by_task',{}),
            'dfs_sessions_started':sum(x.get('backtracking_seed_sessions_started',0) for x in audit['merge_searches']),
            'backtracks':sum(x.get('backtracks',0) for x in audit['merge_searches']),
            'review_seconds':time.perf_counter()-started}
        atomic_json(out/cid/'audit.json',row);rows.append(row)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src') and v)
    links=0
    for path in (run/'report').rglob('*.html'):
        parsed=Links();parsed.feed(path.read_text(encoding='utf8'))
        for value in parsed.links:
            url=urlsplit(value)
            if not url.scheme and url.path:
                assert (path.parent/unquote(url.path)).resolve().exists(),(path,value)
                links+=1
    review={'schema_version':'stage4.v21-full-independent-review.v1','run':str(run),
        'implementation_sha256':request['implementation']['sha256'],
        'request_sha256':request['request_sha256'],'prediction_barrier_sha256':barrier['barrier_sha256'],
        'source_report_receipt_sha256':file_hash(run/'experiment_receipt.json'),
        'original_prediction_files_verified':len(barrier['files']),
        'original_report_files_verified':len(report_receipt['files']),
        'fresh_independent_replay_cases':len(rows),'case_results':rows,
        'all_cases_supported_connected_and_zero_false_neighbors':True,
        'max_probe_error_across_all_cases_px':max(r['maximum_probe_error_px'] for r in rows),
        'worker_seconds_total':sum(r['worker_seconds'] for r in rows),
        'baseline_worker_seconds_total':sum(r['baseline_worker_seconds'] for r in rows),
        'html_local_links_checked':links,'html_missing_local_links':0,
        'primary_results_changed':False,'solver_executed_during_review':False,
        'visual_review':'Record manual visual inspection separately; this script does not inspect pixels semantically.',
        'script_sha256':file_hash(Path(__file__))}
    assert all(file_hash(p)==h for p,h in barrier['files'].items())
    atomic_json(out/'review.json',review)
    print('Independent replay complete: 14/14; saved review:',out/'review.json',flush=True)


if __name__=='__main__':main()

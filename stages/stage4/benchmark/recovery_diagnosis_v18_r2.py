"""Reference-aware V18 comparison; no truth is opened before the full seal."""
import argparse
from pathlib import Path
import numpy as np
from paper_reconstruction_contracts import map_points
from paper_reconstruction_stage4.outputs import read_result,implementation_identity
from .common import atomic_json,digest,file_hash,inventory,read_json,verify_files
from .recovery_experiment_v18_r2 import checked,require,verify_predictions


def retained_accuracy(result,reference):
    """Measure saved poses only; never generate, refine or replace a primary."""
    from paper_reconstruction_stage4.page_geometry import record_state
    reference_poses={p:np.asarray(r['native_to_reference'],float) for p,r in reference['pieces'].items()}
    probes={p['piece_id']:np.asarray(p['native_probes'],float) for p in result['layouts'][0]['pieces']}
    bank=result['search']['geometry_candidate_audit'];rows=[]
    for raw in bank['candidates']:
        state=record_state(raw);groups=sorted(state.components(),key=lambda g:(-len(g),min(g)))
        complete=len(groups)==1;error=None
        if complete:
            root=min(state.poses);align=reference_poses[root]@np.linalg.inv(state.poses[root]);errors=[]
            for pid in sorted(state.poses):
                if pid==root:continue
                errors.extend(np.linalg.norm(map_points(probes[pid],align@state.poses[pid])-
                                             map_points(probes[pid],reference_poses[pid]),axis=1).tolist())
            error=float(np.quantile(errors,.95)) if errors else 0.
        rows.append({'candidate_id':raw['candidate_id'],'selected':raw['candidate_id']==bank['selected_candidate_id'],
            'complete':complete,'component_sizes':list(map(len,groups)),'disposition':raw['disposition'],
            'whole_page_p95_reference_px':error,'geometry':raw.get('geometry')})
    complete=[r for r in rows if r['complete']]
    best=min(complete,key=lambda r:r['whole_page_p95_reference_px']) if complete else None
    return {'records':len(rows),'complete_candidates':len(complete),'best_complete_by_reference_diagnostic':best,
        'candidates':rows,'is_certified_global_top_k':False,'primary_changed_by_diagnostic':False}


def native_diagnosis(result,reference,upstream):
    from .diagnostics import pose_error
    from .scale_pipeline_diagnosis_v14 import accurate_graph
    native=result['search']['native_repair'];generation=native.get('expanded_native_generation') or native.get('native_generation') or {}
    pair_rows=generation.get('pairs',[]);contacts={(c['piece_a'],c['piece_b']):c for c in reference['contacts']}
    best={pair:None for pair in contacts}
    for proposal in native.get('generated_pose_proposals',[]) or []:
        pair=tuple(proposal['pieces'])
        if pair not in contacts:continue
        error=pose_error(proposal['transform_b_to_a'],contacts[pair])
        if error is not None:best[pair]=min(error,best[pair]) if best[pair] is not None else error
    rows=[];lookup={tuple(r['pieces']):r for r in pair_rows}
    for original in upstream['stages']['stage3'].get('contacts',[]):
        pair=original['piece_a'],original['piece_b'];extra=best[pair]
        values=[v for v in (original['archive_best_p95_px'],extra) if v is not None]
        rows.append({**original,'native_best_p95_px':extra,
            'archive_or_native_best_p95_px':min(values) if values else None,
            'native_pair_audit':lookup.get(pair)})
    per_piece={p:{'possible_pairs':len(reference['pieces'])-1,
        'fitted_pairs':sum(r['native_checks']>0 and p in r['pieces'] for r in pair_rows),
        'native_checks':sum(r['native_checks'] for r in pair_rows if p in r['pieces'])} for p in reference['pieces']}
    return {'possible_physical_pairs':len(reference['pieces'])*(len(reference['pieces'])-1)//2,
        'pairs_with_fitted_checks':sum(r['native_checks']>0 for r in pair_rows),
        'native_checks':generation.get('native_checks',0),'maximum_native_checks':generation.get('maximum_native_checks'),
        'generation_budget_abstained':bool(generation.get('budget_abstained',False)),
        'merge_budget_abstained':any(r['budget_abstained'] for r in native['merge_searches']),
        'route_budget_abstained':native.get('route_budget_abstained',native['budget_abstained']),
        'reported_untested_pairs':generation.get('untested_pair_count'),
        'per_piece_coverage':per_piece,'frontier_coverage':generation.get('frontier_coverage'),
        'contact_availability':rows,
        'accuracy_graphs':{name:[accurate_graph(rows,reference['pieces'],field,t) for t in (4,16,32,48)]
            for name,field in [('native','native_best_p95_px'),('archive_or_native','archive_or_native_best_p95_px')]},
        'scope':'Pair coverage includes V14/V18 repair proposals and the unchanged Stage 3 archive. It omits other derived routes/refinement and does not prove compatible support or global impossibility.'}


def measure(case,target,request,reference,upstream,label):
    from evaluation.evaluate import evaluate
    from .quality import assembly_quality
    from paper_reconstruction_stage4.config import SolverConfig,REPAIR_V15_POLICIES,REPAIR_V18_POLICY
    from paper_reconstruction_stage4.inputs import load_problem
    from paper_reconstruction_stage4.constraints import bind_profile
    from paper_reconstruction_stage4.active_support import verify_result_support
    from paper_reconstruction_stage4.native_bridge import install_result_bridges
    from paper_reconstruction_stage4.page_geometry import GeometryContext
    from paper_reconstruction_stage4.model import State
    from paper_reconstruction_stage4.repair_search import extent_hypothesis,fits_extent
    result=read_result(target);config=SolverConfig.from_dict(result['configuration'])
    require(result['input']['manifest_sha256']==case['bundle_sha256'],'Comparison input differs')
    expected=request['source_implementation'] if label=='baseline' else request['implementation']
    require(result['implementation']==expected,'Comparison implementation differs')
    problem=load_problem(case['bundle'],config);bind_profile(problem,config,request['profile'],case['capture_profile'])
    require(problem.constraint_profile==case['constraint_binding'],'Comparison capture binding differs')
    install_result_bridges(problem,result,config);independent=verify_result_support(problem,result,config)
    ref={**reference,'input_manifest_sha256':case['bundle_sha256']}
    evaluation=evaluate(result,ref);quality=assembly_quality(result,ref,evaluation)
    assessment=read_json(target.with_name('stage4.assessment')/'assessment.json')['layouts'][0]
    poses={p:np.asarray(r['native_to_reference'],float) for p,r in reference['pieces'].items()}
    root=min(poses);anchor=np.linalg.inv(poses[root]);oracle=State({p:anchor@h for p,h in poses.items()},{p:root for p in poses})
    context=GeometryContext(problem,config,None);extent=extent_hypothesis(problem)
    uses_area=(config.native_repair_policy==REPAIR_V18_POLICY or REPAIR_V15_POLICIES.get(config.native_repair_policy,(False,False))[0])
    bound=context.component_bounds(oracle) if uses_area else None
    oracle_fits=bound['valid'] if uses_area else fits_extent(context,oracle,extent)
    return result,{'label':label,'status':'predicted','configuration':config.to_dict(),'quality':quality,'evaluation':evaluation,
        'assessment':assessment,'search':result['search'],'resources':result['resources'],
        'independent_native_support_verification':independent,'retained':retained_accuracy(result,reference),
        'native':native_diagnosis(result,reference,upstream),
        'reference_bound_diagnostic':{'policy':'observed-total-area.v1' if uses_area else 'spanning-piece-extent.v1',
            'reference_admitted':bool(oracle_fits),'component_area_check':bound,
            'provisional_extent':extent,'provisional_extent_enforced':not uses_area,
            'scope':'Post-seal diagnostic only; no reference poses or dimensions are supplied to inference.'},
        'visual_review':None,'four_pixel_metrics_are_diagnostic_only':True}


def write_native_gallery(folder,case,row):
    from .recovery_gallery_v18_r2 import unaccepted_gallery
    rejected=unaccepted_gallery(folder/'rejected',case,row)
    from .report import document,esc,table,number,link
    folder.mkdir(parents=True,exist_ok=True);audit=row['search']['native_repair'];coverage=row['native']
    atomic_json(folder/'repair.json',{'native_repair':audit,'coverage':coverage,
        'reference_bound_diagnostic':row['reference_bound_diagnostic']})
    gen=audit.get('native_generation') or {};body='<h1>'+esc(case['case_id'])+' native recovery</h1>'
    body+='<p>Measured native support and final page verification remain required. The area-bound arm assumes no page format, DPI or spanning piece. Pair fitting coverage and generation/merge exhaustion are reported separately. Reference diagnostics are post-seal only.</p>'
    body+=table(['Bound policy','Generation policy','Checks','Pair fits / possible','Generation limit','Merge limit','Protected completion'],[(
        esc(audit.get('component_bound_policy','spanning-piece-extent.v1')),esc(gen.get('policy','not run')),
        esc(coverage['native_checks']),esc(f"{coverage['pairs_with_fitted_checks']}/{coverage['possible_physical_pairs']}"),
        esc(coverage['generation_budget_abstained']),esc(coverage['merge_budget_abstained']),esc(audit['protected_supported_completion']))])
    body+=table(['Piece','Fitted incident pairs','Possible incident pairs','Native checks'],[
        (esc(p),esc(v['fitted_pairs']),esc(v['possible_pairs']),esc(v['native_checks'])) for p,v in coverage['per_piece_coverage'].items()])
    body+=table(['Source groups','Merge checks','Area rejections','Extent rejections','Complete shapes','Budget abstention'],[
        (esc(list(map(len,r.get('source_groups',[])))),esc(r['merge_checks']),esc(r.get('rejections',{}).get('component_area_bound',0)),
         esc(r.get('rejections',{}).get('provisional_extent',0)),esc(r['complete_shape_candidates']),esc(r['budget_abstained'])) for r in audit['merge_searches']])
    body+='<p>'+link(Path(rejected),folder,'Coarse complete poses / native growth / exact support failures')+'</p>'
    body+='<p>'+link(folder/'repair.json',folder,'Complete proposal, frontier, merge and reference-bound records')+'</p>'
    (folder/'index.html').write_text(document('Native recovery coverage',body),encoding='utf-8')
    return str(folder/'index.html')


def write_report(output,request,rows):
    from .report import document,esc,table,number,link,image
    report=output/'report';cells=[];visuals=[]
    for case,row in zip(request['cases'],rows):
        cid=case['case_id'];pics=[]
        for label,measurement in row['variants'].items():
            target=Path(measurement['reused_comparison_result']) if 'reused_comparison_result' in measurement else (Path(case['baseline']) if label=='baseline' else output/'cases'/cid/label/'stage4')
            if measurement['status']!='predicted':
                cells.append((esc(cid),esc(label),esc('failed'),esc('unavailable'),*(esc(None) for _ in range(7))))
                continue
            q=measurement['quality'];native=measurement['native'];retained=measurement['retained']
            gauge=q['common_gauge'];error=gauge['non_anchor_probe_errors_reference_px']['p95'] if gauge['whole_page_error_available'] else None
            cells.append((esc(cid),esc(label),esc(q['component_sizes']),esc(measurement['assessment']['status']),
                number(error),esc(len(measurement['evaluation']['false_neighbor_joins'])),
                esc(f"{native['pairs_with_fitted_checks']}/{native['possible_physical_pairs']}"),esc(native['generation_budget_abstained']),
                esc(retained['complete_candidates']),esc(measurement['reference_bound_diagnostic']['reference_admitted']),
                number(measurement.get('stage4_subprocess_seconds'))))
            pics.append(image(target/'preview.png',report,label))
        content='<h2>'+esc(cid)+'</h2><div class="visuals">'+''.join(pics)+'</div>'
        content+='<p>'+link(Path(case['reference_image']),report,'Reference page — post-seal review')+' | '+link(report/cid/'diagnosis.json',report,'All comparison measurements')+'</p>'
        for label,measurement in row['variants'].items():
            if label=='baseline' or 'reused_comparison_result' in measurement or measurement['status']!='predicted':continue
            galleries=measurement['galleries']
            content+='<p>'+esc(label)+': '+link(Path(galleries['geometry']['index']),report,'Geometry candidates')+' | '+link(Path(galleries['native']),report,'Native coverage / bounds / budgets')+' | '+link(output/'cases'/cid/label/'stage4.assessment/report.md',report,'Assessment / recovery')+' | '+link(output/'cases'/cid/label/'stage4.log',report,'Cold Stage 4 log')+'</p>'
        visuals.append(content);atomic_json(report/cid/'diagnosis.json',row)
    intro='<h1>V18 Stage 4 recovery comparison</h1><p>Eight-case source: two each at 4, 8, 12 and 16 pieces. This run uses the selected cases and ablation arms shown below. Stage 1/2/3 evidence and the V14 baseline are reused byte-for-byte. Every new result is a cold Stage 4 solve with no saved layout seeds. All selected predictions were sealed before reference diagnosis. Page format and DPI are unknown; captures declare shared native scale and quarter turns.</p><p>Correct overall construction with tolerable seam offsets is the goal. Four-pixel tests are diagnostic. A supported page needs visual review; an unresolved/rejected layout is not a solution. Pair pose graphs do not establish compatible supports. Saved candidates are not a certified global top-k. V18 tries native geometry before broad cold search, checks complete constructions immediately, reopens growth alternatives with backtracking, and measures page-edge cut witnesses when a weak seam lacks a cycle. Proposal and final-candidate counts do not terminate V18 geometry recovery. Its configurable overall solve deadline is 600 seconds; loading, publication and assessment are additional. Earlier bounded beam passes are ordering routes, not global exhaustion proofs. The geometry gate permits approximate construction; it does not prove semantic correctness or global search completeness. V15/V16 and available V17 focus columns reuse sealed results; only the adaptive column is new. Source poses, coarse complete states, native growth and final support failures are saved.</p>'
    intro+=table(['Case','Policy','Components','Page assessment','Whole-page p95 px','False neighbor joins','Native pair fits','Generation exhausted','Complete retained','Reference admitted by route bound','Stage 4 seconds'],cells)
    (report/'index.html').write_text(document('V18 recovery ablations',intro+''.join(visuals)),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    output=parser.parse_args(argv).run.resolve();request=checked(read_json(output/'request.json'),'request_sha256')
    barrier=checked(read_json(output/'predictions_sealed.json'),'barrier_sha256');progress=read_json(output/'progress.json')
    require(barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False,'Unsealed V18 predictions')
    sealed=verify_predictions(output,request,progress)
    require(all(sealed[k]==barrier[k] for k in sealed),'V18 prediction seal differs')
    require(implementation_identity()==request['implementation'],'Use the frozen V18 reader')
    receipt_path=output/'experiment_receipt.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path);require(receipt['request_sha256']==request['request_sha256'],'Foreign V18 report')
        verify_files(output,receipt['files']);print('V18 completed comparison verified',flush=True);return 0
    # Reference-bearing baseline diagnoses are first parsed here, after sealing.
    comparison=read_json(Path(request['comparison_v15']['root'])/'report/summary.json')
    comparison_rows={r['case_id']:r for r in comparison['cases']}
    previous=read_json(Path(request['comparison_v16']['root'])/'report/summary.json')
    previous_rows={r['case_id']:r for r in previous['cases']}
    latest=read_json(Path(request['comparison_v17']['root'])/'report/summary.json')
    latest_rows={r['case_id']:r for r in latest['cases']}
    source=Path(request['source']['root']);source_summary=read_json(source/'report/summary.json')
    upstream={r['case_id']:r for r in source_summary['cases']}
    report=output/'report';report.mkdir(exist_ok=True);rows=[]
    from .native_repair_experiment_v14 import write_gallery,write_closure_gallery
    for case in request['cases']:
        cid=case['case_id'];print('[diagnostics] '+cid,flush=True)
        require(file_hash(case['evaluation_reference'])==case['evaluation_reference_sha256'] and
            file_hash(case['reference_image'])==case['reference_image_sha256'],'Reference changed')
        reference=read_json(case['evaluation_reference']);variants={}
        labels=['baseline',*request['arms']]
        for label in labels:
            target=Path(case['baseline']) if label=='baseline' else output/'cases'/cid/label/'stage4'
            if label!='baseline' and progress['cases'][cid][label]['status']!='predicted':
                variants[label]={'status':'failed','attempts':progress['cases'][cid][label]['attempts']};continue
            result,row=measure(case,target,request,reference,upstream[cid],label)
            if label=='baseline':
                row['stage4_subprocess_seconds']=upstream[cid]['timing']['stage4']
            else:
                row['stage4_subprocess_seconds']=sum(a['elapsed_seconds'] for a in progress['cases'][cid][label]['attempts'])
                row['stage4_attempts']=progress['cases'][cid][label]['attempts']
                row['stage4_attempt_count']=len(row['stage4_attempts'])
                ctx={**case};gallery_request={'configuration':request['configurations'][label]}
                # Keep each arm's gallery separate while reusing the tested renderer.
                gallery_root=report/'arms'/label;gallery_root.mkdir(parents=True,exist_ok=True)
                row['galleries']={'geometry':write_gallery(gallery_root,ctx,result,gallery_request),
                    'closure':write_closure_gallery(gallery_root,ctx,result),
                    'native':write_native_gallery(gallery_root/'native_recovery'/cid,case,row)}
            variants[label]=row
        for old in ('coverage','combined'):
            oldrow=dict(comparison_rows[cid]['variants'][old])
            oldrow['reused_comparison_result']=str(Path(request['comparison_v15']['root'])/'cases'/cid/old/'stage4')
            variants['v15_'+old]=oldrow
        oldrow=dict(previous_rows[cid]['variants']['portfolio'])
        oldrow['reused_comparison_result']=str(Path(request['comparison_v16']['root'])/'cases'/cid/'portfolio/stage4')
        variants['v16_portfolio']=oldrow
        if cid in latest_rows:
            oldrow=dict(latest_rows[cid]['variants']['adaptive'])
            oldrow['reused_comparison_result']=str(Path(request['comparison_v17']['root'])/'cases'/cid/'adaptive/stage4')
            variants['v17_adaptive']=oldrow
        (report/cid).mkdir(exist_ok=True)
        rows.append({'case_id':cid,'pieces':case['pieces'],'variants':variants,
            'unchanged_upstream_stage2':upstream[cid]['stages']['stage2'],
            'unchanged_upstream_stage3':upstream[cid]['stages']['stage3'],
            'upstream_diagnosis':str(source/'report'/cid/'diagnosis.json')})
    write_report(output,request,rows)
    atomic_json(report/'summary.json',{'schema_version':'stage4.v18-recovery-comparison.v1',
        'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],'cases':rows,
        'stage2_stage3_inference_run':False,'reference_access':'post-seal diagnostic process only','default_promoted':False})
    atomic_json(receipt_path,{'schema_version':'stage4.v18-recovery-report-receipt.v1','request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'files':[{**f,'path':'report/'+f['path']} for f in inventory(report)]})
    print('V18 comparison report complete:',report/'index.html',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())


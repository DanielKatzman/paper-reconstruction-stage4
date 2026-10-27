"""Report-only repair for sealed V18 r2 predictions; serving source is unchanged."""
import argparse
from pathlib import Path
from .common import atomic_json, digest, file_hash, inventory, read_json, verify_files
from .recovery_experiment_v18_r2 import checked, require, verify_predictions
from .recovery_diagnosis_v18_r2 import measure, write_report, write_native_gallery
from paper_reconstruction_stage4.outputs import implementation_identity


def repair_identity(request):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'scripts/repair_recovery_v18_r2_report.py']
    return {'policy': 'v18-r2-unattempted-closure-report.v1',
        'files': {str(p): file_hash(p) for p in paths},
        'original_diagnosis_sha256': request['controllers'][str(root/'benchmark/recovery_diagnosis_v18_r2.py')],
        'solver_changed': False, 'predictions_changed': False,
        'missing_measurements_fabricated': False}


def write_closure_gallery(report, case, result):
    """Render absent, unattempted and measured routes without inventing timings."""
    from .report import document, table, esc, number, link
    audit = result['search'].get('component_closure')
    path = report/'component_placements'/case['case_id']; path.mkdir(parents=True, exist_ok=True)
    activity = audit or {}
    unattempted = (audit is not None and 'elapsed_seconds' not in activity and
        not activity.get('proposals') and not any(activity.get(k, 0) for k in
            ('generated_proposals', 'fine_checks', 'verified_closures')))
    status = 'not_recorded' if audit is None else 'not_run' if unattempted else 'activity_recorded'
    atomic_json(path/'placements.json', {'source_run_id': result['run_id'], 'audit': audit,
        'route_status': status, 'reference_used_for_order_or_verification': False,
        'missing_measurements_fabricated': False})
    content = '<h1>'+esc(case['case_id'])+' component placement</h1>'
    if status == 'not_run':
        content += '<p>The legacy component-closure route was not run. V18 native-first recovery records its work in the native-recovery audit. No timing or omitted-source count was recorded for this legacy route.</p>'
    elif status == 'not_recorded':
        content += '<p>No component-closure audit was recorded. Missing measurements remain unavailable.</p>'
    else:
        content += '<p>Recorded component-placement activity is shown below. Missing measurements remain unavailable; coarse hypotheses are not accepted seams.</p>'
    content += table(['Route status', 'Generated', 'Fine checked', 'Verified closures', 'Route seconds', 'Source families omitted', 'Budget abstention'], [(
        esc(status), esc(activity.get('generated_proposals')), esc(activity.get('fine_checks')),
        esc(activity.get('verified_closures')), number(activity.get('elapsed_seconds')),
        esc(activity.get('sources_omitted_by_limit')), esc(activity.get('budget_abstained')))])
    rows = []
    for rank, row in enumerate(activity.get('proposals', []), 1):
        bridges = row.get('bridges', [])
        rows.append((esc(rank), esc(row.get('disposition')), number(row.get('coarse_geometry', {}).get('score')),
            number(row.get('fine_geometry', {}).get('fill_fraction')), esc(len(bridges)),
            esc(', '.join('/'.join(b['pieces']) for b in bridges)), esc(row.get('reason', ''))))
    content += table(['Coarse rank', 'Disposition', 'Coarse geometry', 'Fine fill', 'Native seam pairs', 'Pairs', 'Reason'], rows)
    content += '<p>'+link(path/'placements.json', path, 'Original sealed component-placement audit')+'</p>'
    (path/'index.html').write_text(document('Component placement verification', content), encoding='utf-8')
    return {'index': str(path/'index.html'), 'route_status': status,
        'generated': activity.get('generated_proposals'), 'fine_checked': activity.get('fine_checks'),
        'verified_closures': activity.get('verified_closures')}


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
    from .native_repair_experiment_v14 import write_gallery
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
    repair = repair_identity(request)
    atomic_json(report/'summary.json',{'schema_version':'stage4.v18-recovery-comparison.v1',
        'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],'cases':rows,'report_repair':repair,
        'stage2_stage3_inference_run':False,'reference_access':'post-seal diagnostic process only','default_promoted':False})
    atomic_json(receipt_path,{'schema_version':'stage4.v18-recovery-report-receipt.v1','request_sha256':request['request_sha256'],
        'barrier_sha256':barrier['barrier_sha256'],'report_repair':repair,'files':[{**f,'path':'report/'+f['path']} for f in inventory(report)]})
    print('V18 comparison report complete:',report/'index.html',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())


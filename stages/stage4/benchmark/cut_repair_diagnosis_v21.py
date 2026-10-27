"""Compare every fresh V21 result with the sealed V20/confirmed baseline."""
import argparse
from pathlib import Path
import traceback
from .common import read_json,atomic_json,file_hash,inventory,verify_files
from .cut_repair_experiment_v21 import checked,require,authenticate,verify_predictions


def p95(row):
    g=row.get('quality',{}).get('common_gauge',{})
    return g.get('non_anchor_probe_errors_reference_px',{}).get('p95') if g.get('whole_page_error_available') else None


def write_report(output,request,rows):
    from .report import document,esc,table,number,link,image
    root=output/'report';root.mkdir(parents=True,exist_ok=True);cells=[];pictures=[]
    for row in rows:
        case=row['case_id'];base=row['baseline'];new=row['candidate'];audit=new.get('search',{}).get('native_repair',{})
        cut=audit.get('unsupported_cut_repair',{});searches=audit.get('merge_searches',[])
        cells.append((esc(case),esc(row['pieces']),esc(base.get('quality',{}).get('component_sizes')),
            esc(new.get('quality',{}).get('component_sizes')),esc(new.get('assessment',{}).get('status',new.get('status'))),
            esc(len(new['evaluation']['false_neighbor_joins'])) if 'evaluation' in new else esc(None),
            number(p95(base)),number(p95(new)),number(base.get('stage4_worker_seconds')),number(new.get('stage4_worker_seconds')),
            esc(cut.get('verified_completions',0)),esc(sum(t.get('backtracking_seed_sessions_started',0) for t in searches))))
        folder=root/case;folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'diagnosis.json',row)
        text='<h2>'+esc(case)+'</h2><div class="grid">'
        for label,path in [('Baseline',Path(row['baseline_result'])),('V21 cold solve',output/'cases'/case/'stage4')]:
            text+='<section><h3>'+esc(label)+'</h3>'
            text+=image(path/'preview.png',root,label) if (path/'preview.png').exists() else '<p>No result was published.</p>'
            text+='</section>'
        text+='</div><p>'+link(folder/'diagnosis.json',root,'Complete diagnosis, reused upstream evidence, cut repair and search telemetry')+'</p>'
        for name,path in new.get('galleries',{}).items():
            target=Path(path['index']) if isinstance(path,dict) else Path(path)
            text+='<p>'+link(target,root,name+' candidates and verification')+'</p>'
        proof=new.get('bank_constructibility',{})
        if proof.get('records'):
            text+='<p>'+link(root/'constructibility'/case/'diagnostic.json',root,'Reference-assisted bank diagnostic, excluded from blind success')+'</p>'
        pictures.append(text)
    intro='<h1>V21: unsupported-cut repair and resumable native search</h1><p>Every selected case below runs a fresh cold Stage 4 solve. Stage 2/3 evidence is reused byte-for-byte from the sealed capacity inputs. Baseline pictures/results are comparison evidence and are never supplied to the solver. References are opened only after all selected attempts are sealed. Acceptance thresholds and the 600-second cooperative deadline are unchanged. Worker times include loading, publication and assessment; diagnostic times are separate.</p><p>Inspect page construction as well as connectivity. Four-pixel metrics are diagnostic; the goal is correct placement with tolerable offsets. An unsuccessful saved-bank diagnostic is inconclusive. This remains synthetic development data with exact masks, common scale and declared quarter turns.</p>'
    html=table(['Puzzle','Pieces','Baseline groups','V21 groups','V21 assessment','V21 false pairs',
        'Baseline page p95 px','V21 page p95 px','Baseline worker s','V21 worker s','Verified cut repairs','DFS sessions started'],cells)
    (root/'index.html').write_text(document('V21 capacity comparison',intro+html+''.join(pictures)),encoding='utf8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    output=parser.parse_args(argv).run.resolve();request=checked(read_json(output/'request.json'),'request_sha256')
    barrier=checked(read_json(output/'predictions_sealed.json'),'barrier_sha256');progress=read_json(output/'progress.json')
    require(barrier['request_sha256']==request['request_sha256'] and barrier['references_opened'] is False,'Unsealed predictions')
    authenticate(request)
    current=verify_predictions(output,request,progress)
    require(all(barrier[k]==v for k,v in current.items()),'Prediction seal differs')
    from paper_reconstruction_stage4.outputs import implementation_identity
    require(implementation_identity()==request['implementation'],'Use the frozen reader')
    receipt=output/'experiment_receipt.json'
    if receipt.exists():
        verify_files(output,read_json(receipt)['files']);print('V21 report verified:',output/'report/index.html',flush=True);return 0
    # Reference-aware upstream diagnoses and baseline measurements are parsed
    # only beyond the new prediction barrier, never during request creation.
    baseline={c['case_id']:c for c in read_json(Path(request['source'])/'report/summary.json')['cases']}
    from .recovery_diagnosis_v19_r2 import measure,write_native_gallery
    from .native_repair_experiment_v14 import write_gallery
    from .recovery_report_v18_r2_fix import write_closure_gallery
    from .constructibility_v20 import probe
    from .scale_extension_diagnosis_v20 import requires_probe
    rows=[]
    for i,case in enumerate(request['cases'],1):
        cid=case['case_id'];old=baseline[cid];entry=progress['cases'][cid]
        print(f'[diagnostics {i}/{len(request["cases"])}] {cid}: independent replay and comparison',flush=True)
        old_seconds=old['timing']['stage4']
        base={k:old[k] for k in ('quality','evaluation','assessment')}
        base['stage4_worker_seconds']=old_seconds
        row={'case_id':cid,'pieces':case['pieces'],'baseline_result':case['baseline'],'baseline':base,
            'upstream_stages_reused':old['stages'],'upstream_inference_rerun':False,
            'baseline_diagnosis_reused':True,'references_opened_after_seal':True}
        if entry['status']!='predicted':
            row['candidate']={'status':'process_failed','attempts':entry['attempts']};rows.append(row);continue
        require(file_hash(case['evaluation_reference'])==case['evaluation_reference_sha256'],'Reference changed: '+cid)
        reference=read_json(case['evaluation_reference'])
        result,new=measure(case,output/'cases'/cid/'stage4',request,reference,{'stages':old['stages']},'candidate')
        new['stage4_worker_seconds']=sum(a['elapsed_seconds'] for a in entry['attempts'])
        new['stage4_attempts']=entry['attempts']
        gallery=output/'report/candidates'
        new['galleries']={'retained_geometry':write_gallery(gallery,case,result,{'configuration':request['configuration']}),
            'component_closure':write_closure_gallery(gallery,case,result),
            'native_recovery':write_native_gallery(gallery/'native_recovery'/cid,case,new)}
        if requires_probe(new['quality'],new['evaluation'],new['assessment']):
            print('[constructibility] '+cid+': diagnostic only',flush=True)
            try:
                new['bank_constructibility']=probe(case,result,reference,request,output/'report/constructibility'/cid,barrier['barrier_sha256'])
            except Exception as exc:
                new['bank_constructibility']={'status':'diagnostic_failed','error':str(exc),'traceback':traceback.format_exc(),
                    'counts_as_blind_success':False,'negative_result_proves_impossibility':False}
        else:new['bank_constructibility']={'status':'not_needed','counts_as_blind_success':False}
        row['candidate']=new;rows.append(row)
        atomic_json(output/'report'/cid/'diagnosis.json',row)
    write_report(output,request,rows)
    atomic_json(output/'report/summary.json',{'schema_version':'stage4.cut-repair-comparison.v21',
        'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],
        'cases':rows,'cold_stage4_cases':len(rows),'reused_upstream_cases':len(rows),
        'old_layouts_supplied_to_solver':False,'references_opened_after_seal':True,
        'diagnostic_witnesses_count_as_blind_success':False})
    atomic_json(receipt,{'request_sha256':request['request_sha256'],'barrier_sha256':barrier['barrier_sha256'],
        'files':[{**r,'path':'report/'+r['path']} for r in inventory(output/'report')]})
    print('V21 diagnosis finished:',output/'report/index.html',flush=True);return 0

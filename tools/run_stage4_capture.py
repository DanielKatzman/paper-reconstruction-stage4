"""Cold CPU Stage 4 benchmark from immutable saved Stage 2/3 evidence.

No previous assembly or evaluation reference is accepted. Output must be fresh.
"""
import argparse
import html
import os
from pathlib import Path
import shutil
import sys
from common import ROOT,read,save,sha,inventory,environment


def resolve(path):
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True,help='Folder containing stage3/bundle.json and stage1/capture_profile.json')
    parser.add_argument('--output',type=Path,required=True,help='New run directory; relative paths use the release root')
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--validate-only',action='store_true',help='Freeze source and validate the saved input without solving')
    args=parser.parse_args()
    if args.cpu_threads<1:parser.error('--cpu-threads must be positive')
    upstream=resolve(args.upstream);output=resolve(args.output)
    bundle=upstream/'stage3';capture=upstream/'stage1/capture_profile.json'
    if not (bundle/'bundle.json').is_file() or not capture.is_file():
        parser.error('Saved stage3/bundle.json and stage1/capture_profile.json are required')
    if not output.is_relative_to(ROOT/'outputs'):parser.error('Output must stay under this release outputs/ directory')
    if output.exists():parser.error('Use a fresh output directory; cold runs never resume old assemblies')
    if output.is_relative_to(upstream) or upstream.is_relative_to(output):parser.error('Output overlaps immutable upstream input')
    config=read(ROOT/'stages/stage4/configs/joint_v21_cut_repair_development.json')
    config['joint_wall_seconds']=args.seconds
    # Validate before creating any output or starting expensive work.
    sys.path[:0]=[str(ROOT/'contracts/sdk/src'),str(ROOT/'stages/stage4/src')]
    from paper_reconstruction_stage4.config import SolverConfig
    SolverConfig.from_dict(config)
    before=inventory(bundle);capture_hash=sha(capture)
    runtime=output/'runtime'
    for source,target in [(ROOT/'stages/stage4/src/paper_reconstruction_stage4',runtime/'paper_reconstruction_stage4'),
                          (ROOT/'contracts/sdk/src/paper_reconstruction_contracts',runtime/'paper_reconstruction_contracts')]:
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(output/'source_inventory.json',inventory(runtime));save(output/'solver_config.json',config)
    profile=output/'constraint_profile.json'
    shutil.copy2(ROOT/'stages/stage4/configs/shared_scale_complete_rectangle_predicted_v1.json',profile)
    request={'mode':'cold-cpu-stage4-capture','upstream_bundle':str(bundle),'upstream_inventory':before,
        'capture_profile_sha256':capture_hash,'constraint_profile_sha256':sha(profile),
        'stages_1_2_3_regenerated':False,'previous_assembly_used':False,'reference_geometry_used':False,
        'cpu_threads':args.cpu_threads,'cooperative_wall_seconds':args.seconds,
        'source_inventory_sha256':sha(output/'source_inventory.json')}
    save(output/'request.json',request)
    env=environment('stage4');env['PYTHONPATH']=os.pathsep.join([str(runtime),str(ROOT/'stages/stage4')])
    env['CUDA_VISIBLE_DEVICES']='-1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]=str(args.cpu_threads)
    import common
    original=common.environment
    common.environment=lambda stage:env if stage=='stage4' else original(stage)
    result=output/'stage4'
    print('Cold CPU Stage 4: saved upstream evidence only; no Stage 3 regeneration.',flush=True)
    if args.validate_only:
        common.execute('stage4',['-m','paper_reconstruction_stage4','validate','--bundle',bundle,
            '--config',output/'solver_config.json','--constraint-profile',profile,'--capture-profile',capture],output,'validate')
        if inventory(bundle)!=before or sha(capture)!=capture_hash:raise ValueError('Immutable upstream input changed')
        save(output/'preflight.json',{'status':'validated_without_solving','upstream_unchanged':True,
            'reference_geometry_used':False,'previous_assembly_used':False})
        print('Saved upstream validated. Use a different fresh output directory for the benchmark.',flush=True)
        return 0
    receipt=common.execute('stage4',['-m','paper_reconstruction_stage4','solve','--bundle',bundle,
        '--output',result,'--config',output/'solver_config.json','--constraint-profile',profile,
        '--capture-profile',capture,'--progress'],output,'stage4')
    barrier=inventory(result)
    save(output/'predictions_sealed.json',{'predictions':barrier,'references_opened':False,'previous_assembly_used':False})
    # Validation reads only the newly sealed result and the immutable bundle.
    common.execute('stage4',['-m','paper_reconstruction_stage4','assess','--bundle',bundle,'--result',result,
        '--output',output/'assessment','--config',output/'solver_config.json','--constraint-profile',profile,
        '--capture-profile',capture],output,'assessment')
    if inventory(result)!=barrier or inventory(bundle)!=before or sha(capture)!=capture_hash:
        raise ValueError('Sealed prediction or immutable upstream input changed')
    x=read(result/'result.json');assessment=read(output/'assessment/assessment.json')
    quality=x['search'].get('native_repair',{}).get('supported_capture_quality_repair',[])
    summary={'status':x['status'],'component_sizes':[len(g) for g in x['layouts'][0]['components']],
        'pieces':x['input']['counts']['pieces'],'selected_joins':len(x['layouts'][0]['selected_joins']),
        'elapsed_seconds':receipt['elapsed_seconds'],'cpu_only':True,'upstream_unchanged':True,
        'predictions_unchanged':True,'previous_assembly_used':False,'reference_geometry_used':False,
        'accuracy_not_measured_by_reference_free_assessment':True,'assessment':assessment,
        'supported_capture_quality_repair':quality,'implementation':x.get('implementation',x.get('solver'))}
    save(output/'review.json',summary)
    report=output/'report';report.mkdir()
    report.joinpath('index.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Cold real-capture Stage 4 benchmark</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f7fa;color:#18384a}img{max-width:100%;max-height:1000px}pre{white-space:pre-wrap}a{color:#126088}</style>
<h1>Cold CPU Stage 4 capture benchmark</h1><p>Reuses immutable Stage 2/3 evidence. No previous assembly, reference image or reference poses enter the solve. Page shape and native support are checks, not an accuracy probability.</p>
<p><a href="../review.json">Diagnostics</a> | <a href="../stage4.log">Solver progress log</a> | <a href="../assessment/report.md">Assessment</a> | <a href="../predictions_sealed.json">Prediction seal</a></p>
<a href="../stage4/preview.png"><img src="../stage4/preview.png"></a><pre>'''+html.escape(str({k:v for k,v in summary.items() if k not in ['assessment','supported_capture_quality_repair']}))+'</pre></html>',encoding='utf-8')
    print('Capture report: '+str(report/'index.html'),flush=True)
    print('Component sizes: '+str(summary['component_sizes'])+'; status: '+summary['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

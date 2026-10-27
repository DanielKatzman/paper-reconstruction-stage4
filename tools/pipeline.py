"""Portable four-stage pipeline and cold, sealed Stage 4 regression runner."""
import argparse
from pathlib import Path
import shutil
import sys
from common import ROOT, execute, read, save, inventory, sha

CONFIG=ROOT/'stages/stage4/configs/joint_v21_cut_repair_development.json'
PROFILE=ROOT/'stages/stage4/configs/shared_scale_complete_rectangle_v1.json'


def solve(bundle,capture,output,config=CONFIG,profile=PROFILE):
    args=['-m','paper_reconstruction_stage4','solve','--bundle',bundle,'--output',output,'--config',config]
    if profile: args.extend(['--constraint-profile',profile,'--capture-profile',capture])
    return execute('stage4',args,output.parent,'stage4')


def fresh(path,protected):
    path=Path(path).resolve()
    if path.exists(): raise ValueError('Use a new output directory: '+str(path))
    for item in protected:
        item=Path(item).resolve()
        if path.is_relative_to(item) or item.is_relative_to(path): raise ValueError('Output overlaps input: '+str(item))
    path.mkdir(parents=True)
    return path


def benchmark(args):
    cohort=read(ROOT/'baseline/cohort.json')['cases']
    if args.cases: cohort=[c for c in cohort if c['case_id'] in args.cases]
    if not cohort or (args.cases and set(args.cases)!={c['case_id'] for c in cohort}): raise ValueError('Unknown/empty case selection')
    output=fresh(args.output,[ROOT/'assets',ROOT/'stages',ROOT/'contracts',ROOT/'baseline',ROOT/'provenance',ROOT/'tools'])
    save(output/'request.json',{'cases':cohort,'config_sha256':sha(CONFIG),'profile_sha256':sha(PROFILE),
        'serving_references_opened':False,'stage2_stage3_reused':True,'old_assembly_seeds':False})
    for i,c in enumerate(cohort,1):
        print(f'[{i}/{len(cohort)}] {c["case_id"]}: {c["pieces"]} pieces',flush=True)
        solve(ROOT/c['bundle'],ROOT/c['capture_profile'],output/'cases'/c['case_id']/'stage4')
    barrier={c['case_id']:inventory(output/'cases'/c['case_id']/'stage4') for c in cohort}
    save(output/'predictions_sealed.json',{'predictions':barrier,'references_opened':False})
    evaluate_run(output)


def evaluate_run(output):
    output=Path(output).resolve()
    cohort=read(output/'request.json')['cases']
    barrier=read(output/'predictions_sealed.json')['predictions']
    before={c['case_id']:inventory(output/'cases'/c['case_id']/'stage4') for c in cohort}
    if before!=barrier: raise ValueError('Prediction barrier differs before evaluation')
    # Evaluation runs in separate processes only after every primary prediction is sealed.
    for c in cohort:
        execute('stage4',[ROOT/'tools/evaluate_case.py','--result',output/'cases'/c['case_id']/'stage4',
            '--reference',ROOT/c['evaluation_reference'],'--output',output/'cases'/c['case_id']/'evaluation.json'],
            output/'cases'/c['case_id'],'evaluation')
    after={c['case_id']:inventory(output/'cases'/c['case_id']/'stage4') for c in cohort}
    if after!=barrier: raise ValueError('Predictions changed during evaluation')
    rows=[{'case_id':c['case_id'],'pieces':c['pieces'],**read(output/'cases'/c['case_id']/'evaluation.json')} for c in cohort]
    save(output/'review.json',{'cases':rows,'predictions_unchanged':True,
        'all_supported_and_correct_neighbors':all(r['supported_connected_zero_false_neighbors'] for r in rows),
        'all_construction_regressions_pass':all(r['construction_regression_pass'] for r in rows)})
    if not all(r['construction_regression_pass'] for r in rows): raise RuntimeError('Benchmark has unresolved/incorrect assemblies; see review.json')
    print('Benchmark passed: '+str(output/'review.json'),flush=True)


def run(args):
    source=(args.stage1 or args.captures).resolve()
    if not source.is_dir():raise ValueError('Input folder does not exist: '+str(source))
    if args.captures and (args.mode!='shared-scale-capture' or not args.capture_profile):
        raise ValueError('Raw captures need --mode shared-scale-capture and a truthful --capture-profile')
    output=fresh(args.output,[source,ROOT/'stages',ROOT/'contracts',ROOT/'tools',ROOT/'baseline',ROOT/'provenance'])
    save(output/'request.json',{'mode':args.mode,'source':str(source),'source_files':inventory(source),
        'device':args.device,'cpu_threads':args.cpu_threads,'references_opened':False})
    if args.stage1:
        stage1=source
    else:
        stage1=output/'stage1'
        execute('stage1',[ROOT/'stages/stage1/scripts/segment_folder.py',source,stage1,'--review',args.review],output,'stage1')
        if not args.capture_profile: raise ValueError('Shared-scale Stage 3 requires an explicit truthful capture profile')
        shutil.copy2(args.capture_profile,stage1/'capture_profile.json')
    capture=stage1/'capture_profile.json'
    if not capture.is_file(): raise ValueError('Stage 3 requires capture_profile.json in the Stage 1 input')
    if args.mode=='development-page':
        c=read(capture)
        if c.get('stage1_masks')!='generator_exact_bypass': raise ValueError('V21 page profile is qualified only for exact generator masks')
    execute('stage2',[ROOT/'stages/stage2/scripts/run_stage2_inference.py','--input',stage1,'--output',output/'stage2',
        '--release',ROOT/'stages/stage2/models/release.json','--cpu-threads',args.cpu_threads],output,'stage2')
    execute('stage2',[ROOT/'stages/stage2/scripts/verify_stage2_output.py','--input',output/'stage2',
        '--release',ROOT/'stages/stage2/models/release.json'],output,'stage2_verify')
    execute('stage3',[ROOT/'tools/stage3_worker.py','infer','--stage1',stage1,'--stage2',output/'stage2','--output',output/'stage3_native',
        '--device',args.device,'--puzzle-id',args.puzzle_id],output,'stage3_native')
    execute('stage3',[ROOT/'tools/stage3_worker.py','pool','--input',output/'stage3_native','--output',output/'stage3'],output,'stage3_pool')
    config=CONFIG if args.mode=='development-page' else ROOT/'stages/stage4/configs/joint_continuation_development.json'
    solve(output/'stage3',capture,output/'stage4',config,PROFILE if args.mode=='development-page' else None)
    save(output/'predictions_sealed.json',{'predictions':inventory(output/'stage4'),'references_opened':False})
    print('Pipeline complete: '+str(output/'stage4/preview.png'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('benchmark'); b.add_argument('--output',type=Path,required=True); b.add_argument('--cases',nargs='+')
    e=sub.add_parser('evaluate'); e.add_argument('--run',type=Path,required=True)
    r=sub.add_parser('run'); src=r.add_mutually_exclusive_group(required=True); src.add_argument('--stage1',type=Path); src.add_argument('--captures',type=Path)
    r.add_argument('--output',type=Path,required=True); r.add_argument('--capture-profile',type=Path)
    r.add_argument('--mode',choices=['development-page','shared-scale-capture'],default='development-page')
    r.add_argument('--device',choices=['cuda','cpu'],default='cuda'); r.add_argument('--cpu-threads',type=int,default=4)
    r.add_argument('--review',choices=['none','flagged','all'],default='flagged'); r.add_argument('--puzzle-id',default='page')
    a=p.parse_args()
    if a.command=='benchmark': benchmark(a)
    elif a.command=='evaluate': evaluate_run(a.run)
    else: run(a)


if __name__=='__main__': main()

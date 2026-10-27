"""Run serial Stage-4-only ablations over one frozen external evidence spec."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.common import WORKSPACE, atomic_json, file_hash, read_json
from benchmark.joint_experiment import make_plan
from benchmark.runner import execute
from paper_reconstruction_stage4.outputs import implementation_identity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=WORKSPACE/'configs/joint8_external_experiment.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--variants',nargs='+',choices=['baseline','search','refine','active','combined','robust'],default=['search','refine','active','combined','robust'])
    args=parser.parse_args(argv)
    if len(args.variants)!=len(set(args.variants)):
        parser.error('Variants must be unique')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=(args.output or WORKSPACE/('outputs/joint_matrix_'+stamp)).resolve()
    # Validate every route before creating the parent output or launching work.
    plans={name:make_plan(args.spec,output/name,name) for name in args.variants}
    if output.exists():raise ValueError('Matrix output exists; choose a fresh directory')
    output.mkdir(parents=True)
    request={'schema_version':'stage4.joint-matrix.v1','spec':str(args.spec.resolve()),'spec_sha256':file_hash(args.spec),
             'consumer_implementation':implementation_identity(),'variants':args.variants,
             'input_manifest_sha256':next(iter(plans.values()))['input_manifest_sha256'],
             'upstream_inference_repeated':False,'fresh_full_pipeline_timing_measured':False}
    atomic_json(output/'request.json',request)
    progress={'phase':'serving_variants','variants':{}}
    try:
        for i,name in enumerate(args.variants,1):
            if implementation_identity()!=request['consumer_implementation'] or file_hash(args.spec)!=request['spec_sha256']:
                raise ValueError('Matrix source/spec changed; preserve outputs and start a new matrix')
            print(f'[{i}/{len(args.variants)}] Stage 4 {name}',flush=True)
            progress['active_variant']=name;atomic_json(output/'progress.json',progress)
            attempt=execute([sys.executable,'-B',WORKSPACE/'scripts/run_stage4_experiment.py','--spec',args.spec,
                             '--variant',name,'--output',output/name],output/f'logs/{name}.log',echo_progress=True)
            progress['variants'][name]=attempt
            atomic_json(output/'progress.json',progress)
        progress.update(phase='finished',active_variant=None);atomic_json(output/'progress.json',progress)
    except KeyboardInterrupt:
        progress['phase']='interrupted';atomic_json(output/'progress.json',progress);return 130
    except Exception as exc:
        progress.update(phase='failed',error=str(exc));atomic_json(output/'progress.json',progress);raise
    rows=[]
    for name in args.variants:
        p=output/name/'comparison.json'
        if p.is_file():
            c=read_json(p);q=c['experiment']['quality']
            rows.append({'variant':name,'placement_p95_reference_px':q['common_gauge']['non_anchor_probe_errors_reference_px']['p95'] if q['common_gauge']['whole_page_error_available'] else None,
                         'contact_median_native_a_px':q['contact_p95_error_distribution_native_a_px']['p50'],
                         'components':q['component_sizes'],'selected_true_pairs':q['selected_true_neighbor_pairs'],
                         'selected_pairs':q['selected_physical_pairs'],'score':c['experiment']['objective_score'],
                         'consumer_seconds':c['execution']['serving']['elapsed_seconds']})
    atomic_json(output/'matrix_comparison.json',{'request':request,'results':rows,'progress':progress,'manual_visual_assessment':None})
    lines=['# Stage 4 joint ablations','','All variants reuse one official Stage 3 bundle. These are consumer timings, not full pipeline timings.','',
           '| Variant | Placement p95, ref px | Median contact, native-A px | True selected pairs | Components | Consumer seconds |',
           '| --- | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['placement_p95_reference_px']} | {r['contact_median_native_a_px']} | {r['selected_true_pairs']}/{r['selected_pairs']} | {r['components']} | {r['consumer_seconds']:.2f} |")
    lines.extend(['','Inspect each variant\'s sealed primary preview and comparison report. No automatic solved-page label or oracle selection is applied.',''])
    (output/'MATRIX_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Matrix comparison: '+str(output/'MATRIX_SUMMARY.md'),flush=True)
    return 0 if all(a['returncode']==0 for a in progress['variants'].values()) else 1


if __name__=='__main__':raise SystemExit(main())

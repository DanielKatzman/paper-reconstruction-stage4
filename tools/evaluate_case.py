"""Post-seal reference measurement; never imported by serving orchestration."""
import argparse
from common import read,save,sha
from paper_reconstruction_stage4.outputs import read_result
from evaluation.evaluate import evaluate
from benchmark.quality import assembly_quality
from construction import approximate_construction


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--result',required=True); p.add_argument('--reference',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); result=read_result(a.result); reference=read(a.reference)
    reference['input_manifest_sha256']=result['input']['manifest_sha256']
    report=evaluate(result,reference)
    from pathlib import Path
    assessment=read(Path(a.result).with_name(Path(a.result).name+'.assessment')/'assessment.json')
    connected=len(result['layouts'][0]['components'])==1
    false_pairs=len(report['false_neighbor_joins'])
    quality=assembly_quality(result,reference,report)
    report.update(reference_sha256=sha(a.reference),quality=quality,
        supported_connected_zero_false_neighbors=assessment['layouts'][0]['status']=='supported' and connected and false_pairs==0)
    report['construction_regression_pass']=report['supported_connected_zero_false_neighbors'] and approximate_construction(quality)
    report['construction_regression_policy']={'offline_only':True,'maximum_saved_probe_error_px':48.,
        'maximum_relative_orientation_error_degrees':1e-3,'maximum_relative_scale_error':1e-5,
        'serving_acceptance_thresholds_changed':False,'manual_preview_review_still_required':True}
    save(a.output,report)

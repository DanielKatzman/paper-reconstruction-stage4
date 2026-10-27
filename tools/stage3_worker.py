"""Run the pinned Stage 3 implementation and its unchanged observable pool policy."""
import argparse
import time
from pathlib import Path
from common import ROOT, save, sha


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['infer','pool'])
    p.add_argument('--stage1',type=Path); p.add_argument('--stage2',type=Path)
    p.add_argument('--input',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cpu'); p.add_argument('--puzzle-id',default='page')
    a=p.parse_args()
    if a.mode=='pool':
        from stage3_pool_replay_tools import publish_replay, POLICY
        audit=publish_replay(a.input,a.output,{'release':(ROOT/'VERSION').read_text().strip(),'pool_tool_sha256':sha(ROOT/'stages/stage3/scripts/stage3_pool_replay_tools.py')},policy=POLICY)
        save(a.output.parent/'stage3_pool_audit.json',audit); return
    from paper_reconstruction_stage3.alpha_inference import infer
    from stage3_profile_tools import Measurements,instrument,process_memory
    metrics=Measurements(); metrics.started=time.perf_counter()
    targets={'alpha_inference':[('production_inputs','inputs.production_inputs'),('retrieve_parents','ranking.retrieve_parents'),
        ('fuse_parents','ranking.fuse_parents'),('FrozenRankers.__init__','ranking.model_setup'),('FrozenRankers.score','ranking.score')],
        'pose_geometry':[('fit_parent','poses.fit_parent')], 'pose_inference_geometry':[('generate','poses.generate')],
        'alpha_bundle':[('export_bundle','export.bundle')], 'contracts':[('atomic_json','io.atomic_json')]}
    def on_write(path,value):
        if path.name.startswith('pair_') and 'pieces' in value:
            metrics.pairs.append({'pieces':value['pieces'],'elapsed_since_start_seconds':time.perf_counter()-metrics.started})
    with instrument(metrics,on_write,targets=targets):
        result=infer(ROOT/'stages/stage3/models',a.stage1,a.stage2,a.output,device=a.device,puzzle_id=a.puzzle_id,progress=lambda s:print(s,flush=True))
    save(a.output.parent/'stage3_profile.json',{'result':result,'elapsed_seconds':time.perf_counter()-metrics.started,
        'functions':metrics.functions(),'pairs':metrics.pairs,'peak_memory':process_memory(),
        'timer_scope':'Inclusive nested timings overlap; do not sum them','fresh_ranking_requested':True})


if __name__=='__main__': main()

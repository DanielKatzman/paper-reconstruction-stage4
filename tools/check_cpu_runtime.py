"""Verify the frozen Stage 3 rankers using a CPU-only Torch build and CPU tensors."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'contracts/sdk/src'),str(ROOT/'stages/stage3/src')]


def main():
    import torch
    if torch.version.cuda is not None:
        raise ValueError('Portable CPU profile requires the CPU-only Torch wheel; install torch==2.7.1 from https://download.pytorch.org/whl/cpu')
    from paper_reconstruction_stage3.alpha_package import load_package
    from paper_reconstruction_stage3.alpha_inference import FrozenRankers
    _,manifest=load_package(ROOT/'stages/stage3/models')
    rankers=FrozenRankers(ROOT/'stages/stage3/models',manifest,'cpu')
    for name,model in rankers.models:
        if any(x.device.type!='cpu' for x in [*model.parameters(),*model.buffers()]):
            raise ValueError('Non-CPU tensor in frozen model: '+name)
    print(json.dumps({'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,
        'device':'cpu','rankers_loaded':len(rankers.models),'gpu_required':False}))


if __name__=='__main__':main()

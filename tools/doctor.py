"""Authenticate inlined sources and check imports using each selected interpreter."""
from common import ROOT, read, sha, execute


def verify_sources():
    expected=read(ROOT/'provenance/components.json')
    count=0
    for stage,data in expected.items():
        source=ROOT/'stages'/stage/'src'
        actual={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(x.endswith('.egg-info') for x in p.parts)}
        if actual!=data['files']: raise ValueError('Component source changed: '+stage)
        count+=len(actual)
    candidate=read(ROOT/'stages/stage3/provenance/candidate.json')
    from pathlib import Path
    for name,digest in candidate['files'].items():
        if sha(ROOT/'stages/stage3/src/paper_reconstruction_stage3'/name)!=digest: raise ValueError('Pinned Stage 3 candidate differs')
    print(f'Exact source inventories verified: {count} files',flush=True)


def main():
    verify_sources()
    from verify_compatibility import verify
    verify()
    expressions={
        'stage1':"import universal_segmenter,onnxruntime; from universal_segmenter.inference.release import locate_release; print(universal_segmenter.__file__); print(locate_release())",
        'stage2':"import paper_reconstruction_stage2,onnxruntime; print(paper_reconstruction_stage2.__file__)",
        'stage3':"import paper_reconstruction_stage3,torch; from paper_reconstruction_stage3.alpha_package import load_package; from pathlib import Path; print(paper_reconstruction_stage3.__file__); print(load_package(Path('stages/stage3/models'))[1]['manifest_sha256']); print('torch',torch.__version__,'cuda',torch.cuda.is_available())",
        'stage4':"from paper_reconstruction_stage4.outputs import implementation_identity; import paper_reconstruction_stage4; import json; from pathlib import Path; print(paper_reconstruction_stage4.__file__); x=implementation_identity(); print(x['sha256']); expected=json.loads(Path('compatibility.json').read_text())['stages']['stage4']['source_sdk_fingerprint']; assert x['sha256']==expected"}
    for stage,code in expressions.items(): execute(stage,['-c',code],ROOT/'outputs/doctor',stage)
    print('All four inlined runtimes passed preflight.',flush=True)


if __name__=='__main__': main()

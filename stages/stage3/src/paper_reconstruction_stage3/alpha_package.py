"""Load the small inference distribution; no research outputs or datasets."""
from pathlib import Path
from .alpha_bundle import ALGORITHM, file_hash
from .contracts import digest, read_json, require
from paper_reconstruction_contracts import verify_receipt
FROZEN = [{'name': 'pair_mlp_small', 'context': 'cbd559b8e21e9c68bb1c983329f9b0f0d1e7ac342ad44c9456cba7d7cc879bcc', 'weights_sha256': 'cd37a50ad33b5b13de7d984aa67269ca3c17a589fab18dec87823bd9b854eca4', 'upstream_checkpoint_sha256': '88a9581d4d80f0170e86b63fb61877728715cd9e8b69ed324ea6584e83150380', 'artifact_sha256': '5d47fc1f7249fb72b64252e0a448e08e5cc3f09d751621106cc34562eccdb930'}, {'name': 'paired_strip_cnn_small', 'context': '7679d89c40dc939381131edef3edb5e50fe90fd76b1341688567fa8f87cb1fc5', 'weights_sha256': '4c44efe84dba9c25dfa0158e34d7c72e03ef3bce57d8d7a56d14d6acb0a122ac', 'upstream_checkpoint_sha256': '9325d37f46cd42bb3d2fc0a4abf7c6876bced140440616552a0f54f55a435003', 'artifact_sha256': '0b26ec36d171bed520062e7cbef2694c30db7b3e3ff37839b8132dcfcdb6bb3e'}]

def implementation_identity():
    import paper_reconstruction_contracts
    root = Path(__file__).resolve().parent
    sdk = Path(paper_reconstruction_contracts.__file__).resolve().parent
    return digest([{'path': label+'/'+p.relative_to(folder).as_posix(), 'sha256': file_hash(p)}
                   for label, folder in [('stage3', root), ('contract', sdk)]
                   for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.py', '.json')])

def load_package(path, *, full=True):
    root = Path(path).resolve(); m = read_json(root/'alpha.json')
    require(m.get('schema_version') == 'stage3.alpha-runtime.v1', 'Unsupported alpha runtime')
    require(m.get('manifest_sha256') == digest({k:v for k,v in m.items() if k!='manifest_sha256'}), 'Model/config manifest changed')
    require(m['algorithm'] == ALGORITHM and m['correction_scorer_enabled'] is False, 'Unqualified alpha policy')
    require(len(m['rankers']) == len(FROZEN), 'Frozen model inventory changed')
    for r, expected in zip(m['rankers'], FROZEN):
        require(all(r[k] == expected[k] for k in ('name','context','weights_sha256','upstream_checkpoint_sha256'))
                and r['checkpoint']['sha256'] == expected['artifact_sha256'], 'Frozen model identity changed')
        verify_receipt(root, r['checkpoint'])
    m['implementation_sha256'] = implementation_identity()
    return root, m

"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import gzip
import json
import numpy as np
from .common import GEOMETRY, digest, read_json, require, SCHEMA, file_hash, normalized_matrix
from .validation import validate_schema

def relative_file(root, name):
    require(isinstance(name, str) and name and '\\' not in name and not PureWindowsPath(name).drive,
            'Portable path must be a relative POSIX path')
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Portable path escapes bundle')
    target = (root/path).resolve()
    require(target.is_relative_to(root.resolve()) and target.is_file(), 'Missing/escaped portable file')
    return target


def verify_receipt(root, receipt):
    path = relative_file(root, receipt['path'])
    require(path.stat().st_size == receipt['size'] and file_hash(path) == receipt['sha256'], 'Portable file checksum mismatch: '+receipt['path'])
    return path


@dataclass
class AssemblyInput:
    root: Path
    manifest: dict
    entries: dict
    edges: list
    pool: dict

    def hypotheses(self):
        """Lazy full occurrence archive, with IDs/counts checked on exhaustion."""
        ids = set(); occurrences = set(); count = 0
        candidates = {c['candidate_key']: c for c in self.manifest['candidates']}
        candidate_order = [c['candidate_key'] for c in self.manifest['candidates']]
        working = {e['id']: e for e in self.edges}; matched = set()
        with gzip.open(verify_receipt(self.root, self.manifest['archive']), 'rt', encoding='utf-8') as stream:
            for line in stream:
                group = json.loads(line); validate_schema(group, 'hypothesis'); key = group['hypothesis_id']
                require(key not in ids, 'Duplicate archived geometry'); ids.add(key)
                a, b = group['piece_a'], group['piece_b']; h = normalized_matrix(group['transform_b_to_a'])
                require(a < b and a in self.entries and b in self.entries, 'Unknown archived physical pair')
                require(group['observation_a'] == self.entries[a]['observation_key'] and group['observation_b'] == self.entries[b]['observation_key'], 'Archive observation frame mismatch')
                require(key == digest([a, b, group['observation_a'], group['observation_b'], h.tolist()]), 'Archive geometry identity mismatch')
                require(group['occurrences'], 'Empty archived geometry')
                group_ids = []
                for occurrence in group['occurrences']:
                    oid = occurrence['occurrence_id']; require(oid not in occurrences, 'Duplicate occurrence'); occurrences.add(oid)
                    require(oid == digest([occurrence['variant'], occurrence['row_index'], occurrence.get('record_id')]), 'Occurrence identity mismatch')
                    q, t = [candidates.get(occurrence[k]) for k in ('query_key', 'target_key')]
                    require(q is not None and t is not None and {q['piece_id'], t['piece_id']} == {a, b}, 'Archive candidate ownership mismatch')
                    require(occurrence['query_piece'] == q['piece_id'] and occurrence['target_piece'] == t['piece_id'], 'Occurrence physical ownership mismatch')
                    qi, ti = occurrence['pose']['query'], occurrence['pose']['target']
                    require(type(qi) is int and type(ti) is int and 0 <= qi < len(candidate_order) and 0 <= ti < len(candidate_order)
                            and candidate_order[qi] == occurrence['query_key'] and candidate_order[ti] == occurrence['target_key'], 'Occurrence candidate order mismatch')
                    require(q['observation_key'] == occurrence['query_observation'] and t['observation_key'] == occurrence['target_observation'], 'Occurrence observation mismatch')
                    require(np.allclose(normalized_matrix(occurrence['original_canonical_transform_b_to_a']), h, atol=1e-8), 'Occurrence transform mismatch')
                    directed = normalized_matrix(occurrence['pose']['alignment']['transform'])
                    expected = directed if q['piece_id'] == a else normalized_matrix(np.linalg.inv(directed))
                    require(np.allclose(expected, h, rtol=1e-8, atol=1e-6), 'Occurrence direction mismatch')
                    group_ids.append(oid); count += 1
                if key in working:
                    e = working[key]
                    require(np.array_equal(np.asarray(e['transform']), np.asarray(group['transform_b_to_a'])) and e['occurrence_ids'] == group_ids,
                            'Working edge differs from archive')
                    require(e['representative_occurrence'] in group_ids, 'Missing representative occurrence'); matched.add(key)
                yield group
        require(len(ids) == self.manifest['counts']['geometries'] and count == self.manifest['counts']['occurrences'], 'Archive inventory mismatch')
        require(matched == set(working), 'Working edge is absent from archive')

    def summary(self):
        adjacency = {p: set() for p in self.entries}
        for edge in self.edges: adjacency[edge['a']].add(edge['b']); adjacency[edge['b']].add(edge['a'])
        remaining = set(adjacency); components = []
        while remaining:
            todo = [min(remaining)]; found = set()
            while todo:
                p = todo.pop()
                if p not in found: found.add(p); todo.extend(adjacency[p]-found)
            remaining -= found; components.append(sorted(found))
        return {'status': 'ready_for_stage4_alpha', 'puzzle_id': self.manifest['puzzle_id'], **self.manifest['counts'],
                'candidate_graph_components': components, 'graph_connectivity_is_not_solvability': True,
                'demonstrated_reliable_piece_capacity': None, 'training_performed': False, 'global_assembly_performed': False}


def load_bundle(path, *, full=False):
    root = Path(path).resolve(); root = root.parent if root.is_file() else root
    m = read_json(root/'bundle.json'); validate_schema(m, 'bundle'); require(m.get('schema_version') == SCHEMA, 'Unsupported Stage 4 input schema')
    require(m.get('manifest_sha256') == digest({k: v for k, v in m.items() if k != 'manifest_sha256'}), 'Bundle manifest checksum mismatch')
    require(m['correction_scorer']['enabled'] is False and m['validation_truth_opened'] is False and m['protected_truth_opened'] is False,
            'Unqualified/protected alpha evidence')
    assets = {}
    for name, receipt in m['assets'].items():
        require(name == receipt['path'], 'Asset key mismatch'); assets[name] = verify_receipt(root, receipt)
    verify_receipt(root, m['archive']); pool = read_json(verify_receipt(root, m['working_pool']))
    validate_schema(pool, 'working_pool')
    require(pool['schema_version'] == 'stage3.working-pool.v1', 'Unknown working pool schema')
    entries = {}
    from PIL import Image
    for piece_id, piece in m['pieces'].items():
        require(piece_id == piece['piece_id'], 'Physical piece key mismatch')
        require(all(piece[k] in assets for k in ('image', 'mask', 'contour')), 'Unsealed native asset')
        for field in ('image', 'mask'):
            with Image.open(assets[piece[field]]) as image:
                require(list(image.size[::-1]) == piece['size_hw'], 'Native crop size mismatch')
        with np.load(assets[piece['contour']], allow_pickle=False) as z:
            require(set(z.files) == {'points_xy', *GEOMETRY, 'role_probabilities', 'transition_probabilities'}, 'Unexpected portable contour array')
            require(all(np.isfinite(z[k]).all() for k in z.files), 'Nonfinite portable contour')
            n = len(z['points_xy']); require(n >= 3, 'Empty portable contour')
            for name in z.files:
                shape = (n, 3) if name == 'role_probabilities' else (n, 2) if name in ('points_xy', 'tangent_xy', 'inward_normal_xy') else (n,)
                require(z[name].shape == shape and np.issubdtype(z[name].dtype, np.number), 'Portable contour shape/type mismatch')
        entries[piece_id] = {'piece_id': piece_id, 'observation_key': piece['observation_key'],
                             'image_path': str(assets[piece['image']]), 'mask_path': str(assets[piece['mask']]), 'size_hw': piece['size_hw']}
    require(len({p['observation_key'] for p in entries.values()}) == len(entries), 'Unregistered duplicate observation')
    candidates = m['candidates']; require(len({c['candidate_key'] for c in candidates}) == len(candidates), 'Duplicate portable candidate')
    require(all(c['piece_id'] in entries and c['observation_key'] == entries[c['piece_id']]['observation_key'] for c in candidates), 'Portable candidate frame mismatch')
    edges = []; pairs = {}; ids = set()
    for edge in pool['edges']:
        a, b = edge['piece_a'], edge['piece_b']; key = edge['hypothesis_id']
        require(a < b and a in entries and b in entries and key not in ids, 'Invalid working edge identity'); ids.add(key)
        require(edge['observation_a'] == entries[a]['observation_key'] and edge['observation_b'] == entries[b]['observation_key'], 'Working edge frame mismatch')
        matrix = normalized_matrix(edge['transform_b_to_a'])
        require(key == digest([a, b, edge['observation_a'], edge['observation_b'], matrix.tolist()]), 'Working edge geometry identity mismatch')
        require(np.isfinite(edge['weight']), 'Nonfinite working score')
        pair = (a, b); pairs[pair] = pairs.get(pair, 0)+1
        edges.append({**edge, 'id': key, 'a': a, 'b': b, 'transform': edge['transform_b_to_a']})
    require(all(n <= pool['maximum_per_physical_pair'] for n in pairs.values()), 'Working pool cap exceeded')
    require(len(edges) == m['counts']['working_edges'] and len(entries) == m['counts']['pieces'] and len(candidates) == m['counts']['candidates'], 'Portable inventory mismatch')
    result = AssemblyInput(root, m, entries, edges, pool)
    if full:
        for _ in result.hypotheses(): pass
    return result

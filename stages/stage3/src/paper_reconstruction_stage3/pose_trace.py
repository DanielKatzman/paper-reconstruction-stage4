"""Bounded, optional observation hooks. No trace value is a prediction input."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import time

import numpy as np

ACTIVE = None
LIMITS = {'rows_per_pair': 200000, 'rows_per_case': 2000000,
          'maximum_numeric_bytes': 384*1024*1024}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class TraceWriter:
    def __init__(self, root, signature, gallery, limits=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=False)
        self.signature = signature; self.gallery = gallery; self.limits = dict(LIMITS if limits is None else limits)
        if set(self.limits) != set(LIMITS) or any(type(v) is not int or v < 1 for v in self.limits.values()):
            raise ValueError('Invalid trace limits')
        self.rows = 0; self.numeric_bytes = 0; self.pair_rows = Counter(); self.files = []; self.skips = []
        self.events = []; self.events_truncated = 0; self.started = time.perf_counter(); self.io_seconds = 0.

    def save(self, kind, pair, arrays, metadata):
        """Truncate observation only, deterministically; never truncate search."""
        start = time.perf_counter(); pair = tuple(pair)
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        n = len(arrays['matrix'])
        if any(v.dtype.hasobject or len(v) != n for v in arrays.values()):
            raise ValueError('Trace must contain aligned numeric/non-object arrays')
        bytes_per_row = sum(v.nbytes for v in arrays.values())//max(1, n)
        cap = min(n, self.limits['rows_per_pair']-self.pair_rows[pair], self.limits['rows_per_case']-self.rows,
                  (self.limits['maximum_numeric_bytes']-self.numeric_bytes)//max(1, bytes_per_row))
        cap = max(0, cap)
        if cap < n:
            self.skips.append({'kind': kind, 'pair': list(pair), 'observed_rows': n, 'saved_rows': cap,
                               'reason': 'observation_budget_only', 'search_affected': False})
        if cap:
            arrays = {k: np.ascontiguousarray(v[:cap]) for k, v in arrays.items()}
            name = '%04d_%s.npz' % (len(self.files), kind)
            path = self.root/name
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            self.files.append({'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'rows': cap, 'observed_rows': n, 'pair': list(pair), 'kind': kind,
                'numeric_bytes': sum(v.nbytes for v in arrays.values()), **metadata})
            self.rows += cap; self.pair_rows[pair] += cap
            self.numeric_bytes += sum(v.nbytes for v in arrays.values())
        self.io_seconds += time.perf_counter()-start

    def overlap(self, q, t, phase, coarse, fine, base, parent_index=None):
        pair = sorted(self.gallery[i][1]['entry']['piece_id'] for i in (q, t))
        values = coarse+fine
        self.save('overlap', pair, {
            'matrix': np.asarray([p['alignment']['transform'] for p in values], np.float64).reshape(-1, 3, 3),
            'parameters': np.asarray([p['_parameters'] for p in values], np.float64).reshape(-1, 4),
            'raw_id': np.asarray([p['proposal_id'] for p in values], 'S24'),
            'origin': np.asarray([0]*len(coarse)+[1]*len(fine), np.uint8),
            'geometry_pass': np.asarray([p['alignment']['normalized_residual'] <= base['maximum_normalized_residual']
                and p['alignment']['inward_normal_penalty'] <= base['maximum_normal_penalty'] for p in values], bool),
            'underconstrained': np.asarray([p['alignment']['underconstrained'] for p in values], bool),
            **{k: np.asarray([p['alignment'][k] for p in values], np.float64) for k in
               ('scale', 'geometry_score', 'normalized_residual', 'inward_normal_penalty', 'shape_anisotropy')}
        }, {'query': q, 'target': t, 'phase': phase, 'parent_index': parent_index,
            'coordinates': 'target crop -> query crop', 'target_traversal': 'reversed once by unchanged fitter'})

    def seal(self):
        data = {'schema_version': 'stage3.prearchive-observations.v1', 'inference_signature': self.signature,
            'gallery': [{'key': k, 'piece': item['entry']['piece_id']} for k, item in self.gallery],
            'limits': self.limits, 'rows': self.rows, 'numeric_bytes': self.numeric_bytes,
            'files': self.files, 'truncations': self.skips, 'bootstrap_events_truncated': self.events_truncated,
            'complete_within_observed_calls': not self.skips and not self.events_truncated,
            'bootstrap_events': self.events, 'io_seconds': self.io_seconds,
            'references_opened': False, 'search_affected': False,
            'limitations': ['Raw overlap traces cover returned coarse/fine fits, not every internal similarity-grid trial.',
                'No raw pose is supplied to serving; an omitted/unseen fit is not evidence of nonexistence.']}
        data['manifest_sha256'] = digest(data)
        (self.root/'manifest.json').write_text(json.dumps(data, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        return data


def overlap_trace(q, t, phase, coarse, fine, base, parent_index=None):
    if ACTIVE is not None:
        ACTIVE.overlap(q, t, phase, coarse, fine, base, parent_index)


def bootstrap_stage(q, t, values, stage):
    if ACTIVE is not None:
        # At most 96 screened fits per selected parent; budget this small event
        # stream separately rather than allowing a truncated raw file to erase it.
        if len(ACTIVE.events) >= 65536:
            ACTIVE.events_truncated += 1
            return
        ACTIVE.events.append({'query': q, 'target': t, 'stage': stage,
                              'raw_ids': [p['proposal_id'] for p in values]})


def generated_trace(pair, arrays, audit):
    if ACTIVE is not None:
        from .pose_inference_contracts import STAGES, ORIGINS
        ACTIVE.save('generated', pair, arrays, {'stages': list(STAGES), 'origins': list(ORIGINS),
                                               'stage_counts_before_trace_limits': audit['stage_counts']})

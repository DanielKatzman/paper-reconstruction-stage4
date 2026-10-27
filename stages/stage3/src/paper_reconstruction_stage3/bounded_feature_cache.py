"""Immutable-input, byte-bounded feature reuse for the isolated candidate.

Copied into the candidate as ``bounded_feature_cache.py``. Limits describe
retained object graphs, not process RSS or temporary/GPU allocations.
"""
from collections import OrderedDict
from collections.abc import MutableMapping
import json
from pathlib import Path
import sys

import numpy as np


def byte_limit(value):
    if type(value) is not int or not 0 <= value <= sys.maxsize:
        raise ValueError('Cache byte limits must be nonnegative integers within the platform size range')
    return value


def retained_objects(value):
    """Sizes of distinct retained containers/buffers, including ndarray bases.

    sys.getsizeof owns ndarray storage only for owning arrays. Views retain
    their base, which is traversed and charged once, including across entries.
    Object arrays and unknown external storage are rejected rather than counted
    as if their referents or backing allocation cost nothing.
    """
    sizes = {}
    def visit(obj):
        identity = id(obj)
        if identity in sizes:
            return
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError('Object arrays are not immutable numeric cache payloads')
            sizes[identity] = sys.getsizeof(obj)
            if obj.base is not None:
                visit(obj.base)
        elif isinstance(obj, memoryview):
            sizes[identity] = sys.getsizeof(obj); visit(obj.obj)
        elif isinstance(obj, dict):
            sizes[identity] = sys.getsizeof(obj)
            for key, item in obj.items():
                visit(key); visit(item)
        elif isinstance(obj, (tuple, list)):
            sizes[identity] = sys.getsizeof(obj)
            for item in obj:
                visit(item)
        elif isinstance(obj, (str, bytes, bytearray, int, float, bool, np.generic)) or obj is None:
            sizes[identity] = sys.getsizeof(obj)
        else:
            raise TypeError('Unsupported retained storage: '+type(obj).__name__)
    visit(value)
    return sizes


class ByteBoundedLRU(MutableMapping):
    """One cache, with shared-allocation accounting across retained entries.

    Cached values must remain unmodified, as in the frozen FeatureBank. A fixed
    per-entry bookkeeping allowance also bounds tiny entries. Actual allocator
    and map overhead are not an RSS guarantee and are measured separately.
    """
    ENTRY_ALLOWANCE = 256

    def __init__(self, maximum_bytes):
        self.maximum_bytes = byte_limit(maximum_bytes)
        self._items = OrderedDict(); self._resources = {}
        self.retained_bytes = self.peak_retained_bytes = self.peak_entries = 0
        self.hits = self.misses = self.insertions = self.evictions = 0
        self.oversize_bypasses = self.disabled_bypasses = self.largest_item_bytes = 0

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, key):
        # Instrumentation membership probes must not alter recency or counters.
        return key in self._items

    def __getitem__(self, key):
        try:
            value, _ = self._items[key]
        except KeyError:
            self.misses += 1; raise
        self.hits += 1; self._items.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        _, resources = self._items.pop(key)
        self.retained_bytes -= self.ENTRY_ALLOWANCE
        for identity, size in resources.items():
            self._resources[identity][1] -= 1
            if self._resources[identity][1] == 0:
                self.retained_bytes -= size; del self._resources[identity]

    def put(self, key, value):
        # Size/validate before mutating any retained item. A failed calculation
        # happens before put and never poisons the cache.
        resources = retained_objects(key)
        resources.update(retained_objects(value))
        standalone = self.ENTRY_ALLOWANCE+sum(resources.values())
        self.largest_item_bytes = max(self.largest_item_bytes, standalone)
        if key in self._items:
            del self[key]
        if self.maximum_bytes == 0:
            self.disabled_bypasses += 1; return False
        if standalone > self.maximum_bytes:
            self.oversize_bypasses += 1; return False
        def added():
            return self.ENTRY_ALLOWANCE+sum(size for identity, size in resources.items() if identity not in self._resources)
        while self._items and self.retained_bytes+added() > self.maximum_bytes:
            del self[next(iter(self._items))]; self.evictions += 1
        for identity, size in resources.items():
            if identity not in self._resources:
                self._resources[identity] = [size, 0]; self.retained_bytes += size
            self._resources[identity][1] += 1
        self.retained_bytes += self.ENTRY_ALLOWANCE
        self._items[key] = (value, resources); self.insertions += 1
        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        self.peak_entries = max(self.peak_entries, len(self))
        if self.retained_bytes > self.maximum_bytes:
            raise RuntimeError('Retained cache accounting exceeded its byte limit')
        return True

    def summary(self):
        return {'maximum_bytes': self.maximum_bytes, 'retained_bytes': self.retained_bytes,
            'peak_retained_bytes': self.peak_retained_bytes, 'entries': len(self), 'peak_entries': self.peak_entries,
            'hits': self.hits, 'misses': self.misses, 'insertions': self.insertions, 'evictions': self.evictions,
            'oversize_bypasses': self.oversize_bypasses, 'disabled_bypasses': self.disabled_bypasses,
            'largest_item_standalone_bytes': self.largest_item_bytes,
            'entry_bookkeeping_allowance_bytes': self.ENTRY_ALLOWANCE}


class FeatureBank:
    """The frozen G+P+A arithmetic, with puzzle-scoped byte-bounded reuse."""
    def __init__(self, inputs, *, feature_budget_bytes=None, image_budget_bytes=None):
        if feature_budget_bytes is None or image_budget_bytes is None:
            policy = json.loads(Path(__file__).with_name('feature_cache_policy.json').read_text())
            if policy.get('schema_version') != 'stage3.feature-image-cache.v1':
                raise ValueError('Unknown feature/image cache configuration')
            if feature_budget_bytes is None:
                feature_budget_bytes = policy['feature_bytes']
            if image_budget_bytes is None:
                image_budget_bytes = policy['image_bytes']
        self.inputs = inputs
        self.cache = ByteBoundedLRU(feature_budget_bytes)
        self.images = ByteBoundedLRU(image_budget_bytes)

    def image(self, owner):
        from paper_reconstruction_stage3.features import masked_pixels
        obs = self.inputs.gallery[owner][1]['entry']['observation_key']
        try:
            return self.images[obs]
        except KeyError:
            value = masked_pixels(self.inputs.contours[obs])
            self.images.put(obs, value)
            return value

    def features(self, owner):
        from paper_reconstruction_stage3.features import FeatureConfig, fragment_features
        try:
            return self.cache[owner]
        except KeyError:
            pass
        item = self.inputs.gallery[owner][1]; contour = self.inputs.contours[item['entry']['observation_key']]
        f = fragment_features(contour, item['candidate'], FeatureConfig(512, 33, 0), appearance=True, pixels=self.image(owner))
        strip = np.moveaxis(f['strip'], -1, 0).copy(); valid = strip[3]
        rgb = strip[:3].sum(axis=2).T/np.maximum(valid.sum(axis=1), 1e-6)[:, None]
        seq = np.concatenate((f['geometry'], np.full((512, 1), np.log1p(f['observed_length_px'][0]), np.float32),
                              f['probabilities'], np.column_stack((rgb, valid.mean(axis=1)))), axis=1).astype(np.float32)
        value = {'sequence': seq, 'strip': strip, 'points': f['points_xy_crop'],
                 'normals': f['inward_normal_xy_crop'], 'observed_length': f['observed_length_px']}
        self.cache.put(owner, value)
        return value

    def summary(self):
        return {'schema_version': 'stage3.feature-image-cache-report.v1', 'scope': 'One immutable-input puzzle/FeatureBank',
            'features': self.cache.summary(), 'images': self.images.summary(),
            'maximum_combined_accounted_bytes': self.cache.maximum_bytes+self.images.maximum_bytes,
            'accounted_combined_peak_upper_bound_bytes': self.cache.peak_retained_bytes+self.images.peak_retained_bytes,
            'accounting': 'Distinct retained numeric buffers including bases, referenced Python containers/keys and a per-entry allowance. '
                          'No double counting shared buffers within a cache. Separate cache sums are conservative. '
                          'Allocator overhead, in-flight references/temporaries and GPU memory are outside this limit; inspect process RSS.'}

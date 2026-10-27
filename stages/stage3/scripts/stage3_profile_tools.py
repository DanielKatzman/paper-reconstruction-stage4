"""Development-only instrumentation; never imported by the inference package."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from importlib import import_module
from itertools import zip_longest
from pathlib import Path
from unittest.mock import patch
import sys
import time


class FirstPairComplete(Exception):
    """Stop a diagnostic replay after a fully committed physical-pair cache."""


class Measurements:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.stack = []
        self.rows = defaultdict(lambda: {'calls': 0, 'inclusive_seconds': 0., 'exclusive_seconds': 0., 'exception_exits': 0})
        self.counters = defaultdict(int)
        self.gallery = None
        self.receipts = None
        self.pairs = []
        self.pair_started = None
        self.events = []
        self.started = None

    @contextmanager
    def measure(self, label):
        frame = [label, self.clock(), 0.]
        self.stack.append(frame)
        row = self.rows[label]
        row['calls'] += 1
        try:
            yield
        except BaseException:
            row['exception_exits'] += 1
            raise
        finally:
            elapsed = self.clock() - frame[1]
            self.stack.pop()
            row['inclusive_seconds'] += elapsed
            row['exclusive_seconds'] += max(0., elapsed-frame[2])
            if self.stack:
                self.stack[-1][2] += elapsed

    def functions(self):
        return [{'function': name, **row} for name, row in sorted(
            self.rows.items(), key=lambda item: (-item[1]['inclusive_seconds'], item[0]))]

    def before(self, label, args, kwargs):
        if label == 'FeatureBank.features':
            owner = args[1] if len(args) > 1 else kwargs['owner']
            self.counters['rank_features_cache_'+('hit' if owner in args[0].cache else 'miss')] += 1
        elif label == 'FeatureBank.image':
            owner = args[1] if len(args) > 1 else kwargs['owner']
            obj = args[0]
            obs = obj.inputs.gallery[owner][1]['entry']['observation_key']
            self.counters['rank_images_cache_'+('hit' if obs in obj.images else 'miss')] += 1
        elif label == 'ObservableInputs.features':
            return args[0].feature_constructions
        elif label == 'boundary.correspondences':
            a, b = args[0], args[2]
            workload = 'visual' if any(frame[0] == 'visual.visual_profile' for frame in self.stack) else 'other'
            self.counters['correspondence_calls_'+workload] += 1
            self.counters['point_segment_comparisons_'+workload] += len(a)*max(0, len(b)-1)
        elif label == 'visual.cache_evaluate':
            return args[0].hits, args[0].misses
        return None

    def after(self, label, args, result, token):
        if label == 'inputs.production_inputs':
            _, self.gallery, self.receipts = result
        elif label == 'ObservableInputs.features':
            misses = args[0].feature_constructions-token
            self.counters['geometry_features_cache_miss'] += misses
            self.counters['geometry_features_cache_hit'] += int(misses == 0)
        elif label == 'visual.cache_evaluate':
            self.counters['visual_result_cache_hit'] += args[0].hits-token[0]
            self.counters['visual_result_cache_miss'] += args[0].misses-token[1]


# Inclusive rows deliberately overlap; exclusive rows partition tracked work.
TARGETS = {
    'alpha_inference': [('production_inputs', 'inputs.production_inputs'), ('retrieve_parents', 'ranking.retrieve_parents'),
        ('fuse_parents', 'ranking.fuse_parents'), ('attach_windows', 'windows.attach_windows'),
        ('sliding_starts', 'poses.sliding_starts'), ('FeatureBank.features', 'FeatureBank.features'),
        ('FeatureBank.image', 'FeatureBank.image'), ('FrozenRankers.__init__', 'ranking.model_setup'),
        ('FrozenRankers.score', 'ranking.score')],
    'baselines': [('ObservableInputs.__init__', 'inputs.ObservableInputs'), ('ObservableInputs.features', 'ObservableInputs.features'),
        ('ObservableInputs.subarcs', 'ObservableInputs.subarcs'), ('ObservableInputs.mask', 'ObservableInputs.mask'),
        ('ObservableInputs.mask_samples', 'ObservableInputs.mask_samples')],
    'features': [('fragment_features', 'features.fragment_features'), ('masked_pixels', 'features.image_decode'),
        ('bilinear', 'features.bilinear')],
    'pose_geometry': [('fit_parent', 'poses.fit_parent'), ('hypotheses', 'poses.hypotheses')],
    'pose_correction_geometry': [('curve', 'features.curve_construction')],
    'pose_overlap_geometry': [('sliding_fits', 'poses.sliding_fits')],
    'pose_perspective_geometry': [('refine', 'poses.projective_refine')],
    'pose_inference_geometry': [('generate', 'poses.generate')],
    'pose_boundary_evidence': [('evidence', 'boundary.evidence'), ('correspondences', 'boundary.correspondences')],
    'pose_visual_evidence': [('visual_profile', 'visual.visual_profile')],
    'projective': [('warp_check', 'guards.warp_check'), ('warp_normals', 'guards.warp_normals')],
    'alignment': [('collision_fraction', 'guards.collision_fraction'), ('fit_transforms', 'poses.fit_transforms'),
        ('_subarcs', 'poses.subarc_construction')],
    'alpha_bundle': [('export_bundle', 'export.bundle'), ('group_records', 'export.group_records'),
        ('working_pool', 'export.working_pool'), ('file_hash', 'io.file_hash')],
    'alpha_inputs': [('unchanged', 'inputs.verify_unchanged')],
    'contracts': [('atomic_json', 'io.atomic_json'), ('sha256', 'io.sha256'), ('array_bundle_sha256', 'io.array_hash'),
        ('digest', 'io.canonical_digest')],
    '@paper_reconstruction_contracts.reader': [('load_bundle', 'validation.load_bundle'), ('verify_receipt', 'validation.file_receipt')],
    '@paper_reconstruction_contracts.validation': [('validate_schema', 'validation.schema')],
}


@contextmanager
def instrument(metrics, on_write, targets=None):
    """Patch functions and already-imported aliases in this process only.

    Files, policy, ordering and returned production values are unchanged. New
    local imports resolve to patched functions too. All patches restore on exit.
    """
    originals = {}
    locations = []
    for module_name, methods in (TARGETS if targets is None else targets).items():
        module = import_module(module_name[1:] if module_name.startswith('@') else 'paper_reconstruction_stage3.'+module_name)
        for path, label in methods:
            owner = module
            parts = path.split('.')
            for part in parts[:-1]:
                owner = getattr(owner, part)
            original = getattr(owner, parts[-1])
            originals[original] = label
            locations.append((owner, parts[-1], original))
    candidate = sys.modules.get('paper_reconstruction_stage3.exact_acceleration')
    if targets is None and candidate is not None:
        original = candidate.VisualCache.evaluate
        originals[original] = 'visual.cache_evaluate'
        locations.append((candidate.VisualCache, 'evaluate', original))
    wrappers = {}
    for original, label in originals.items():
        def make(function, name):
            @wraps(function)
            def wrapped(*args, **kwargs):
                with metrics.measure(name):
                    token = metrics.before(name, args, kwargs)
                    result = function(*args, **kwargs)
                    metrics.after(name, args, result, token)
                if name == 'io.atomic_json':
                    path = args[0] if args else kwargs['path']
                    value = args[1] if len(args) > 1 else kwargs['value']
                    on_write(Path(path), value)
                return result
            return wrapped
        wrappers[original] = make(original, label)
    # Imported function aliases must refer to the same wrapper, not nested ones.
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith(('paper_reconstruction_stage3', 'paper_reconstruction_contracts')):
            continue
        for name, value in list(vars(module).items()):
            if callable(value):
                try:
                    watched = value in wrappers
                except TypeError:
                    watched = False
                if watched:
                    locations.append((module, name, value))
    with ExitStack() as stack:
        seen = set()
        for owner, name, original in locations:
            if (id(owner), name) not in seen:
                stack.enter_context(patch.object(owner, name, wrappers[original]))
                seen.add((id(owner), name))
        yield


def compare_groups(expected, actual):
    """Exhaust both streams: the SDK checks archive counts at exhaustion."""
    missing = object()
    counts = {'expected_groups': 0, 'actual_groups': 0, 'different_groups': 0}
    differences = []
    for left, right in zip_longest(expected, actual, fillvalue=missing):
        counts['expected_groups'] += int(left is not missing)
        counts['actual_groups'] += int(right is not missing)
        if left != right:
            counts['different_groups'] += 1
            if len(differences) < 8:
                differences.append({'expected_id': None if left is missing else left['hypothesis_id'],
                                    'actual_id': None if right is missing else right['hypothesis_id']})
    return {'equal': counts['different_groups'] == 0, **counts, 'difference_examples': differences,
            'comparison': 'exact complete geometry/support/occurrence/evidence objects; no tolerance or ignored fields'}


def process_memory():
    """OS process high-water RSS, including native arrays; no optional package."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            size = ctypes.c_size_t
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    *[(name, size) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                        'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]]
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            values = Counters(); values.cb = ctypes.sizeof(values)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return {'peak_rss_bytes': values.PeakWorkingSetSize, 'current_rss_bytes': values.WorkingSetSize,
                    'current_private_bytes': values.PrivateUsage, 'source': 'Windows GetProcessMemoryInfo'}
        import resource
        return {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
                'source': 'getrusage'}
    except (ImportError, OSError, AttributeError) as error:
        return {'peak_rss_bytes': None, 'reason': type(error).__name__+': '+str(error)}

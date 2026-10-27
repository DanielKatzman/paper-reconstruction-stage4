"""Exact source-band preparation for the isolated visual-reuse-v3 runtime.

Copied as visual_preparation.py. One instance belongs to one physical pair,
whose candidate curves and observed pixels remain immutable. No global cache,
cross-offset association reuse, reduced sampling, or learned scoring is used.
"""
import numpy as np

DEFAULT_PREPARATION_BYTES = 8 * 1024 * 1024


def prepare_source(first, pixels, policy):
    # Keep the frozen visual_profile operations and dtypes, including the
    # order of mean/threshold reductions. Do not replace this with rank features.
    from paper_reconstruction_stage3.features import bilinear
    from paper_reconstruction_stage3.pose_observability_geometry import tangent
    rgb, mask = pixels
    a = np.asarray(first['points_xy_crop'], float)
    an = np.asarray(first['inward_normal_xy_crop'], float)
    direction = tangent(first)
    depths = np.asarray(policy['visual_depths_query_px'])
    qa = a[:, None] + an[:, None] * depths[None, :, None]
    ia = bilinear(rgb, qa).mean(axis=1)
    va = (bilinear(mask, qa) >= policy['visual_mask_validity']).all(axis=1)
    return {'a': a, 'an': an, 'direction': direction, 'depths': depths, 'ia': ia, 'va': va}


class VisualPreparationCache:
    """Byte-bounded query-only reuse under the physical-pair input contract.

    Owner identifies an immutable candidate/observation in that pair. A new
    pair gets a new instance. Depths and mask threshold are also keyed, so a
    changed sampling policy cannot reuse a differently prepared source band.
    Cached arrays are read only by the visual calculation; images are not
    retained by this cache. Accounting includes bases of retained curve views.
    """
    def __init__(self, pair_key=(), maximum_bytes=DEFAULT_PREPARATION_BYTES, *, cache_factory=None):
        if cache_factory is None:
            from paper_reconstruction_stage3.bounded_feature_cache import ByteBoundedLRU
            cache_factory = ByteBoundedLRU
        self.sources = cache_factory(maximum_bytes)
        self.pair_key = tuple(pair_key)
        self.center_reuses = self.center_byte_mismatch_fallbacks = 0

    def source(self, owner, first, pixels, policy):
        depths = np.asarray(policy['visual_depths_query_px'])
        key = (owner, depths.dtype.str, tuple(depths.shape), depths.tobytes(),
               float(policy['visual_mask_validity']).hex())
        try:
            return self.sources[key]
        except KeyError:
            value = prepare_source(first, pixels, policy)
            self.sources.put(key, value)
            return value

    def summary(self):
        return {'schema_version': 'stage3.visual-preparation-report.v1',
                'scope': 'One physical pair; immutable candidate curves and observed pixels; query-only bands',
                'pair': list(self.pair_key), 'sources': self.sources.summary(),
                'center_reuses': self.center_reuses,
                'center_byte_mismatch_fallbacks': self.center_byte_mismatch_fallbacks,
                'center_rule': 'Reuse only when the zero-offset translated float64 matrix has identical bytes'}

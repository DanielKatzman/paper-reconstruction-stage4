"""Post-seal regression measurements; no serving acceptance thresholds are changed."""
import math


def approximate_construction(quality):
    gauge=quality['common_gauge'];rows=gauge['per_piece']
    if not gauge['whole_page_error_available'] or quality['inventory_coverage']!=1.:return False
    # Existing offline bank-witness tolerance, applied to every saved probe here.
    maximum=gauge['non_anchor_probe_errors_reference_px']['maximum']
    if maximum is None or not math.isfinite(maximum) or maximum>48.:return False
    for p in rows:
        angle=p.get('local_orientation_error_degrees');scale=p.get('local_scale_ratio')
        if angle is None or scale is None or not math.isfinite(angle) or not math.isfinite(scale):return False
        if abs(angle)>1e-3 or abs(scale-1.)>1e-5:return False
    return True

"""Explicit acquisition assumptions, separate from immutable upstream evidence."""
import json
from pathlib import Path
import numpy as np
from .outputs import digest, file_hash

POLICY = 'shared-scale-complete-rectangle.v1'
RIGID_TOLERANCE = 1e-6
MASK_PROVENANCE = {
    'exact_generator_development': ('generator_exact_bypass',),
    'predicted_segmentation': ('segformer_v3_predicted', 'segformer_v3_reviewed'),
}


def rigid_check(h):
    h = np.asarray(h, float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
        return {'valid': False, 'reason': 'profile_nonrigid_transform'}
    h = h / h[2, 2]
    residual = max(float(np.max(np.abs(h[2] - [0, 0, 1]))),
                   float(np.max(np.abs(h[:2, :2].T @ h[:2, :2] - np.eye(2)))),
                   abs(float(np.linalg.det(h[:2, :2])) - 1.))
    return {'valid': residual <= RIGID_TOLERANCE, 'reason': 'checked' if residual <= RIGID_TOLERANCE else 'profile_nonrigid_transform',
            'rigidity_residual': residual, 'numerical_tolerance': RIGID_TOLERANCE}


def require_rigid(h):
    if not rigid_check(h)['valid']:
        raise ValueError('Constraint profile forbids scale, reflection, shear or perspective')


class CaptureRotationError(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def require_native_rotation(h, capture_binding):
    """Apply the authenticated rotation family consistently at every replay."""
    if not rigid_check(h)['valid']:
        raise CaptureRotationError('nonrigid_transform', 'Native pose forbids scale, reflection, shear or perspective')
    family = (capture_binding or {}).get('rotation_family')
    if family not in {'quarter_turn', 'continuous_rigid'}:
        raise CaptureRotationError('undeclared_rotation_family', 'Native pose requires authenticated rigid capture metadata')
    rotation = np.asarray(h, float)[:2, :2] / np.asarray(h, float)[2, 2]
    if family == 'quarter_turn' and not np.allclose(rotation, np.rint(rotation), atol=1e-7, rtol=0.):
        raise CaptureRotationError('undeclared_rotation', 'Quarter-turn pose seed contains an undeclared rotation')


def bind_profile(problem, config, profile_path, capture_path):
    if config.page_policy == 'disabled.v1':
        if profile_path or capture_path:
            raise ValueError('Constraint profiles require the explicit page policy')
        return
    if not profile_path or not capture_path:
        raise ValueError('Complete-page policy needs --constraint-profile and --capture-profile')
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8-sig'))
    mask_mode = profile.get('mask_mode')
    if mask_mode not in MASK_PROVENANCE:
        raise ValueError('Unsupported mask provenance in complete-page constraint profile')
    expected = {'schema_version': 'stage4.constraint-profile.v1', 'policy': POLICY,
                'motion_model': 'proper_rigid', 'relative_piece_scale': 1.0,
                'page_completeness': 'all_pieces_one_page', 'page_shape': 'rectangle',
                'page_dimensions': 'unknown', 'page_aspect_ratio': 'unknown', 'dpi': 'unknown',
                'mask_mode': mask_mode, 'rigid_tolerance': RIGID_TOLERANCE}
    if profile != expected:
        raise ValueError('Unsupported or mismatched complete-page constraint profile')
    capture = json.loads(Path(capture_path).read_text(encoding='utf-8-sig'))
    if not (capture.get('schema_version') == 'stage1.capture-profile.v1' and
            capture.get('motion_model') == 'rotation_translation' and capture.get('relative_piece_scale') == 1.0 and
            capture.get('shared_native_pixel_scale') is True and capture.get('per_piece_resize') is False and
            capture.get('perspective') is False):
        raise ValueError('Capture profile does not declare shared native scale and rigid acquisition')
    perturbation = capture.get('mask_perturbation') if mask_mode == 'exact_generator_development' else capture.get('mask_perturbation', False)
    if capture.get('stage1_masks') not in MASK_PROVENANCE[mask_mode] or perturbation is not False:
        raise ValueError('Capture mask provenance does not match the declared mask mode')
    sha = file_hash(capture_path)
    provenance = problem.manifest.get('provenance', {})
    while 'source_provenance' in provenance:
        provenance = provenance['source_provenance']
    receipts = provenance.get('inputs', [])
    if not any(isinstance(r, dict) and r.get('sha256') == sha and r.get('size') == Path(capture_path).stat().st_size for r in receipts):
        raise ValueError('Capture profile is not bound into this bundle provenance')
    problem.constraint_profile = {'profile': profile, 'profile_sha256': digest(profile),
        'profile_file_sha256': file_hash(profile_path), 'capture_profile_sha256': sha,
        'capture_profile_file_size': Path(capture_path).stat().st_size}
    rotation_family=capture.get('rotation_family','unknown')
    if mask_mode == 'predicted_segmentation' and rotation_family in {'unknown','continuous','continuous_rigid'}:
        rotation_family='continuous_rigid'
    problem.capture_metadata={'rotation_family':rotation_family,
        'capture_profile_sha256':sha}


def require_bound(problem, config):
    if config.page_policy != 'disabled.v1' and not problem.constraint_profile:
        raise ValueError('Complete-page policy requires a bound acquisition and constraint profile')

"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
import json
import hashlib
from pathlib import Path
import numpy as np
from typing import Any

GEOMETRY = ("arc_length_px", "tangent_xy", "inward_normal_xy", "curvature_per_px", "corner_strength", "inward_thickness_px")


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

SCHEMA = "stage3.assembly-evidence.v2"

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def normalized_matrix(value):
    h = np.asarray(value, dtype=np.float64)
    require(h.shape == (3, 3) and np.isfinite(h).all(), 'Invalid native transform')
    require(abs(np.linalg.det(h)) > 1e-12 and abs(h[2, 2]) > 1e-12, 'Singular native transform')
    return h / h[2, 2]

def read_json(path):
    def reject(value): raise ContractError('Nonfinite JSON constant: '+value)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)
    require(isinstance(value, dict), 'JSON object required')
    return value

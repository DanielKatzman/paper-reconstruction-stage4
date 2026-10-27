from __future__ import annotations

import hashlib
import json
import os
import ast
import math
import mmap
import struct
import time
from tempfile import NamedTemporaryFile, TemporaryFile
from zipfile import ZipFile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

ROLE_NAMES = ["seam", "page_border", "uncertain"]
FROZEN = {
    "canonical": "ae1d7dd59504f39a15ffa517fe488a31441056e15d514da99e10ddde955d543c",
    "observations": "c5ab61fdbf2647c8ed981642719ac70788de56400b2095802116c328380b0833",
    "boundaries": "6f0b7adee1bff720e387b91917466337ee87ee7b1fcb08fd376aa4358d0f6c50",
    "onnx": "3e80716280375146e2ef66942688e2a262e503df1011918caff262f0bae388a4",
    "checkpoint": "36dd4eb0f13bc49fc2efc791cca1de03d34997704e11cce5d7cff8a326c1a6fc",
    "release": "0381683f5c8226974d73915b355bdd5936d80707219e7f083dcdf08f1ecfcfbc",
}
ROOT_CONTRACTS = {
    "canonical": ("dataset_manifest.json", "stage2.canonical-dataset.v2", "dataset_fingerprint"),
    "observations": ("observation_dataset_manifest.json", "stage2.observation-dataset.v1", "observation_dataset_fingerprint"),
    "boundaries": ("boundary_dataset_manifest.json", "stage2.boundary-dataset.v1", "boundary_dataset_fingerprint"),
    "predictions": ("prediction_manifest.json", "stage2.boundary-role-predictions.v1", None),
}
GEOMETRY = ("arc_length_px", "tangent_xy", "inward_normal_xy", "curvature_per_px", "corner_strength", "inward_thickness_px")


from paper_reconstruction_contracts import ContractError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


_REPLACE_RETRY_DELAYS = (.02, .05, .1, .2, .4, .8, 1., 1., 1., 1.)


def atomic_replace(temporary: Path, target: Path) -> None:
    """Replace a closed file, tolerating brief Windows access/sharing denials.

    Keep the old target intact until replacement succeeds. Permanent access
    errors still propagate after a bounded 5.57 seconds of retry delays;
    unrelated errors (including POSIX permission failures) are immediate.
    """
    for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(temporary, target)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS[attempt])


def _json_boolean(value):
    # Numerical comparisons can produce np.bool_; persist the same JSON boolean.
    # Keep unsupported arrays/scalars and nonfinite values subject to strict errors.
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Create a unique sibling; close it before replacement on Windows.
        with NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_boolean)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A scanner can still hold a failed temporary file. Do not
                # hide the original write error or damage the prior target.
                pass


def resolve(root: Path, relative: str, base: Path | None = None) -> Path:
    require(isinstance(relative, str) and bool(relative), "A nonempty relative artifact path is required")
    require(not PureWindowsPath(relative).is_absolute() and not Path(relative).is_absolute(), f"Absolute artifact path is not portable: {relative}")
    target = ((base or root) / relative.replace("\\", "/")).resolve()
    require(target.is_relative_to(root.resolve()), f"Artifact escapes its dataset root: {relative}")
    require(target.is_file(), f"Missing artifact: {target}")
    return target


def check_hash(path: Path, expected: str) -> None:
    require(isinstance(expected, str) and len(expected) == 64, f"Missing/invalid SHA-256 for {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch: {path}")


def _hash_fortran_payload(h, stream, shape: tuple[int, ...], itemsize: int, expected: int, context: str) -> None:
    """Reorder opaque element bytes, without interpreting supervision values."""
    with TemporaryFile() as raw:
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            raw.write(block)
        require(total == expected, f"Truncated/inconsistent NPY data: {context}")
        if not expected:
            return
        raw.flush()
        with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            # Void elements preserve every byte, including endian/NaN payloads.
            opaque = np.ndarray(shape, dtype=f"V{itemsize}", buffer=mapped, order="F")
            iterator = chunk = None
            try:
                with np.nditer(opaque, flags=["buffered", "external_loop"],
                               op_flags=["readonly"], order="C", casting="no",
                               buffersize=max(1, (1024 * 1024) // itemsize)) as iterator:
                    for chunk in iterator:
                        h.update(chunk.tobytes(order="C"))
            finally:
                # Release buffer references before closing the map (also on Windows).
                chunk = iterator = opaque = None


def array_bundle_sha256(path: Path) -> str:
    """Stage 2 hashes canonical array bytes, not compressed NPZ file bytes.

    Stream C-order payloads; spool Fortran-order payloads to temporary disk
    and reorder opaque bytes in bounded chunks. No supervision values are
    interpreted. Both layouts match Stage 2's np.ascontiguousarray checksum.
    """
    h = hashlib.sha256()
    with ZipFile(path) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)) and all(x.endswith(".npy") for x in members), "Invalid NPZ member names")
        for member in sorted(members):
            context = f"{path}:{member}"
            with archive.open(member) as stream:
                magic = stream.read(8)
                require(magic[:6] == b"\x93NUMPY", "Invalid NPY header")
                version = tuple(magic[6:8])
                require(version in {(1, 0), (2, 0), (3, 0)}, "Unsupported NPY format version")
                width = 2 if version == (1, 0) else 4
                raw_size = stream.read(width)
                require(len(raw_size) == width, "Truncated NPY header")
                header_length = struct.unpack("<H" if width == 2 else "<I", raw_size)[0]
                require(header_length <= 1024 * 1024, "NPY header is too large")
                header = ast.literal_eval(stream.read(header_length).decode("utf-8" if version == (3, 0) else "latin1"))
                dtype = np.dtype(header["descr"])
                require(not dtype.hasobject and dtype.fields is None and dtype.itemsize > 0,
                        f"Stage 2 integrity requires primitive non-object arrays: {context}")
                shape = header["shape"]
                require(isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape)
                        and type(header["fortran_order"]) is bool, f"Invalid NPY shape/order: {context}")
                h.update(member[:-4].encode("utf-8"))
                h.update(str(dtype).encode("ascii"))
                # np.ascontiguousarray promotes a scalar to a one-element array.
                h.update(json.dumps(list(shape or (1,)), sort_keys=True, separators=(",", ":")).encode())
                expected = math.prod(shape) * dtype.itemsize
                if header["fortran_order"]:
                    _hash_fortran_payload(h, stream, shape, dtype.itemsize, expected, context)
                    continue
                total = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    total += len(block)
                    h.update(block)
                require(total == expected, f"Truncated/inconsistent NPY data: {context}")
    return h.hexdigest()


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"Invalid integer {name}: {value!r}")
    return value


def normalized_origin(origin: str) -> str:
    require(isinstance(origin, str) and bool(origin), "Missing source-file origin")
    return origin.replace("\\", "/").casefold() if ("\\" in origin or PureWindowsPath(origin).drive) else origin


def verify_fingerprint(value: dict, key: str, omitted: tuple[str, ...] = ()) -> None:
    payload = {k: v for k, v in value.items() if k not in {key, *omitted}}
    require(digest(payload) == value.get(key), f"Manifest content does not match {key}")


def validate_candidates(candidates: list[dict], count: int) -> list[dict]:
    require(isinstance(candidates, list), "candidate_fragments must be a list")
    ids, ranks, result = set(), set(), []
    for original in candidates:
        require(isinstance(original, dict), "Candidate must be an object")
        c = dict(original)
        s = integer(c.get("start_index"), "start_index")
        length = integer(c.get("length_points"), "length_points", 1)
        e = integer(c.get("end_index"), "end_index")
        rank = integer(c.get("rank"), "rank", 1)
        require(s < count and length <= count and e == (s + length - 1) % count, "Inconsistent circular candidate span")
        require(isinstance(c.get("wraps"), bool) and c["wraps"] == (e < s), "Inconsistent candidate wraps flag")
        require(rank not in ranks, "Duplicate candidate rank")
        ranks.add(rank)
        require(np.isfinite(float(c.get("score", float("nan")))), "Candidate score must be finite")
        # Research exports have no candidate IDs. Hash their preserved evidence.
        c.setdefault("candidate_id", "derived_" + digest(original)[:24])
        require(isinstance(c["candidate_id"], str) and bool(c["candidate_id"]) and c["candidate_id"] not in ids, "Missing/duplicate candidate ID")
        ids.add(c["candidate_id"])
        # Only observable Stage 2 evidence is copied into the derived index.
        allowed = {"candidate_id", "rank", "start_index", "end_index", "wraps", "length_points", "length_fraction", "mean_seam_probability", "mean_uncertainty_probability", "endpoint_transition_probability", "score", "source", "hypothesis", "split_threshold", "start_xy_crop", "end_xy_crop", "start_xy_source", "end_xy_source"}
        result.append({k: v for k, v in c.items() if k in allowed})
    require(ranks == set(range(1, len(result) + 1)), "Candidate ranks must be contiguous from 1")
    return result


def circular_indices(candidate: dict, point_count: int) -> np.ndarray:
    validate_candidates([{**candidate, "rank": 1, "score": candidate.get("score", 0.0)}], point_count)
    return (int(candidate["start_index"]) + np.arange(int(candidate["length_points"]))) % point_count


def map_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    h = np.asarray(homography, dtype=np.float64)
    require(h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-12, "Invalid crop-to-source homography")
    projected = np.column_stack((points, np.ones(len(points)))) @ h.T
    require(np.all(np.abs(projected[:, 2]) > 1e-12), "Homography projects points to infinity")
    return projected[:, :2] / projected[:, 2:]

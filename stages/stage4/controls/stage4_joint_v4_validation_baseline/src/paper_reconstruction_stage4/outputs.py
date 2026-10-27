"""Versioned results with an atomic ready manifest and independent verification."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
import time
import numpy as np
from jsonschema import Draft202012Validator
from paper_reconstruction_contracts import map_points
from . import __version__
from .geometry import normalize, relative


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def implementation_identity():
    from paper_reconstruction_contracts import __file__ as sdk_file
    roots = (("stage4", Path(__file__).parent), ("contract_sdk", Path(sdk_file).parent))
    files = {}
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
                files[f"{label}/{path.relative_to(root).as_posix()}"] = file_hash(path)
    return {"sha256": digest(files), "files": files,
            "versions": {p: version(p) for p in ("numpy", "Pillow", "jsonschema", "scipy", "paper-reconstruction-contracts")},
            "python": platform.python_version()}


def request_identity(problem, config):
    implementation = implementation_identity()
    request = {"input_manifest_sha256": problem.manifest["manifest_sha256"], "configuration": config.to_dict(),
               "implementation": implementation}
    if "initial_result" in problem.input_summary:
        request["initial_result"] = problem.input_summary["initial_result"]
    return {"run_id": digest(request), **request}


@contextmanager
def output_session(output, request, resume=False):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        existing = read_result(output)
        if not resume or existing["run_id"] != request["run_id"]:
            raise ValueError("Output is immutable or belongs to a different run; choose a new directory")
        yield None
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty unpublished output directory; choose a fresh output path")
    work = output.with_name(output.name + ".work")
    work.mkdir(parents=True, exist_ok=True)
    prior = work / "request.json"
    if prior.exists():
        if not resume or json.loads(prior.read_text())["run_id"] != request["run_id"]:
            raise ValueError("Existing work identity differs or --resume was not supplied")
    lock = work / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Run lock exists: {lock}; inspect the owning process before removing a stale lock") from exc
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        atomic_json(prior, request)
        yield work
    except BaseException as exc:
        atomic_json(work / "failure.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                          "run_id": request["run_id"], "error_type": type(exc).__name__, "detail": str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def native_probes(piece):
    y, x = np.nonzero(piece.mask)
    indices = np.linspace(0, len(x) - 1, min(65, len(x))).astype(int)
    return np.column_stack((x[indices], y[indices])).astype(float).tolist()


def serialize_layout(problem, state, index, feasibility, refinement):
    edges = {e.id: e for e in problem.edges}
    joins = []
    for eid in state.selected:
        edge = edges[eid]
        oid = state.assignments.get(eid)
        occurrence = next((o for o in edge.occurrences if o["occurrence_id"] == oid), None)
        joins.append({"geometry_id": eid, "piece_a": edge.a, "piece_b": edge.b,
                      "occurrence_id": oid, "support_assignment_checked": occurrence is not None,
                      "tree_join": eid in state.tree, "source_transform_b_to_a": edge.transform.tolist(),
                      "final_transform_b_to_a": relative(state.poses, edge.a, edge.b).tolist(),
                      "source_weight": edge.weight, "score_method": edge.raw.get("score_method", "fixture"),
                      "score_variant": edge.raw.get("score_variant", "fixture"),
                      "support": None if occurrence is None else {k: occurrence[k] for k in ("query_key", "target_key", "pose")}})
        if eid in state.active_support:
            joins[-1]["active_support"] = state.active_support[eid]
    pieces = [{"piece_id": pid, "observation_key": problem.pieces[pid].observation, "component_id": state.groups[pid],
               "native_to_component": normalize(state.poses[pid]).tolist(), "size_hw": list(problem.pieces[pid].shape),
               "native_probes": native_probes(problem.pieces[pid]),
               "status": "isolated" if sum(g == state.groups[pid] for g in state.groups.values()) == 1 else "component_relative"}
              for pid in sorted(state.poses)]
    layout = {"layout_index": index, "pieces": pieces, "components": state.components(), "selected_joins": joins,
              "objective_score": float(state.score), "confidence_calibrated": False,
              "all_pieces_connected": len(state.components()) == 1,
              "feasibility": feasibility, "refinement": refinement,
              "unresolved_piece_ids": [p["piece_id"] for p in pieces if p["status"] == "isolated"]}
    layout["layout_sha256"] = digest({k: v for k, v in layout.items() if k not in {"feasibility", "refinement"}})
    return layout


def build_result(problem, config, search_result, request, peak_memory_bytes=None):
    layouts = [serialize_layout(problem, state, i, search_result.diagnostics["final_feasibility"][i],
                                 search_result.diagnostics["refinement"][i]) for i, state in enumerate(search_result.states)]
    best = layouts[0]
    termination = search_result.termination
    status = ("interrupted" if termination == "interrupted" else "budget_exhausted" if termination in {"proposal_budget", "time_budget"}
              else "complete_candidate" if best["all_pieces_connected"] else "partial_candidate")
    value = {"schema_version": "stage4.assembly-result.v2" if config.solver == "joint_beam_v3" else "stage4.assembly-result.v1", "solver_version": __version__, "run_id": request["run_id"],
             "solver": config.solver, "status": status, "termination": termination, "configuration": config.to_dict(),
             "input": {"manifest_sha256": problem.manifest["manifest_sha256"], "puzzle_id": problem.manifest["puzzle_id"],
                       "schema_version": problem.manifest["schema_version"], "counts": problem.manifest["counts"],
                       "working_pool_pairs": problem.pool["pairs"], "load": problem.input_summary},
             "implementation": request["implementation"], "layouts": layouts,
             "search": search_result.diagnostics, "rejections": search_result.rejections,
             "resources": {"peak_process_memory_bytes": peak_memory_bytes, "peak_memory_scope": "process_peak_through_solver"}, "confidence_calibrated": False,
             "evaluation_truth_opened": False, "demonstrated_reliable_piece_capacity": None,
             "legacy_control_support_enforced": config.solver != "legacy_forest_control_v1"}
    if config.solver == "joint_beam_v3":
        from .active_support import verify_result_support
        value["search"]["independent_support_verification"] = verify_result_support(problem, value, config)
    validate_result(value)
    return value


def validate_result(value):
    schema = json.loads((Path(__file__).parent / "schemas" / "assembly-result.schema.json").read_text())
    Draft202012Validator(schema).validate(value)
    expected = value["input"]["counts"]["pieces"]
    inventory = None
    for layout in value["layouts"]:
        pieces = layout["pieces"]
        ids = [p["piece_id"] for p in pieces]
        if len(ids) != expected or len(set(ids)) != len(ids):
            raise ValueError("Output physical inventory differs from input")
        if inventory is not None and set(ids) != inventory:
            raise ValueError("Alternative layout inventory differs")
        inventory = set(ids)
        components = layout["components"]
        if sorted(p for component in components for p in component) != sorted(ids):
            raise ValueError("Output components do not partition pieces")
        owners = {p: min(component) for component in components for p in component}
        for piece in pieces:
            normalize(piece["native_to_component"])
            if piece["component_id"] != owners[piece["piece_id"]]:
                raise ValueError("Nondeterministic component gauge")
            if piece["piece_id"] == piece["component_id"] and not np.allclose(piece["native_to_component"], np.eye(3), atol=1e-8):
                raise ValueError("Component root must be identity")
        pair_ids = set()
        for join in layout["selected_joins"]:
            a, b = join["piece_a"], join["piece_b"]
            if a not in inventory or b not in inventory or not a < b or (a, b) in pair_ids or owners[a] != owners[b]:
                raise ValueError("Invalid selected physical pair")
            pair_ids.add((a, b))
            if value["solver"] != "legacy_forest_control_v1" and (not join["support_assignment_checked"] or not join["occurrence_id"]):
                raise ValueError("Improved solver omitted occurrence assignment")
            if join.get("active_support"):
                from .active_support import validate_active_record
                from .config import SolverConfig
                config = SolverConfig.from_dict(value["configuration"])
                if value["schema_version"] != "stage4.assembly-result.v2" or config.support_policy != "active-subarc.v1":
                    raise ValueError("Active support omitted its output/policy version")
                validate_active_record(join["active_support"], config)
                if join["active_support"]["occurrence_id"] != join["occurrence_id"]:
                    raise ValueError("Active support occurrence binding differs")
        expected_hash = digest({k: v for k, v in layout.items() if k not in {"feasibility", "refinement", "display_packing", "layout_sha256"}})
        if expected_hash != layout["layout_sha256"]:
            raise ValueError("Layout hash mismatch")


def publish(output, work, value, problem, states, resource_probe=None):
    from .preview import render
    publication_start = time.perf_counter()
    staging = Path(work) / "publication"
    staging.mkdir(exist_ok=True)
    render_start = time.perf_counter()
    packing = render(problem, states[0], staging / "preview.png")
    render_seconds = time.perf_counter()-render_start
    if resource_probe is not None:
        value["resources"]["peak_process_memory_bytes"] = resource_probe()
        value["resources"]["peak_memory_scope"] = "process_peak_through_preview"
    value["layouts"][0]["display_packing"] = packing
    validation_start = time.perf_counter()
    validate_result(value)
    validation_seconds = time.perf_counter()-validation_start
    value["resources"].update(render_seconds=render_seconds, preseal_validation_seconds=validation_seconds)
    seal_start = time.perf_counter()
    atomic_json(staging / "result.json", value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name in ("result.json", "preview.png"):
        target = staging / name
        receipts.append({"path": name, "size": target.stat().st_size, "sha256": file_hash(target)})
    manifest = {"schema_version": "stage4.result-manifest.v1", "run_id": value["run_id"],
                "created_utc": datetime.now(timezone.utc).isoformat(), "files": receipts}
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(staging / "manifest.json", manifest)
    if output.exists():
        output.rmdir()  # Only succeeds for an empty, explicitly selected directory.
    os.replace(staging, output)
    atomic_json(Path(work) / "publication_timing.json", {"run_id": value["run_id"], "render_seconds": render_seconds,
        "preseal_validation_seconds": validation_seconds, "serialization_receipts_and_seal_seconds": time.perf_counter()-seal_start,
        "publication_wall_seconds": time.perf_counter()-publication_start,
        "scope": "Sidecar timing through atomic publication; wall includes rendering and validation, do not sum nested rows"})


def read_result(path):
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "stage4.result-manifest.v1" or manifest.get("manifest_sha256") != digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Invalid result manifest")
    names = set()
    for receipt in manifest["files"]:
        name = receipt["path"]
        if name not in {"result.json", "preview.png"} or name in names:
            raise ValueError("Unexpected or duplicate result receipt")
        names.add(name)
        target = (root / name).resolve()
        if not target.is_relative_to(root) or target.stat().st_size != receipt["size"] or file_hash(target) != receipt["sha256"]:
            raise ValueError("Result artifact checksum/path mismatch")
    if names != {"result.json", "preview.png"}:
        raise ValueError("Incomplete result publication")
    value = json.loads((root / "result.json").read_text())
    validate_result(value)
    if value["run_id"] != manifest["run_id"]:
        raise ValueError("Manifest/result run identity mismatch")
    return value

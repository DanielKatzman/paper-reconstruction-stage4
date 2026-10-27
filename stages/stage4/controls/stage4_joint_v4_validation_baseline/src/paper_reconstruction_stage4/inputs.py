"""The only upstream dependency is the independent released contract reader."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
from paper_reconstruction_contracts import ContractError, load_bundle, relative_file
from .model import Edge, Piece, Problem


def load_problem(path, config):
    start = time.perf_counter()
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    if not (root / "bundle.json").is_file():
        raise ContractError("Input is not ready: bundle.json is absent")
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8-sig"))
    receipts = [*manifest.get("assets", {}).values(), manifest.get("archive", {}), manifest.get("working_pool", {})]
    total_bytes = sum(int(r.get("size", 0)) for r in receipts)
    if total_bytes > config.max_input_bytes:
        raise ContractError("Declared input byte budget exceeded before full validation")
    data = load_bundle(root, full=True)
    validation_seconds = time.perf_counter() - start
    wanted = {e["id"] for e in data.edges}
    groups = {}
    count = 0
    def archive():
        nonlocal count
        for group in data.hypotheses():
            if group["hypothesis_id"] in wanted:
                count += len(group["occurrences"])
                if count > config.max_working_occurrences:
                    raise ContractError("Working occurrence indexing budget exceeded")
                groups[group["hypothesis_id"]] = group
            yield group
    additions, expansion = [], []
    if config.evidence_policy == "working_plus_diverse_archive.v1":
        from .evidence import diverse_additions
        anchors = [Edge(e["id"], e["a"], e["b"], np.asarray(e["transform"], float), float(e["weight"]), [], dict(e)) for e in data.edges]
        additions, expansion = diverse_additions(archive(), anchors, data.manifest, config)
        count += sum(len(e.occurrences) for e in additions)
        if count > config.max_working_occurrences:
            raise ContractError("Expanded occurrence indexing budget exceeded")
    else:
        for _ in archive():
            pass
    pieces = {}
    for piece_id, entry in sorted(data.entries.items()):
        with Image.open(entry["mask_path"]) as mask_image:
            mask = np.asarray(mask_image.convert("L")) >= 128
        if not mask.any():
            raise ContractError(f"Empty physical mask: {piece_id}")
        contour_path = relative_file(data.root, data.manifest["pieces"][piece_id]["contour"])
        with np.load(contour_path, allow_pickle=False) as archive:
            contour = {k: archive[k].copy() for k in archive.files}
        pieces[piece_id] = Piece(piece_id, entry["observation_key"], Path(entry["image_path"]), mask, contour)
    edges = [Edge(e["id"], e["a"], e["b"], np.asarray(e["transform"], float), float(e["weight"]),
                  groups[e["id"]]["occurrences"], dict(e)) for e in data.edges]
    edges.extend(additions)
    edges.sort(key=lambda e: (-e.weight, e.a, e.b, e.id))
    return Problem(pieces, edges, {c["candidate_key"]: c for c in data.manifest["candidates"]},
                   data.manifest, data.pool, data.root,
                   {"declared_bytes": total_bytes, "validation_seconds": validation_seconds,
                    "load_and_index_seconds": time.perf_counter() - start, "indexed_occurrences": count,
                    "archive_passes": 2, "full_archive_indexed": False,
                    "consumer_evidence_policy": config.evidence_policy, "consumer_edges": len(edges),
                    "consumer_archive_additions": expansion})

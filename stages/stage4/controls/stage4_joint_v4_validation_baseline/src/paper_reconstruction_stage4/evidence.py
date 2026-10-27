"""Bounded consumer-side archive diversity; never rewrites upstream evidence."""
from collections import defaultdict
import numpy as np
from paper_reconstruction_contracts import map_points
from .geometry import probes
from .model import Edge


def occurrence_score(occurrence):
    for method in ("mlp_start", "alpha_fused", "cnn_budget1", "reference"):
        value = occurrence.get("methods", {}).get(method, {}).get("weight")
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value), method
    return float(occurrence["pose"]["alignment"].get("geometry_score", 0.)), "geometry_only"


def archive_edge(group):
    occurrences = [o for o in group["occurrences"] if o["variant"] == "source_control"] or group["occurrences"]
    occurrence = min(occurrences, key=lambda o: (-occurrence_score(o)[0], o["occurrence_id"]))
    score, method = occurrence_score(occurrence)
    return Edge(group["hypothesis_id"], group["piece_a"], group["piece_b"], np.asarray(group["transform_b_to_a"], float), score,
                group["occurrences"], {"score_method": method, "score_variant": occurrence["variant"],
                 "pose": occurrence["pose"], "consumer_origin": "verified_archive", "representative_occurrence": occurrence["occurrence_id"]})


def diverse_additions(groups, working, manifest, config):
    """One archive pass, bounded candidates/pair; exact input group IDs retained."""
    present = {e.id for e in working}
    by_pair = defaultdict(dict)
    seen = defaultdict(int)
    for group in groups:
        if group["hypothesis_id"] not in present:
            edge = archive_edge(group)
            pair = (edge.a, edge.b)
            seen[pair] += 1
            shape = manifest["pieces"][edge.b]["size_hw"]
            center = np.array([shape[1]/2, shape[0]/2])
            try:
                xy = map_points([center, center+[1., 0.]], edge.transform)
            except ValueError:
                continue
            vector = xy[1]-xy[0]
            diagonal = max(float(np.linalg.norm(shape)), 1.)
            cell = (*np.floor(xy[0]/(.05*diagonal)).astype(int),
                    int(np.floor(np.arctan2(vector[1], vector[0])/np.deg2rad(10.))),
                    int(np.floor(np.log(max(np.linalg.norm(vector), 1e-8))/.1)))
            prior = by_pair[pair].get(cell)
            if prior is None or (edge.weight, edge.id) > (prior.weight, prior.id):
                by_pair[pair][cell] = edge
            if len(by_pair[pair]) > config.archive_candidates_per_pair:
                worst = min(by_pair[pair], key=lambda key: (by_pair[pair][key].weight, by_pair[pair][key].id))
                del by_pair[pair][worst]
    anchors = defaultdict(list)
    for edge in working:
        anchors[edge.a, edge.b].append(edge)
    additions, audit = [], []
    for pair, cells in sorted(by_pair.items()):
        values = list(cells.values())
        shape = manifest["pieces"][pair[1]]["size_hw"]
        points = probes(shape, 3)
        diagonal = max(float(np.linalg.norm(shape)), 1.)
        best = max((e.weight for e in [*values, *anchors[pair]]), default=0.)
        eligible = [e for e in values if e.weight >= best-config.archive_score_margin]
        # Shortlisted representatives of distinct pose basins, rather than
        # filling the remaining slots with nearly identical high-score fits.
        mapped = {}
        for edge in [*anchors[pair], *eligible]:
            try:
                mapped[edge.id] = map_points(points, edge.transform)/diagonal
            except ValueError:
                continue
        eligible = [e for e in eligible if e.id in mapped]
        covered = [mapped[e.id] for e in anchors[pair] if e.id in mapped]
        selected = []
        for _ in range(min(config.archive_extra_per_pair, len(eligible))):
            def priority(edge):
                distance = min((float(np.sqrt(np.mean((mapped[edge.id]-xy)**2))) for xy in covered), default=1.)
                # Saturation keeps extreme, low-quality transforms from taking
                # every slot solely through an unbounded displacement.
                return (min(distance, .5) * (max(edge.weight, 0.)+.1), edge.weight, edge.id)
            edge = max(eligible, key=priority)
            eligible.remove(edge)
            selected.append(edge)
            covered.append(mapped[edge.id])
        additions.extend(selected)
        audit.append({"pieces": list(pair), "archive_nonworking_geometries": seen[pair], "bounded_pose_cells": len(cells),
                      "added_geometry_ids": [e.id for e in selected], "score_floor": best-config.archive_score_margin})
    return additions, audit

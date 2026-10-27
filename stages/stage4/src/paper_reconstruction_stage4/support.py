"""Map occurrence-local windows/fractions to native circular physical contours."""
from dataclasses import dataclass
from collections import OrderedDict
import json
import numpy as np
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import map_points
from .geometry import relative


@dataclass
class Arc:
    piece: str
    intervals: tuple[tuple[float, float], ...]
    points: np.ndarray
    length: float


def native_arc(problem, candidate_key, span, fraction):
    descriptor = problem.candidates[candidate_key]
    piece_id = descriptor["piece_id"]
    xy = problem.pieces[piece_id].contour["points_xy"]
    candidate = descriptor["candidate"]
    n = len(xy)
    start, end = candidate["start_index"], candidate["end_index"]
    if type(start) is not int or type(end) is not int or not 0 <= start < n or not 0 <= end < n:
        raise ValueError("Invalid candidate contour indices")
    expected_length = (end - start) % n + 1
    if candidate.get("length_points", expected_length) != expected_length or bool(candidate.get("wraps", end < start)) != (end < start):
        raise ValueError("Inconsistent native candidate length/wrapping")
    if bool(candidate.get("wraps", end < start)):
        indices = np.r_[np.arange(start, n), np.arange(0, end + 1)]
    else:
        if end < start:
            raise ValueError("Contradictory candidate wrapping")
        indices = np.arange(start, end + 1)
    points = xy[indices]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if len(arc) < 2 or arc[-1] <= 1e-8:
        raise ValueError("Empty candidate support")
    lo, hi = (0., float(arc[-1])) if span is None else (float(span["start_px"]), float(span["end_px"]))
    if not 0 <= lo < hi <= arc[-1] + 1e-5:
        raise ValueError("Support span escapes native candidate")
    f0, f1 = map(float, fraction)
    if not 0 <= f0 < f1 <= 1:
        raise ValueError("Invalid support fractions")
    lo, hi = lo + f0 * (hi - lo), lo + f1 * (hi - lo)
    sample = np.linspace(lo, hi, 65)
    samples = np.column_stack([np.interp(sample, arc, points[:, k]) for k in range(2)])
    circumference_edges = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    circumference = float(circumference_edges.sum())
    origin = float(circumference_edges[:start].sum())
    begin = (origin + lo) % circumference
    length = hi - lo
    if length >= circumference - 1e-6:
        intervals = ((0., circumference),)
    elif begin + length <= circumference:
        intervals = ((begin, begin + length),)
    else:
        intervals = ((begin, circumference), (0., begin + length - circumference))
    return Arc(piece_id, intervals, samples, length)


def occurrence_arcs(problem, occurrence):
    if 'native_bridge' in occurrence:
        from .native_bridge import record_arcs
        return record_arcs(problem, occurrence['native_bridge'])
    pose = occurrence["pose"]
    alignment = pose["alignment"]
    return (native_arc(problem, occurrence["query_key"], pose["span_a"], alignment.get("query_arc_fraction", (0., 1.))),
            native_arc(problem, occurrence["target_key"], pose["span_b"], alignment.get("target_arc_fraction", (0., 1.))))


def overlaps(first, second, config):
    if first.piece != second.piece:
        return False
    overlap = sum(max(0., min(b, d) - max(a, c)) for a, b in first.intervals for c, d in second.intervals)
    allowed = max(config.support_endpoint_tolerance_px, config.support_overlap_fraction * min(first.length, second.length))
    return overlap > allowed + 1e-8


def contact_error(arcs, poses):
    first, second = arcs
    h = relative(poses, first.piece, second.piece)
    mapped = map_points(second.points, h)
    forward = cKDTree(first.points).query(mapped)[0]
    reverse = cKDTree(second.points).query(map_points(first.points, np.linalg.inv(h)))[0]
    return max(float(np.quantile(forward, .95)), float(np.quantile(reverse, .95)))


class SupportIndex:
    def __init__(self, problem, config):
        self.problem, self.config = problem, config
        self.arcs = {}
        self.invalid = {}
        self.contact_cache = OrderedDict()
        self.duplicate_options_skipped = 0
        shared = {}
        for edge in problem.edges:
            for occurrence in edge.occurrences:
                oid = occurrence["occurrence_id"]
                try:
                    pose = occurrence["pose"]
                    signature = json.dumps([occurrence["query_key"], occurrence["target_key"], pose["span_a"], pose["span_b"],
                                            pose["alignment"].get("query_arc_fraction", (0., 1.)),
                                            pose["alignment"].get("target_arc_fraction", (0., 1.))], sort_keys=True)
                    if 'native_bridge' in occurrence:
                        from .native_bridge import validate_record
                        validate_record(occurrence['native_bridge'])
                        signature=('native_bridge',occurrence['native_bridge']['bridge_id'])
                    if signature not in shared:
                        shared[signature] = occurrence_arcs(problem, occurrence)
                    self.arcs[oid] = shared[signature]
                except (ValueError, KeyError, IndexError) as exc:
                    self.invalid[oid] = str(exc)

    def assign(self, edge_ids, poses, edges):
        options = {}
        for eid in edge_ids:
            viable, seen_support = [], set()
            for occurrence in sorted(edges[eid].occurrences, key=lambda o: o["occurrence_id"]):
                oid = occurrence["occurrence_id"]
                if oid not in self.arcs:
                    continue
                arcs = self.arcs[oid]
                signature = tuple((arc.piece, arc.intervals) for arc in arcs)
                if signature in seen_support:
                    self.duplicate_options_skipped = getattr(self, "duplicate_options_skipped", 0) + 1
                    continue
                seen_support.add(signature)
                cache = getattr(self, "contact_cache", None)
                key = (signature, relative(poses, arcs[0].piece, arcs[-1].piece).tobytes()) if cache is not None else None
                error = cache.get(key) if cache is not None else None
                if error is None:
                    error = contact_error(arcs, poses)
                    if cache is not None:
                        if len(cache) >= self.config.maximum_cache_entries:
                            cache.popitem(last=False)
                        cache[key] = error
                if error <= self.config.contact_tolerance_px:
                    viable.append(oid)
            options[eid] = sorted(set(viable))
            if not viable:
                return None, "no_valid_contact_support", 0
        ordered = sorted(edge_ids, key=lambda eid: (len(options[eid]), eid))
        result, used = {}, []
        nodes = 0
        exhausted = False

        def visit(index):
            nonlocal nodes, exhausted
            if index == len(ordered):
                return dict(result)
            eid = ordered[index]
            for oid in options[eid]:
                if nodes >= self.config.max_assignment_nodes:
                    exhausted = True
                    return None
                nodes += 1
                arcs = self.arcs[oid]
                if any(overlaps(a, b, self.config) for a in arcs for b in used):
                    continue
                result[eid] = oid
                used.extend(arcs)
                found = visit(index + 1)
                if found is not None:
                    return found
                del used[-len(arcs):]
                result.pop(eid)
            return None

        found = visit(0)
        return found, ("assigned" if found is not None else "support_assignment_budget" if exhausted else "support_conflict"), nodes

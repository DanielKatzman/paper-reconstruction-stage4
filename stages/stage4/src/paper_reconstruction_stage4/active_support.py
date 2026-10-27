"""Consumer-derived contiguous subarcs; incoming occurrences stay immutable."""
from collections import Counter, OrderedDict
import copy
import numpy as np
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import map_points
from .geometry import relative
from .outputs import digest
from .support import Arc, SupportIndex, contact_error, overlaps


def slice_arc(problem, arc, first=0., last=1., count=65):
    """Slice by native arclength, preserving the order of circular intervals."""
    if not 0 <= first < last <= 1:
        raise ValueError("Invalid native subarc fraction")
    lo, hi = first*arc.length, last*arc.length
    intervals, offset = [], 0.
    for a, b in arc.intervals:
        start, end = max(lo-offset, 0.), min(hi-offset, b-a)
        if end > start:
            intervals.append((a+start, a+end))
        offset += b-a
    if not intervals:
        raise ValueError("Empty native subarc")
    xy = problem.pieces[arc.piece].contour["points_xy"]
    curve = np.vstack([xy, xy[0]])
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    length = sum(b-a for a, b in intervals)
    distances = np.linspace(0., length, count)
    coordinates, offset = np.zeros(count), 0.
    for index, (a, b) in enumerate(intervals):
        take = (distances >= offset) & ((distances <= offset+b-a) if index == len(intervals)-1 else (distances < offset+b-a))
        coordinates[take] = a+distances[take]-offset
        offset += b-a
    points = np.column_stack([np.interp(coordinates, cumulative, curve[:, k]) for k in range(2)])
    return Arc(arc.piece, tuple(intervals), points, float(length))


def arc_record(arc):
    return {"piece": arc.piece, "intervals": [list(v) for v in arc.intervals], "length": arc.length,
            "points": arc.points.tolist()}


def record_arc(value):
    return Arc(value["piece"], tuple(tuple(v) for v in value["intervals"]), np.asarray(value["points"], float), value["length"])


class ActiveSupportIndex(SupportIndex):
    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.option_cache = OrderedDict()
        self.normal_index = {p: cKDTree(v.contour["points_xy"]) for p, v in problem.pieces.items()}
        self.stats, self.conflicts = Counter(), []
        self.last_active, self.last_assignment = {}, None

    def arcs_for(self, state, eid):
        active = state.active_support.get(eid)
        return tuple(record_arc(a) for a in active["arcs"]) if active else self.arcs[state.assignments[eid]]

    def bind(self, state):
        if state.assignments != self.last_assignment:
            raise ValueError("Active support assignment is not bound to this state")
        state.active_support = copy.deepcopy(self.last_active)

    def derived_options(self, oid, poses):
        arcs = self.arcs[oid]
        key = (oid, relative(poses, arcs[1].piece, arcs[0].piece).tobytes())
        if key in self.option_cache:
            self.stats["option_cache_hits"] += 1
            return self.option_cache[key]
        self.stats["option_extractions"] += 1
        first, second = (slice_arc(self.problem, a, count=257) for a in arcs)
        h = relative(poses, second.piece, first.piece)
        mapped = map_points(first.points, h)
        distances, nearest = cKDTree(second.points).query(mapped)
        ni = self.normal_index[first.piece].query(first.points)[1]
        nj = self.normal_index[second.piece].query(second.points[nearest])[1]
        normals_a = self.problem.pieces[first.piece].contour["inward_normal_xy"][ni]
        normals_b = self.problem.pieces[second.piece].contour["inward_normal_xy"][nj]
        mapped_normals = map_points(first.points+normals_a, h)-mapped
        cosine = np.sum(mapped_normals*normals_b, axis=1)/np.maximum(np.linalg.norm(mapped_normals, axis=1)*np.linalg.norm(normals_b, axis=1), 1e-12)
        # Mutual contact and opposing observed interiors, not arbitrary shortening.
        inverse_distances = cKDTree(first.points).query(map_points(second.points, np.linalg.inv(h)))[0]
        tangent = np.gradient(second.points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        projected = np.sum((mapped-second.points[nearest])*tangent[nearest], axis=1)
        endpoint_supported = ~(((nearest == 0) & (projected < -.5)) | ((nearest == 256) & (projected > .5)))
        good = ((distances <= self.config.active_distance_px) & (inverse_distances[nearest] <= self.config.active_distance_px)
                & (cosine < -.25) & endpoint_supported)
        changes = np.diff(np.r_[False, good, False].astype(int))
        options = []
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
            if end-start < 3:
                continue
            indices = nearest[start:end]
            steps = np.diff(indices)
            monotone = max(np.mean(steps >= 0), np.mean(steps <= 0))
            bounds = ((start/256., (end-1)/256.), (float(indices.min()/256.), float(indices.max()/256.)))
            if monotone < .85 or any(hi <= lo for lo, hi in bounds):
                continue
            trimmed = tuple(slice_arc(self.problem, raw, *bound) for raw, bound in zip(arcs, bounds, strict=True))
            coverage = [a.length/raw.length for a, raw in zip(trimmed, arcs, strict=True)]
            if min(coverage) < self.config.active_minimum_fraction or min(a.length for a in trimmed) < self.config.active_minimum_length_px:
                self.stats["insufficient_active_coverage"] += 1
                continue
            if min(np.linalg.norm(a.points[-1]-a.points[0]) for a in trimmed) < self.config.active_minimum_length_px*.5:
                continue
            if contact_error(trimmed, poses) > self.config.contact_tolerance_px:
                continue
            record = {"policy": "active-subarc.v1", "occurrence_id": oid, "arcs": [arc_record(a) for a in trimmed],
                      "raw_arcs": [arc_record(a) for a in arcs],
                      "raw_lengths": [a.length for a in arcs], "coverage": coverage,
                      "normal_cosine_median": float(np.median(cosine[start:end])), "monotone_fraction": float(monotone)}
            record["support_id"] = digest(record)
            options.append((trimmed, record))
        options.sort(key=lambda item: (-min(item[1]["coverage"]), item[1]["support_id"]))
        if len(options) > self.config.active_max_options:
            self.stats["active_option_truncations"] += 1
        options = options[:self.config.active_max_options]
        if len(self.option_cache) >= self.config.maximum_cache_entries:
            self.option_cache.popitem(last=False)
        self.option_cache[key] = options
        return options

    def assign(self, edge_ids, poses, edges):
        self.last_active, self.last_assignment = {}, None
        assigned, reason, raw_nodes = super().assign(edge_ids, poses, edges)
        if assigned is not None:
            self.last_assignment = assigned
            return assigned, reason, raw_nodes
        self.stats["raw_"+reason] += 1
        if reason == "support_assignment_budget":
            return None, reason, raw_nodes
        choices = {}
        for eid in edge_ids:
            options, seen = [], set()
            for occurrence in sorted(edges[eid].occurrences, key=lambda o: o["occurrence_id"]):
                oid = occurrence["occurrence_id"]
                if oid not in self.arcs:
                    continue
                raw = self.arcs[oid]
                signature = tuple((a.piece, a.intervals) for a in raw)
                if signature in seen:
                    continue
                seen.add(signature)
                if contact_error(raw, poses) <= self.config.contact_tolerance_px:
                    options.append((oid, raw, None))
                options.extend((oid, arcs, record) for arcs, record in self.derived_options(oid, poses))
            choices[eid] = options
            if not options:
                return None, "no_valid_contact_support", raw_nodes
        order = sorted(edge_ids, key=lambda e: (len(choices[e]), e))
        selected, active, used = {}, {}, []
        nodes, exhausted = 0, False
        def visit(index):
            nonlocal nodes, exhausted
            if index == len(order):
                return dict(selected), copy.deepcopy(active)
            eid = order[index]
            for oid, arcs, record in choices[eid]:
                if raw_nodes+nodes >= self.config.max_assignment_nodes:
                    exhausted = True
                    return None
                nodes += 1
                if any(overlaps(a, b, self.config) for a in arcs for b in used):
                    continue
                selected[eid] = oid
                if record is not None:
                    active[eid] = record
                used.extend(arcs)
                result = visit(index+1)
                if result is not None:
                    return result
                del used[-2:]
                selected.pop(eid, None); active.pop(eid, None)
            return None
        found = visit(0)
        self.stats["active_assignment_nodes"] += nodes
        if len(self.conflicts) < 64:
            interval_rows = [{"geometry_id": eid, "pair": [edges[eid].a, edges[eid].b],
                              "options": [{"occurrence_id": oid, "active": record is not None,
                                           "intervals": [{"piece": a.piece, "intervals": a.intervals} for a in arcs]} for oid, arcs, record in values[:8]],
                              "options_truncated_in_report": len(values) > 8} for eid, values in choices.items()]
            self.conflicts.append({"raw_reason": reason, "active_resolution": found is not None, "joins": interval_rows})
        if found is None:
            return None, "support_assignment_budget" if exhausted else "support_conflict", raw_nodes+nodes
        self.last_assignment, self.last_active = found
        self.stats["resolved_with_active_support"] += 1
        return self.last_assignment, "assigned_active", raw_nodes+nodes

    def report(self):
        return {"policy": self.config.support_policy, "counts": dict(self.stats), "conflict_examples": self.conflicts,
                "conflict_examples_limit": 64, "raw_occurrences_preserved": True,
                "minimum_native_length_px": self.config.active_minimum_length_px,
                "minimum_raw_coverage_fraction": self.config.active_minimum_fraction}


def validate_active_record(record, config, raw=None):
    if record.get("policy") != "active-subarc.v1" or record.get("support_id") != digest({k: v for k, v in record.items() if k != "support_id"}):
        raise ValueError("Invalid active support identity/policy")
    if len(record["arcs"]) != 2 or len(record["raw_arcs"]) != 2:
        raise ValueError("Active support must have two physical arcs")
    if raw is None:
        raw = tuple(record_arc(a) for a in record["raw_arcs"])
    for index, (value, original) in enumerate(zip(record["arcs"], raw, strict=True)):
        declared_raw = record_arc(record["raw_arcs"][index])
        if declared_raw.piece != original.piece or declared_raw.intervals != original.intervals or abs(declared_raw.length-original.length) > 1e-7 or abs(record["raw_lengths"][index]-original.length) > 1e-7:
            raise ValueError("Active support raw provenance differs")
        arc = record_arc(value)
        if arc.piece != original.piece or not np.isfinite(arc.points).all() or arc.points.shape != (65, 2):
            raise ValueError("Invalid active native geometry")
        if any(not np.isfinite([a,b]).all() or a >= b or sum(max(0., min(b,d)-max(a,c)) for c,d in original.intervals) < b-a-1e-7 for a,b in arc.intervals):
            raise ValueError("Active support escapes raw occurrence")
        length = sum(b-a for a,b in arc.intervals)
        if abs(length-arc.length) > 1e-6 or length < config.active_minimum_length_px or length/original.length < config.active_minimum_fraction-1e-8:
            raise ValueError("Active support has insufficient length/coverage")
        if abs(record["coverage"][index]-length/original.length) > 1e-8:
            raise ValueError("Active support coverage mismatch")
        if any(min(b,d)-max(a,c) > 1e-8 for i,(a,b) in enumerate(arc.intervals) for c,d in arc.intervals[i+1:]):
            raise ValueError("Duplicated active native intervals")


def verify_result_support(problem, result, config):
    """Independently check published intervals against the actual input contours."""
    from .support import occurrence_arcs
    from .native_bridge import install_result_bridges, check_arcs
    install_result_bridges(problem,result,config)
    edges = {e.id: e for e in problem.edges}
    checked = 0
    for layout in result["layouts"]:
        from .native_bridge_v2 import verify_layout
        verify_layout(problem,layout,config)
        from .native_bridge_v3 import verify_layout as verify_page_boundary_layout
        verify_page_boundary_layout(problem,layout,config)
        poses = {p["piece_id"]: np.asarray(p["native_to_component"], float) for p in layout["pieces"]}
        used = []
        for join in layout["selected_joins"]:
            edge = edges[join["geometry_id"]]
            occurrence = next(o for o in edge.occurrences if o["occurrence_id"] == join["occurrence_id"])
            raw = occurrence_arcs(problem, occurrence)
            derived=edge.raw.get('native_bridge')
            if derived:
                if join.get('native_bridge')!=derived or join.get('active_support'):
                    raise ValueError('Derived native evidence is missing, changed or trimmed')
                if not np.allclose(relative(poses,edge.a,edge.b),edge.transform,atol=1e-7,rtol=0.):
                    raise ValueError('Derived bridge disagrees with final relative pose')
                check_arcs(problem,raw,poses,config,used)
            record = join.get("active_support")
            if record:
                validate_active_record(record, config, raw)
                if record["occurrence_id"] != occurrence["occurrence_id"]:
                    raise ValueError("Active support occurrence mismatch")
                arcs = tuple(record_arc(a) for a in record["arcs"])
                for a in arcs:
                    expected = slice_arc(problem, a)
                    if not np.allclose(expected.points, a.points, atol=1e-7, rtol=0.):
                        raise ValueError("Published active points differ from native contour")
                first, second = arcs
                mapped = map_points(first.points, relative(poses, second.piece, first.piece))
                closest = cKDTree(second.points).query(mapped)[1]
                ni = cKDTree(problem.pieces[first.piece].contour["points_xy"]).query(first.points)[1]
                nj = cKDTree(problem.pieces[second.piece].contour["points_xy"]).query(second.points[closest])[1]
                normal_a = problem.pieces[first.piece].contour["inward_normal_xy"][ni]
                normal_b = problem.pieces[second.piece].contour["inward_normal_xy"][nj]
                transformed = map_points(first.points+normal_a, relative(poses, second.piece, first.piece))-mapped
                cosine = np.sum(transformed*normal_b, axis=1)/np.maximum(np.linalg.norm(transformed,axis=1)*np.linalg.norm(normal_b,axis=1),1e-12)
                if float(np.median(cosine)) >= -.25:
                    raise ValueError("Published active contact has incompatible observed interiors")
                checked += 1
            else:
                arcs = raw
            if contact_error(arcs, poses) > config.contact_tolerance_px+1e-7 or any(overlaps(a,b,config) for a in arcs for b in used):
                raise ValueError("Published contact distance/interval exclusivity failed")
            used.extend(arcs)
    return {"layouts_checked": len(result["layouts"]), "active_joins_checked": checked, "valid": True}

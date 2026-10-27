"""Port of the archived control, preserving join-count-first selection.

Origin: Stage 3 research_archive/v0.9.31/pose_solver.py. Candidate generation
is intentionally omitted; edges come from the independent v2 contract.
See docs/LEGACY_CONTROL.md for parity scope and added operational limits.
"""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter
from ..geometry import warp_check
from ..model import State as ResultState
from .common import Budget, SearchResult


@dataclass
class LegacyState:
    poses: dict
    groups: dict
    edges: tuple = ()
    score: float = 0.


def map_points(points, matrix):
    """Preserve archived affine operation order rather than homogeneous dot."""
    points, matrix = np.asarray(points), np.asarray(matrix)
    numerator = points @ matrix[:2, :2].T + matrix[:2, 2]
    if np.array_equal(matrix[2], [0., 0., 1.]):
        return numerator
    denominator = points @ matrix[2, :2] + matrix[2, 2]
    if not np.isfinite(denominator).all() or not np.all(np.abs(denominator) > 1e-10):
        raise ValueError("Transform crosses a projective horizon")
    return numerator / denominator[..., None]


def state_identity(state):
    return tuple((p, state.groups[p], tuple(np.round(state.poses[p].ravel(), 5))) for p in sorted(state.poses))


def retain_states(states, width):
    unique = {}
    for state in states:
        key = state_identity(state)
        old = unique.get(key)
        if old is None or state.score > old.score:
            unique[key] = state
    ordered = sorted(unique.values(), key=lambda s: (-len(s.edges), -s.score, state_identity(s)))
    tiers = {}
    for state in ordered:
        tiers.setdefault(len(state.edges), state)
    counts = sorted(tiers)
    if len(counts) > width:
        counts = [counts[i] for i in np.linspace(0, len(counts) - 1, width).round().astype(int)]
    selected = [tiers[k] for k in counts]
    ids = {state_identity(s) for s in selected}
    for state in ordered:
        key = state_identity(state)
        if key not in ids and len(selected) < width:
            selected.append(state)
            ids.add(key)
    return selected


def mask_samples(mask, grid):
    h, w = mask.shape
    x, y = np.meshgrid(np.linspace(0, w - 1, grid), np.linspace(0, h - 1, grid))
    points = np.column_stack((x.ravel(), y.ravel()))
    paper = mask[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)] >= 128
    return points, paper


def collision_fraction(first, second, matrix, sampled):
    def direction(mask, other, transform, sample):
        points, paper = sample
        xx, yy = np.rint(map_points(points, transform)).astype(np.int64).T
        inside = (xx >= 0) & (xx < other.shape[1]) & (yy >= 0) & (yy < other.shape[0])
        hits = np.zeros(len(points), bool)
        hits[inside] = other[yy[inside], xx[inside]] >= 128
        return float(np.sum(paper & hits) / np.sum(paper)) if paper.any() else 0.
    return max(direction(first, second, np.linalg.inv(matrix), sampled[0]), direction(second, first, matrix, sampled[1]))


def merge(state, edge, index, masks, sampled, config, cache):
    a, b = edge.a, edge.b
    ga, gb = state.groups[a], state.groups[b]
    if ga == gb:
        return None
    first = [p for p in state.poses if state.groups[p] == ga]
    second = [p for p in state.poses if state.groups[p] == gb]
    move = state.poses[a] @ edge.transform @ np.linalg.inv(state.poses[b])
    poses = {**state.poses, **{p: move @ state.poses[p] for p in second}}
    for x in first:
        for y in second:
            transform = np.linalg.inv(poses[x]) @ poses[y]
            if not np.array_equal(transform[2], [0., 0., 1.]):
                if not warp_check(transform, masks[y].shape, masks[x].shape, config, grid=3)["valid"]:
                    return None
            else:
                with np.errstate(invalid="ignore"):
                    scale = float(np.sqrt(np.linalg.det(transform[:2, :2])))
                if not .1 <= scale <= 10:
                    return None
            key = (x, y, tuple(np.round(transform.ravel(), 6)))
            overlap = cache.get(key)
            if overlap is None:
                overlap = collision_fraction(masks[x], masks[y], transform, (sampled[x], sampled[y]))
                if len(cache) >= 8192:
                    cache.clear()
                cache[key] = overlap
            if overlap > config.maximum_assembly_collision:
                return None
    members = first + second
    root = min(members)
    gauge = np.linalg.inv(poses[root])
    poses = {**poses, **{p: gauge @ poses[p] for p in members}}
    for p in members:
        if not np.array_equal(poses[p][2], [0., 0., 1.]):
            if abs(poses[p][2, 2]) < 1e-10:
                return None
            poses[p] = poses[p] / poses[p][2, 2]
    groups = {**state.groups, **{p: root for p in members}}
    return LegacyState(poses, groups, (*state.edges, index), state.score + edge.weight)


def cycle_summary(state, edges, masks, config):
    by_pair = defaultdict(list)
    for edge in edges:
        a, b = edge.a, edge.b
        if state.groups[a] != state.groups[b]:
            continue
        transform = np.linalg.inv(state.poses[a]) @ state.poses[b]
        h, w = masks[b].shape
        corners = np.array([[0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]])
        error = float(np.max(np.linalg.norm(map_points(corners, transform) - map_points(corners, edge.transform), axis=1)))
        by_pair[a, b].append((error, edge.weight))
    satisfied = sum(min(e for e, _ in values) <= config.cycle_tolerance_px for values in by_pair.values())
    bonus = sum(max((w for e, w in values if e <= config.cycle_tolerance_px), default=0.) for values in by_pair.values())
    penalty = sum(max(w for _, w in values) for values in by_pair.values() if min(e for e, _ in values) > config.cycle_tolerance_px)
    return {"piece_pairs_consistent": satisfied, "piece_pairs_in_conflict": len(by_pair) - satisfied,
            "score_adjustment": .1 * (bonus - penalty), "scope": "historical all-candidate-pairs diagnostic"}


def solve(problem, config):
    masks = {p: np.asarray(Image.fromarray(piece.mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)))
             for p, piece in problem.pieces.items()}
    sampled = {p: mask_samples(mask, config.collision_grid) for p, mask in masks.items()}
    seed = LegacyState({p: np.eye(3) for p in problem.pieces}, {p: p for p in problem.pieces})
    beam, cache, budget, accepted, processed = [seed], {}, Budget(config), 0, 0
    try:
        stopped = False
        for index, edge in enumerate(problem.edges):
            alternatives = list(beam)
            for state in beam:
                if not budget.take():
                    stopped = True
                    break
                joined = merge(state, edge, index, masks, sampled, config, cache)
                if joined is not None:
                    accepted += 1
                    alternatives.append(joined)
            beam = retain_states(alternatives, config.beam_width)
            processed += 1
            if stopped:
                break
    except KeyboardInterrupt:
        budget.reason = "interrupted"
    scored = [(state, cycle_summary(state, problem.edges, masks, config)) for state in beam]
    scored.sort(key=lambda item: (-len(item[0].edges), -(item[0].score + item[1]["score_adjustment"]), state_identity(item[0])))
    states = [ResultState(s.poses, s.groups, tuple(problem.edges[i].id for i in s.edges),
                          tuple(problem.edges[i].id for i in s.edges), {}, s.score + c["score_adjustment"]) for s, c in scored]
    return SearchResult(states, {"proposals": budget.proposals, "accepted_proposals": accepted,
                        "processed_edges": processed, "candidate_edges": len(problem.edges),
                        "historical_cycle_diagnostics": scored[0][1], "occurrence_support_enforced": False,
                        "objective": "tree-join-count first, then weight plus final cycle adjustment"}, budget.reason)

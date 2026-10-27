from __future__ import annotations

from collections import defaultdict, deque
from math import hypot

import numpy as np


Point = tuple[int, int]
Edge = tuple[Point, Point]


def piece_id(label: int) -> str:
    return f"piece_{label:03d}"


def _canonical_edge(a: Point, b: Point) -> Edge:
    return (a, b) if a <= b else (b, a)


def _edge_components(edges: list[Edge]) -> list[list[Edge]]:
    by_node: dict[Point, list[int]] = defaultdict(list)
    for index, (a, b) in enumerate(edges):
        by_node[a].append(index)
        by_node[b].append(index)
    remaining = set(range(len(edges)))
    components: list[list[Edge]] = []
    while remaining:
        seed = next(iter(remaining))
        queue = deque([seed])
        component_indices: set[int] = set()
        while queue:
            edge_index = queue.popleft()
            if edge_index in component_indices:
                continue
            component_indices.add(edge_index)
            remaining.discard(edge_index)
            for node in edges[edge_index]:
                for neighbor_index in by_node[node]:
                    if neighbor_index not in component_indices:
                        queue.append(neighbor_index)
        components.append([edges[index] for index in sorted(component_indices)])
    return components


def _component_to_trails(edges: list[Edge]) -> list[list[Point]]:
    adjacency: dict[Point, list[tuple[Point, Edge]]] = defaultdict(list)
    unused: set[Edge] = set()
    for a, b in edges:
        edge = _canonical_edge(a, b)
        unused.add(edge)
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))

    trails: list[list[Point]] = []
    while unused:
        incident_nodes = {node for edge in unused for node in edge}
        endpoints = [
            node
            for node in incident_nodes
            if sum(edge in unused for _, edge in adjacency[node]) == 1
        ]
        start = min(endpoints or incident_nodes)
        trail = [start]
        current = start
        previous: Point | None = None
        while True:
            choices = [
                (neighbor, edge)
                for neighbor, edge in adjacency[current]
                if edge in unused and neighbor != previous
            ]
            if not choices:
                choices = [
                    (neighbor, edge)
                    for neighbor, edge in adjacency[current]
                    if edge in unused
                ]
            if not choices:
                break
            neighbor, edge = sorted(choices, key=lambda item: item[0])[0]
            unused.remove(edge)
            trail.append(neighbor)
            previous, current = current, neighbor
            remaining_degree = sum(
                candidate in unused for _, candidate in adjacency[current]
            )
            if remaining_degree > 1:
                break
        if len(trail) >= 2:
            trails.append(trail)
    return trails


def _extract_pair_edges(labels: np.ndarray) -> dict[tuple[int, int], list[Edge]]:
    result: dict[tuple[int, int], list[Edge]] = defaultdict(list)
    left = labels[:, :-1]
    right = labels[:, 1:]
    ys, xs = np.nonzero(left != right)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        pair = tuple(sorted((int(left[y, x]), int(right[y, x]))))
        result[pair].append(((x + 1, y), (x + 1, y + 1)))

    top = labels[:-1, :]
    bottom = labels[1:, :]
    ys, xs = np.nonzero(top != bottom)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        pair = tuple(sorted((int(top[y, x]), int(bottom[y, x]))))
        result[pair].append(((x, y + 1), (x + 1, y + 1)))
    return result


def extract_seams(labels: np.ndarray) -> list[dict]:
    seams: list[dict] = []
    seam_index = 0
    for pair, edges in sorted(_extract_pair_edges(labels).items()):
        for component in _edge_components(edges):
            for trail in _component_to_trails(component):
                seam_index += 1
                length = sum(
                    hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(trail[:-1], trail[1:], strict=True)
                )
                seams.append(
                    {
                        "seam_id": f"seam_{seam_index:04d}",
                        "owner_a": piece_id(pair[0]),
                        "owner_b": piece_id(pair[1]),
                        "owner_a_direction": "forward",
                        "owner_b_direction": "reverse",
                        "canonical_polyline_xy": [[float(x), float(y)] for x, y in trail],
                        "length_px": round(float(length), 4),
                        "grid_edge_count": len(trail) - 1,
                    }
                )
    return seams


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for index in range(1, len(values)):
        value = int(values[index])
        if value != current:
            result.append((start, index, current))
            start = index
            current = value
    result.append((start, len(values), current))
    return result


def extract_page_borders(labels: np.ndarray) -> list[dict]:
    height, width = labels.shape
    segments: list[dict] = []
    index = 0
    definitions = [
        ("top", labels[0, :], lambda a, b: [[float(a), 0.0], [float(b), 0.0]]),
        ("right", labels[:, -1], lambda a, b: [[float(width), float(a)], [float(width), float(b)]]),
        ("bottom", labels[-1, :], lambda a, b: [[float(b), float(height)], [float(a), float(height)]]),
        ("left", labels[:, 0], lambda a, b: [[0.0, float(b)], [0.0, float(a)]]),
    ]
    for side, values, coordinate_builder in definitions:
        for start, end, label in _runs(values):
            index += 1
            segments.append(
                {
                    "border_segment_id": f"border_{index:04d}",
                    "piece_id": piece_id(label),
                    "page_side": side,
                    "canonical_polyline_xy": coordinate_builder(start, end),
                    "length_px": float(end - start),
                    "mate": None,
                }
            )
    return segments


def extract_piece_records(
    labels: np.ndarray,
    seams: list[dict],
    border_segments: list[dict],
) -> list[dict]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    seam_ids: dict[str, list[str]] = defaultdict(list)
    for seam in seams:
        a = seam["owner_a"]
        b = seam["owner_b"]
        neighbors[a].add(b)
        neighbors[b].add(a)
        seam_ids[a].append(seam["seam_id"])
        seam_ids[b].append(seam["seam_id"])
    border_ids: dict[str, list[str]] = defaultdict(list)
    for segment in border_segments:
        border_ids[segment["piece_id"]].append(segment["border_segment_id"])

    pieces: list[dict] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        mask = labels == label
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        identifier = piece_id(label)
        pieces.append(
            {
                "piece_id": identifier,
                "label": label,
                "area_px": int(mask.sum()),
                "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
                "centroid_xy": [round(float(xs.mean()), 5), round(float(ys.mean()), 5)],
                "neighbors": sorted(neighbors[identifier]),
                "seam_ids": sorted(seam_ids[identifier]),
                "page_border_segment_ids": sorted(border_ids[identifier]),
                "canonical_piece_path": None,
                "canonical_mask_path": None,
            }
        )
    return pieces


def extract_truth(labels: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    seams = extract_seams(labels)
    borders = extract_page_borders(labels)
    pieces = extract_piece_records(labels, seams, borders)
    return pieces, seams, borders


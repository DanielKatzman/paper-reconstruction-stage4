from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class Piece:
    id: str
    observation: str
    image_path: Path
    mask: np.ndarray
    contour: dict[str, np.ndarray]

    @property
    def shape(self):
        return self.mask.shape


@dataclass
class Edge:
    id: str
    a: str
    b: str
    transform: np.ndarray
    weight: float
    occurrences: list[dict]
    raw: dict = field(default_factory=dict)


@dataclass
class Problem:
    pieces: dict[str, Piece]
    edges: list[Edge]
    candidates: dict[str, dict]
    manifest: dict[str, Any]
    pool: dict[str, Any]
    root: Path
    input_summary: dict = field(default_factory=dict)
    structural: Any = None
    initial_states: list = field(default_factory=list)
    deadline: float | None = None
    constraint_profile: dict | None = None
    page_context: Any = None
    capture_metadata: dict | None = None


@dataclass
class State:
    poses: dict[str, np.ndarray]
    groups: dict[str, str]
    selected: tuple[str, ...] = ()
    tree: tuple[str, ...] = ()
    assignments: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    active_support: dict[str, dict] = field(default_factory=dict)

    def components(self):
        return [sorted(p for p in self.groups if self.groups[p] == g) for g in sorted(set(self.groups.values()))]


def singleton_state(problem):
    return State({p: np.eye(3) for p in sorted(problem.pieces)}, {p: p for p in sorted(problem.pieces)})

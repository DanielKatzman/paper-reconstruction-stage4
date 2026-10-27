from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def normalized_distribution(values: dict[str, float]) -> tuple[list[str], np.ndarray]:
    names = [name for name, value in values.items() if value > 0]
    weights = np.asarray([values[name] for name in names], dtype=np.float64)
    weights /= weights.sum()
    return names, weights


def exact_schedule(
    count: int,
    distribution: dict[str, float],
    *,
    seed: int,
) -> list[str]:
    names, weights = normalized_distribution(distribution)
    raw = weights * count
    counts = np.floor(raw).astype(int)
    remainder = count - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    schedule = [
        name
        for name, item_count in zip(names, counts.tolist(), strict=True)
        for _ in range(item_count)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(schedule)
    return schedule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_source_splits(
    sources: list[Path],
    source_dir: Path,
    ratios: dict[str, float],
    *,
    seed: int,
) -> tuple[dict[Path, str], list[dict]]:
    if not sources:
        return {}, []
    names, weights = normalized_distribution(ratios)
    if len(sources) < len(names):
        raise ValueError(
            f"{len(sources)} source files cannot populate {len(names)} non-empty "
            "splits; provide more sources or set unused split ratios to zero"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)
    remaining_count = len(shuffled) - len(names)
    raw = weights * remaining_count
    extra = np.floor(raw).astype(int)
    remainder = remaining_count - int(extra.sum())
    order = np.argsort(-(raw - extra), kind="stable")
    for index in order[:remainder]:
        extra[index] += 1
    counts = extra + 1

    assignments: dict[Path, str] = {}
    records: list[dict] = []
    cursor = 0
    for split_name, split_count in zip(names, counts.tolist(), strict=True):
        for path in shuffled[cursor : cursor + split_count]:
            relative = path.relative_to(source_dir).as_posix()
            assignments[path] = split_name
            records.append(
                {
                    "source_relpath": relative,
                    "source_sha256": sha256_file(path),
                    "split": split_name,
                }
            )
        cursor += split_count
    records.sort(key=lambda item: item["source_relpath"])
    return assignments, records


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CUT_WEIGHTS = {
    "straight": 0.08,
    "smooth_wave": 0.16,
    "sharp_zigzag": 0.22,
    "deep_notches": 0.20,
    "torn": 0.14,
    "mixed_scissor": 0.20,
}

DEFAULT_DIFFICULTY_MIX = {
    "simple": 0.15,
    "moderate": 0.35,
    "hard": 0.35,
    "extreme": 0.15,
}

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "holdout": 0.10,
}

DIFFICULTY_CUT_WEIGHTS = {
    "simple": {
        "straight": 0.35,
        "smooth_wave": 0.35,
        "sharp_zigzag": 0.05,
        "deep_notches": 0.05,
        "torn": 0.15,
        "mixed_scissor": 0.05,
    },
    "moderate": {
        "straight": 0.15,
        "smooth_wave": 0.25,
        "sharp_zigzag": 0.15,
        "deep_notches": 0.12,
        "torn": 0.20,
        "mixed_scissor": 0.13,
    },
    "hard": {
        "straight": 0.05,
        "smooth_wave": 0.15,
        "sharp_zigzag": 0.22,
        "deep_notches": 0.20,
        "torn": 0.18,
        "mixed_scissor": 0.20,
    },
    "extreme": {
        "straight": 0.02,
        "smooth_wave": 0.08,
        "sharp_zigzag": 0.30,
        "deep_notches": 0.25,
        "torn": 0.10,
        "mixed_scissor": 0.25,
    },
}


@dataclass(slots=True)
class GeneratorConfig:
    output_dir: Path
    source_dir: Path | None = None
    puzzle_count: int = 10
    seed: int = 20261004
    min_pieces: int = 2
    max_pieces: int = 30
    page_width: int = 1240
    page_height: int = 1754
    min_piece_area_fraction: float = 0.0025
    min_piece_dimension_px: int = 10
    max_split_attempts: int = 300
    preview_max_side: int = 1400
    resume: bool = False
    cut_family_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CUT_WEIGHTS)
    )
    difficulty_mix: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIFFICULTY_MIX)
    )
    split_ratios: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS)
    )

    def validate(self) -> None:
        if self.puzzle_count < 1:
            raise ValueError("puzzle_count must be at least 1")
        if self.min_pieces < 2:
            raise ValueError("min_pieces must be at least 2 (one cut)")
        if self.max_pieces < self.min_pieces:
            raise ValueError("max_pieces must be >= min_pieces")
        if self.page_width < 64 or self.page_height < 64:
            raise ValueError("page dimensions must both be at least 64 pixels")
        if not 0.0 < self.min_piece_area_fraction < 0.25:
            raise ValueError("min_piece_area_fraction must be in (0, 0.25)")
        if self.min_piece_area_fraction * self.max_pieces >= 0.80:
            raise ValueError(
                "min_piece_area_fraction is too large for max_pieces; "
                "their product must be below 0.80"
            )
        if self.min_piece_dimension_px < 2:
            raise ValueError("min_piece_dimension_px must be at least 2")
        if self.max_split_attempts < 10:
            raise ValueError("max_split_attempts must be at least 10")
        unknown = set(self.cut_family_weights) - set(DEFAULT_CUT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown cut families: {sorted(unknown)}")
        if not self.cut_family_weights or any(v < 0 for v in self.cut_family_weights.values()):
            raise ValueError("cut family weights must be non-negative")
        if sum(self.cut_family_weights.values()) <= 0:
            raise ValueError("at least one cut family must have positive weight")
        self._validate_distribution(
            self.difficulty_mix,
            allowed=set(DEFAULT_DIFFICULTY_MIX),
            name="difficulty_mix",
        )
        self._validate_distribution(
            self.split_ratios,
            allowed=set(DEFAULT_SPLIT_RATIOS),
            name="split_ratios",
        )

    @staticmethod
    def _validate_distribution(
        values: dict[str, float], *, allowed: set[str], name: str
    ) -> None:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
        if not values or any(value < 0 for value in values.values()):
            raise ValueError(f"{name} values must be non-negative")
        if sum(values.values()) <= 0:
            raise ValueError(f"{name} must contain a positive value")

    @property
    def page_area(self) -> int:
        return self.page_width * self.page_height

    @property
    def min_piece_area_px(self) -> int:
        return max(32, round(self.page_area * self.min_piece_area_fraction))

    def normalized_cut_weights(
        self, difficulty_tier: str | None = None
    ) -> tuple[list[str], list[float]]:
        values = DIFFICULTY_CUT_WEIGHTS.get(
            difficulty_tier or "", self.cut_family_weights
        )
        names = list(values)
        total = sum(values.values())
        return names, [values[name] / total for name in names]

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        result["source_dir"] = str(self.source_dir) if self.source_dir else None
        result.pop("resume", None)
        return result

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        output_dir: Path | None = None,
        source_dir: Path | None = None,
    ) -> "GeneratorConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw.pop("schema_version", None)
        if output_dir is not None:
            raw["output_dir"] = output_dir
        elif "output_dir" in raw:
            raw["output_dir"] = Path(raw["output_dir"])
        else:
            raise ValueError("output_dir is required in YAML or on the command line")
        if source_dir is not None:
            raw["source_dir"] = source_dir
        elif raw.get("source_dir"):
            raw["source_dir"] = Path(raw["source_dir"])
        config = cls(**raw)
        config.validate()
        return config

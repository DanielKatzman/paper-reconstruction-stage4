from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True, slots=True)
class ProfileResult:
    family: str
    values: np.ndarray
    parameters: dict[str, float | int]


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def _random_anchor_signal(
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    smooth_sigma: float,
) -> np.ndarray:
    anchors_x = np.linspace(0.0, 1.0, count)
    anchors_y = rng.uniform(-1.0, 1.0, size=count)
    anchors_y[[0, -1]] *= 0.35
    signal = np.interp(t, anchors_x, anchors_y)
    if smooth_sigma > 0:
        signal = gaussian_filter1d(signal, smooth_sigma, mode="nearest")
    maximum = float(np.max(np.abs(signal)))
    return signal / maximum if maximum > 1e-8 else signal


def make_profile(
    family: str,
    t: np.ndarray,
    transverse_span: float,
    rng: np.random.Generator,
    difficulty_tier: str = "moderate",
) -> ProfileResult:
    """Create a single-valued canonical cut profile in local cut coordinates."""
    t = np.asarray(t, dtype=np.float64)
    amplitude_multiplier = {
        "simple": 0.55,
        "moderate": 0.80,
        "hard": 1.05,
        "extreme": 1.30,
    }.get(difficulty_tier, 1.0)
    amplitude_scale = (
        transverse_span * rng.uniform(0.08, 0.24) * amplitude_multiplier
    )
    phase = rng.uniform(0.0, 2.0 * np.pi)
    params: dict[str, float | int] = {
        "amplitude_px": round(float(amplitude_scale), 5),
        "phase_rad": round(float(phase), 6),
        "difficulty_tier": difficulty_tier,
    }

    if family == "straight":
        slope = rng.uniform(-0.20, 0.20) * transverse_span
        values = slope * (t - 0.5)
        params["slope_span"] = round(float(slope), 5)

    elif family == "smooth_wave":
        cycles = int(rng.integers(1, 4))
        secondary = rng.uniform(0.15, 0.40)
        values = amplitude_scale * (
            np.sin(2.0 * np.pi * cycles * t + phase)
            + secondary * np.sin(2.0 * np.pi * (cycles + 1) * t - phase * 0.5)
        )
        params.update(cycles=cycles, secondary=round(float(secondary), 5))

    elif family == "sharp_zigzag":
        teeth = int(rng.integers(3, 10))
        irregular = _random_anchor_signal(
            t, rng, count=teeth + 2, smooth_sigma=0.0
        )
        triangle = _triangle_wave(2.0 * np.pi * teeth * t + phase)
        values = amplitude_scale * (0.72 * triangle + 0.28 * irregular)
        params["teeth"] = teeth

    elif family == "deep_notches":
        notch_count = int(rng.integers(2, 6))
        values = np.zeros_like(t)
        for _ in range(notch_count):
            center = rng.uniform(0.12, 0.88)
            width = rng.uniform(0.025, 0.11)
            depth = rng.uniform(0.65, 1.25) * rng.choice((-1.0, 1.0))
            values += depth * np.exp(-0.5 * ((t - center) / width) ** 2)
        values += 0.20 * np.sin(2.0 * np.pi * rng.integers(1, 4) * t + phase)
        maximum = float(np.max(np.abs(values)))
        values = amplitude_scale * values / max(maximum, 1e-8)
        params["notch_count"] = notch_count

    elif family == "torn":
        low = _random_anchor_signal(t, rng, count=int(rng.integers(6, 12)), smooth_sigma=3.0)
        high = _random_anchor_signal(t, rng, count=int(rng.integers(20, 45)), smooth_sigma=0.7)
        values = amplitude_scale * (0.78 * low + 0.22 * high)
        params["multiscale"] = 1

    elif family == "mixed_scissor":
        control_count = int(rng.integers(5, 11))
        sharp = _random_anchor_signal(t, rng, count=control_count, smooth_sigma=0.0)
        smooth = _random_anchor_signal(t, rng, count=max(4, control_count // 2), smooth_sigma=4.0)
        cycles = int(rng.integers(1, 4))
        wave = np.sin(2.0 * np.pi * cycles * t + phase)
        values = amplitude_scale * (0.48 * sharp + 0.32 * smooth + 0.20 * wave)
        params.update(control_count=control_count, cycles=cycles)

    else:
        raise ValueError(f"unsupported cut family: {family}")

    edge_taper = np.sin(np.pi * np.clip(t, 0.0, 1.0)) ** 0.35
    values = values * (0.72 + 0.28 * edge_taper)
    return ProfileResult(family=family, values=values, parameters=params)

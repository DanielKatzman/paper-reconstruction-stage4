from __future__ import annotations

from dataclasses import replace

from .model import BoundaryRoleModelConfig


MODEL_CANDIDATE_NAMES = (
    "base_tcn",
    "wide_tcn",
    "large_tcn",
    "hybrid_attention",
    "xlarge_hybrid_attention",
    "global_context_hybrid",
    "geometry_only",
    "raster_only",
)


def candidate_model_config(name: str) -> BoundaryRoleModelConfig:
    """Return one controlled contour-model candidate for Stage 2 experiments."""

    base = BoundaryRoleModelConfig()
    candidates = {
        "base_tcn": base,
        "wide_tcn": replace(base, branch_channels=96, fusion_channels=192),
        "large_tcn": replace(base, branch_channels=128, fusion_channels=256),
        "hybrid_attention": replace(
            base,
            branch_channels=96,
            fusion_channels=192,
            architecture="hybrid_attention",
            attention_layers=2,
            attention_heads=4,
        ),
        "xlarge_hybrid_attention": replace(
            base,
            branch_channels=160,
            fusion_channels=320,
            architecture="hybrid_attention",
            attention_layers=4,
            attention_heads=8,
        ),
        "global_context_hybrid": replace(
            base,
            branch_channels=128,
            fusion_channels=256,
            architecture="global_context_hybrid",
            attention_layers=2,
            attention_heads=8,
            use_global_context=True,
            global_context_points=256,
            global_mask_size=64,
            global_attention_layers=2,
        ),
        "geometry_only": replace(base, use_raster=False),
        "raster_only": replace(base, use_geometry=False),
    }
    try:
        config = candidates[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model candidate {name!r}; choose from {MODEL_CANDIDATE_NAMES}"
        ) from exc
    config.validate()
    return config


def candidate_names() -> tuple[str, ...]:
    return MODEL_CANDIDATE_NAMES

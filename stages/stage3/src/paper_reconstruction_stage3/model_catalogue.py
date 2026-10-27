"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    task: str = "retrieval"
    width: int = 256
    layers: int = 4
    heads: int = 8
    bundle: str = "G+P"
    backbone: str | None = None
    pretrained: bool = False
    reference_mode: str = "full"



def catalogue():
    specs = [ModelSpec("pair_mlp", "P1", "pair"), ModelSpec("paired_strip_cnn", "P2", "pair", bundle="G+P+A")]
    return {s.name: s for s in specs}


@dataclass(frozen=True)
class ProfileConfig:
    """Frozen architecture construction needs no training or dataset config."""
    seed: int = 4071

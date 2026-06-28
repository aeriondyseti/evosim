"""World modules: pluggable spatial topologies and their query primitives."""

from __future__ import annotations

from . import fields
from .base import World
from .fields import decay, diffuse, life_like, map_field, regrow
from .grid import MOORE_OFFSETS, VON_NEUMANN_OFFSETS, ToricGrid2D

__all__ = [
    "World",
    "ToricGrid2D",
    "MOORE_OFFSETS",
    "VON_NEUMANN_OFFSETS",
    "fields",
    "decay",
    "diffuse",
    "life_like",
    "map_field",
    "regrow",
]

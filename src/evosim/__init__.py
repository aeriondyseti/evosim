"""evosim — a modular, high-performance library for large-scale agent-based
evolutionary simulations, built on a JAX core.

See ``SPEC.md`` for the full design. Public API is grown module-by-module as the
build progresses; import submodules directly until the surface stabilizes.
"""

from __future__ import annotations

from .schema import Field, Schema, RESERVED_FIELDS, enable_x64

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Field",
    "Schema",
    "RESERVED_FIELDS",
    "enable_x64",
]

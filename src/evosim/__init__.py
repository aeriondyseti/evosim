"""evosim — a modular, high-performance library for large-scale agent-based
evolutionary simulations, built on a JAX core.

See ``SPEC.md`` for the full design. Public API is grown module-by-module as the
build progresses; import submodules directly for anything not re-exported here.
"""

from __future__ import annotations

from . import backend, interaction, operators, population, rng, world
from .backend import Backend, JAXBackend, get_backend, use_backend
from .schema import RESERVED_FIELDS, Field, Schema, enable_x64
from .scheduler import Scheduler
from .state import State, state_fingerprint
from .system import DEFAULT_STAGES, Context, System, system

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # schema
    "Field",
    "Schema",
    "RESERVED_FIELDS",
    "enable_x64",
    # state
    "State",
    "state_fingerprint",
    # systems / scheduling
    "System",
    "system",
    "Context",
    "Scheduler",
    "DEFAULT_STAGES",
    # backend
    "Backend",
    "JAXBackend",
    "get_backend",
    "use_backend",
    # submodules
    "rng",
    "population",
    "backend",
    "world",
    "operators",
    "interaction",
]

"""Backend abstraction (JAX first).

Per SPEC, the engine is built against a backend interface so an alternative compute
backend (Numba/CuPy, Rust) could be slotted in later. The backend owns the *compilation
and control-flow primitives* — ``jit``, ``scan``, ``vmap`` — plus the array namespace
(``xp``) and device info. Low-level elementwise math in systems uses the array namespace
directly; a future backend would reimplement systems against its own ``xp``.

The default (and currently only) backend is :class:`JAXBackend`. Use :func:`get_backend`
to obtain it, or :func:`use_backend` / :func:`set_default_backend` to swap.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = [
    "Backend",
    "JAXBackend",
    "register_backend",
    "get_backend",
    "set_default_backend",
    "use_backend",
]


class Backend:
    """Abstract compute backend: compilation/control-flow primitives + array namespace."""

    name: str = "abstract"

    @property
    def xp(self) -> Any:
        """The array namespace (numpy-like module)."""
        raise NotImplementedError

    def jit(self, fn: Callable, *, static_argnums=None, donate_argnums=None) -> Callable:
        raise NotImplementedError

    def scan(self, f: Callable, init: Any, xs: Any = None, length: int | None = None):
        raise NotImplementedError

    def vmap(self, fn: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
        raise NotImplementedError

    def devices(self) -> list:
        raise NotImplementedError

    def default_device(self):
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Backend {self.name!r}>"


class JAXBackend(Backend):
    """JAX/XLA backend. Runs identically on CPU/GPU/TPU depending on installed jaxlib."""

    name = "jax"

    @property
    def xp(self):
        return jnp

    def jit(self, fn, *, static_argnums=None, donate_argnums=None):
        kwargs: dict[str, Any] = {}
        if static_argnums is not None:
            kwargs["static_argnums"] = static_argnums
        if donate_argnums is not None:
            kwargs["donate_argnums"] = donate_argnums
        return jax.jit(fn, **kwargs)

    def scan(self, f, init, xs=None, length=None):
        return jax.lax.scan(f, init, xs, length=length)

    def vmap(self, fn, in_axes=0, out_axes=0):
        return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)

    def devices(self):
        return jax.devices()

    def default_device(self):
        return jax.devices()[0]


# -- registry ----------------------------------------------------------------

_REGISTRY: dict[str, Callable[[], Backend]] = {}
_DEFAULT_NAME: str = "jax"
_INSTANCES: dict[str, Backend] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend factory under ``name``."""
    _REGISTRY[name] = factory


def get_backend(name: str | None = None) -> Backend:
    """Return the backend instance for ``name`` (or the current default)."""
    key = name or _DEFAULT_NAME
    if key not in _INSTANCES:
        if key not in _REGISTRY:
            raise KeyError(f"unknown backend {key!r}; registered: {sorted(_REGISTRY)}")
        _INSTANCES[key] = _REGISTRY[key]()
    return _INSTANCES[key]


def set_default_backend(name: str) -> None:
    """Set the process-wide default backend by name."""
    global _DEFAULT_NAME
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    _DEFAULT_NAME = name


class use_backend:
    """Context manager to temporarily switch the default backend.

    >>> with use_backend("jax"):
    ...     ...
    """

    def __init__(self, name: str):
        self._name = name
        self._prev: str | None = None

    def __enter__(self) -> Backend:
        global _DEFAULT_NAME
        self._prev = _DEFAULT_NAME
        set_default_backend(self._name)
        return get_backend(self._name)

    def __exit__(self, *exc) -> None:
        global _DEFAULT_NAME
        if self._prev is not None:
            _DEFAULT_NAME = self._prev


register_backend("jax", JAXBackend)

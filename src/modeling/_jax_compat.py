"""
JAX <-> NumPyro compatibility shim.

Recent JAX releases (>= 0.7) have moved `Primitive` to `jax.extend.core` and
removed `MapPrimitive` / `xla_pmap_p` entirely (since `pmap` was deprecated).
NumPyro 0.18 still does

    from jax.extend.core.primitives import call_p, closed_call_p, jit_p, xla_pmap_p

and uses `xla_pmap_p` only as a dispatch-dict key inside
`numpyro.ops.provenance.track_deps_rules`. The key is never actually matched
unless a user model contains a `pmap`, which ours do not.

Importing this module BEFORE `jax` / `numpyro` ensures an inert stub
primitive exists at every legacy location so the chained `from ... import`
succeeds without modifying any site-packages file. It also defaults macOS to
CPU when the user has not explicitly selected a JAX platform, which avoids a
hard crash in environments where the experimental `METAL` backend is present
but no usable GPU is visible.

Usage: `from src.modeling import _jax_compat  # noqa: F401`
at the very top of any module that imports `jax` or `numpyro`.
"""
from __future__ import annotations

import importlib
import os
import platform


def _canonicalize_platform_value(raw: str) -> str:
    """Normalize known platform names to the spelling JAX expects on this env."""
    value = raw.strip()
    if not value:
        return value
    if value.lower() == "metal":
        return "METAL"
    return value


def _canonicalize_platform_env() -> None:
    """Normalize explicit user platform env vars before JAX reads them."""
    if os.environ.get("JAX_PLATFORMS"):
        parts = os.environ["JAX_PLATFORMS"].split(",")
        os.environ["JAX_PLATFORMS"] = ",".join(_canonicalize_platform_value(p) for p in parts)
    if os.environ.get("JAX_PLATFORM_NAME"):
        os.environ["JAX_PLATFORM_NAME"] = _canonicalize_platform_value(os.environ["JAX_PLATFORM_NAME"])


def _configure_platform_env() -> None:
    """Default macOS to CPU unless the user explicitly chose a platform."""
    _canonicalize_platform_env()
    if platform.system() != "Darwin":
        return
    if os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME"):
        return
    os.environ["JAX_PLATFORMS"] = "cpu"


def _resolve_primitive_class():
    """Find a Primitive class in whichever module this JAX exposes it from."""
    candidates = (
        ("jax.extend.core", "Primitive"),
        ("jax._src.core", "Primitive"),
        ("jax.core", "Primitive"),
    )
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        cls = getattr(mod, attr, None)
        if cls is not None:
            return cls
    return None


def _make_stub():
    """Build an inert primitive named 'xla_pmap' with multiple_results=True."""
    try:
        from jax._src.core import MapPrimitive  # type: ignore[attr-defined]

        return MapPrimitive("xla_pmap")
    except Exception:
        pass

    prim_cls = _resolve_primitive_class()
    if prim_cls is None:
        return None

    stub = prim_cls("xla_pmap")
    try:
        stub.multiple_results = True
    except Exception:
        pass
    return stub


def _install_stub() -> None:
    stub = _make_stub()
    if stub is None:
        return

    for mod_name in (
        "jax.extend.core.primitives",
        "jax.interpreters.pxla",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if not hasattr(mod, "xla_pmap_p"):
            try:
                setattr(mod, "xla_pmap_p", stub)
            except Exception:
                pass

_configure_platform_env()
_install_stub()

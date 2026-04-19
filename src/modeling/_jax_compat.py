"""
JAX <-> NumPyro compatibility shim.

JAX 0.7 removed `xla_pmap_p` from `jax.extend.core.primitives` and from
`jax.interpreters.pxla`. NumPyro's `numpyro.ops.provenance` still imports that
primitive (it uses it only as a dispatch-dict key inside `track_deps_rules`,
so it is never actually invoked unless the user model contains a `pmap`, which
ours does not).

Importing this module BEFORE `numpyro` injects a benign stub primitive into
both legacy locations if they are missing, so the import chain completes
without modifying any site-packages file.

Usage: `import src.modeling._jax_compat  # noqa: F401` at the very top of any
module that imports `numpyro`.
"""
from __future__ import annotations


def _install_stub() -> None:
    try:
        from jax._src.core import MapPrimitive  # type: ignore[attr-defined]

        stub = MapPrimitive("xla_pmap")
    except Exception:  # pragma: no cover - fallback if internals move again
        from jax.core import Primitive

        stub = Primitive("xla_pmap")
        stub.multiple_results = True

    try:
        import jax.extend.core.primitives as _p1  # type: ignore[import-not-found]

        if not hasattr(_p1, "xla_pmap_p"):
            _p1.xla_pmap_p = stub
    except Exception:
        pass

    try:
        import jax.interpreters.pxla as _p2  # type: ignore[import-not-found]

        if not hasattr(_p2, "xla_pmap_p"):
            _p2.xla_pmap_p = stub
    except Exception:
        pass


_install_stub()

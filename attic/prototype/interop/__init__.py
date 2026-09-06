"""
bartorch.interop — interoperability wrappers for third-party frameworks.

Currently available
-------------------
deepinv
    :func:`~bartorch.interop._deepinv.BartLinearPhysics` wraps any
    :class:`~bartorch.lib.BartLinop` encoding operator as a
    ``deepinv.physics.LinearPhysics`` object with a :meth:`solve` method
    backed by BART's CG solver.

    Requires ``deepinv`` as an optional dependency:
    ``pip install bartorch[deepinv]``
"""

from __future__ import annotations


def _deepinv_physics() -> type:
    """Lazy import of :class:`BartLinearPhysics` (avoids hard deepinv dep)."""
    from bartorch.interop._deepinv import BartLinearPhysics  # noqa: PLC0415

    return BartLinearPhysics


__all__ = [
    "BartLinearPhysics",
]


def __getattr__(name: str) -> object:
    if name == "BartLinearPhysics":
        return _deepinv_physics()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Handing bartorch's operators to other libraries.

Each module here is imported on use, not by the package, so an optional
dependency costs nothing to anyone who has not asked for it.
"""

from __future__ import annotations

__all__ = ["deepinv"]


def __getattr__(name: str):
    if name == "deepinv":
        from bartorch.interop import deepinv

        return deepinv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

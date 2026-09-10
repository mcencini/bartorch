"""The operator surface as it was, kept working while callers move.

Everything here has moved: :mod:`bartorch.linop` holds the linear operators
and :mod:`bartorch.nlop` the nonlinear ones, one class each rather than
classmethods on a class that was really a namespace.  The classes are the same
objects, so ``isinstance`` still holds and an operator built either way is the
same operator; what this module adds is the old spellings, which warn.

Deleting this file removes the whole deprecated surface, which is why the
classmethods are attached here rather than left on the classes.
"""

from __future__ import annotations

import warnings

from bartorch.linop import (
    FFT,
    NUFFT,
    Callback,
    Diagonal,
    LinearOperator,
    MultiplySum,
    Sampling,
    Sense,
)
from bartorch.linop.nufft import default_kspace_shape as _default_kspace_shape  # noqa: F401
from bartorch.nlop import Callback as NonlinearCallback
from bartorch.nlop import FromTorch, NonlinearOperator

__all__ = ["LinearOperator", "NonlinearOperator"]


def _shim(cls, old: str, new: str, target) -> None:
    """Attach ``cls.old`` as a classmethod that builds *target* and warns."""

    def build(_cls, *args, **kwargs):
        warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=2)
        return target(*args, **kwargs)

    name = old.split(".")[-1]
    build.__name__ = name
    build.__qualname__ = old
    build.__doc__ = f"Deprecated. Use :class:`{new}`.\n\n{target.__doc__ or ''}"
    setattr(cls, name, classmethod(build))


_shim(LinearOperator, "LinearOperator.fft", "bartorch.linop.FFT", FFT)
_shim(LinearOperator, "LinearOperator.diagonal", "bartorch.linop.Diagonal", Diagonal)
_shim(LinearOperator, "LinearOperator.sampling", "bartorch.linop.Sampling", Sampling)
_shim(LinearOperator, "LinearOperator.multiply_sum", "bartorch.linop.MultiplySum", MultiplySum)
_shim(LinearOperator, "LinearOperator.nufft", "bartorch.linop.NUFFT", NUFFT)
_shim(LinearOperator, "LinearOperator.sense", "bartorch.linop.Sense", Sense)
_shim(LinearOperator, "LinearOperator.from_callbacks", "bartorch.linop.Callback", Callback)
_shim(
    NonlinearOperator,
    "NonlinearOperator.from_callbacks",
    "bartorch.nlop.Callback",
    NonlinearCallback,
)
_shim(NonlinearOperator, "NonlinearOperator.from_torch", "bartorch.nlop.FromTorch", FromTorch)

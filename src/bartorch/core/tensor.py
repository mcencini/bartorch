"""bartorch.core.tensor — private normalisation utilities and the ``@bart_op`` decorator.

The :func:`bart_op` decorator is the **single entry point** for dtype
normalisation and should be applied to every public bartorch op function.

On the **input side** the decorator:

* Casts every ``torch.Tensor`` argument to ``complex64`` (zero-copy
  passthrough when the dtype already matches).
* Converts ``numpy.ndarray`` inputs to ``complex64`` ``torch.Tensor``.
* All other arguments (ints, strings, lists, …) pass through unchanged.

On the **output side** the decorator returns the result as-is (``complex64``
from BART).  Pass ``real_output=True`` for ops that are semantically
real-valued; the decorator will then return ``result.real`` instead.

All other names in this module are **private** (``_`` prefix) and are not
part of the public API.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["bart_op"]


# ---------------------------------------------------------------------------
# Private normalisation helpers
# ---------------------------------------------------------------------------


def _as_complex64(t: torch.Tensor) -> torch.Tensor:
    """Cast *t* to ``torch.complex64``, zero-copy if already correct."""
    if t.dtype == torch.complex64:
        return t
    return t.to(torch.complex64)


def _reverse_dims(dims: list[int] | tuple[int, ...]) -> list[int]:
    """Reverse a dim list: C-order ↔ Fortran-order (or vice versa).

    bartorch users express shapes in C-order (last index varies fastest in
    memory); BART uses Fortran order (first index varies fastest).  Reversing
    the dim list *without* touching the data switches between the two
    conventions because a C-order ``(a, b, c)`` array and a Fortran-order
    ``(c, b, a)`` array share identical byte layouts — zero copy.
    """
    return list(reversed(dims))


def _normalise_input(v):
    """Cast one argument to ``complex64 torch.Tensor`` if it is array-like.

    * ``torch.Tensor``  → cast to complex64 (zero-copy if already correct)
    * ``numpy.ndarray`` → convert to complex64 torch.Tensor
    * anything else     → returned unchanged
    """
    if isinstance(v, torch.Tensor):
        return _as_complex64(v)
    try:
        import numpy as np  # optional dep

        if isinstance(v, np.ndarray):
            return torch.from_numpy(np.asarray(v, dtype=np.complex64))
    except ImportError:
        pass
    return v


# ---------------------------------------------------------------------------
# Internal stride helper (used by the CFL writer and tests)
# ---------------------------------------------------------------------------


def _fortran_strides(dims: list[int]) -> list[int]:
    """Compute column-major (Fortran) strides for the given shape."""
    strides: list[int] = []
    s = 1
    for d in dims:
        strides.append(s)
        s *= d
    return strides


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def bart_op(func=None, *, real_output: bool = False):
    """Decorator for bartorch op functions.

    Applies transparent transformations on input and output:

    **Input normalisation** — every positional or keyword argument that is a
    ``torch.Tensor`` or ``numpy.ndarray`` is normalised to ``torch.complex64``
    (zero-copy passthrough when the dtype already matches).  A tensor keeps
    the device it is on: where a tool runs is
    :func:`bartorch.core.graph.dispatch`'s decision, taken from the inputs.

    **Output** — the return value is passed through as-is (``complex64`` from
    BART).  Set *real_output=True* for ops that are semantically real-valued;
    the decorator will then return ``result.real`` (float32).

    The decorator can be used with or without arguments::

        @bart_op                          # complex output
        def fft(input, flags, ...): ...

        @bart_op(real_output=True)        # real output
        def rss(coil_images, ...): ...

    Parameters
    ----------
    func :
        The op function to decorate.  Supplied automatically when the
        decorator is used without parentheses (``@bart_op``).
    real_output : bool
        When ``True``, return ``result.real`` (float32) instead of the
        full complex result.  Default ``False``.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            new_args = tuple(_normalise_input(a) for a in args)
            new_kwargs = {k: _normalise_input(v) for k, v in kwargs.items()}

            result = f(*new_args, **new_kwargs)

            if real_output and isinstance(result, torch.Tensor):
                return result.real
            return result

        return wrapper

    # Support both @bart_op and @bart_op(real_output=True)
    if func is not None:
        return decorator(func)
    return decorator

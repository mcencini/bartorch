"""bartorch.tools — BART CLI tool wrappers.

Every function in this sub-package routes through
:func:`bartorch.core.graph.dispatch`, which calls the embedded BART C++
extension.  All wrappers accept plain ``torch.Tensor`` / NumPy array inputs
(normalised to ``complex64`` automatically) and return plain ``torch.Tensor``
results.

Axis indices (C-order)
----------------------
Where BART expects a bitmask to select axes, the Python wrappers accept a
scalar axis index or a tuple of indices instead — including negative indices.
The conversion to BART's Fortran-order bitmask is handled transparently inside
each wrapper (see :func:`bartorch.utils.flags._axes_to_flags`).

Example: ``bt.fft(x, axes=(-1, -2))`` — 2-D FFT over the last two axes.

Tool layers
-----------
* :mod:`bartorch.tools._generated` — auto-generated thin wrappers for every
  BART command (100+), produced by ``build_tools/gen_tools.py``.
* :mod:`bartorch.tools._commands`  — imports the full generated suite and
  overrides a small set of commands with richer Pythonic APIs (e.g.
  :func:`ecalib`, :func:`caldir`, :func:`pics`).
* This ``__init__`` re-exports the final public API from ``_commands``.
"""

from __future__ import annotations

from bartorch._options import describe

__all__: list[str] = ["describe"]

# Full suite: auto-generated wrappers + special-case overrides.
# Missing when the package has not been built; silently ignored.
try:
    from bartorch.tools._commands import *  # noqa: F401,F403
    from bartorch.tools._commands import __all__ as _commands_all

    __all__ = ["describe", *_commands_all]
except ImportError:
    pass

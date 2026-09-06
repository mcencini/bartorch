"""
bartorch.lib — BART library-level operators and solvers.

This sub-package exposes persistent BART linear operators and sparse solvers
directly from the embedded BART C library, bypassing the CLI layer entirely.

The operators are constructed once, kept alive as Python objects, and applied
repeatedly without incurring construction overhead.  Solvers run entirely in
C with no Python callbacks inside the iteration loop.

Available
---------
BartLinop
    Persistent linear operator wrapping a ``BartLinopHandle`` from the C++
    extension.  Supports forward, adjoint, normal application and CG solve.

encoding_op
    Factory that creates a BART SENSE encoding operator for Cartesian or
    non-Cartesian MRI reconstruction (optionally with subspace projection).

conjgrad_solve
    Standalone conjugate-gradient solver using BART's ``lsqr + iter_conjgrad``
    entirely in C.
"""

from bartorch.lib.encoding import encoding_op
from bartorch.lib.linops import BartLinop
from bartorch.lib.solvers import conjgrad_solve

__all__ = [
    "BartLinop",
    "encoding_op",
    "conjgrad_solve",
]

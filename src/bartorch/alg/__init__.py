"""BART's iterations, driven from here.

Nothing in this package is an algorithm.  Every iteration is BART's own, and
what is here assembles the three things BART's ``lsqr2`` takes -- an encoding,
a set of proximal operators, and which iteration to run -- so that a problem
put together out of :mod:`bartorch.linop` is solved by the same code as the
precompiled tool::

    from bartorch import alg, linop, prox

    A = linop.Sense(kernels, (8, 128, 128), traj=traj, kernels=True)
    x = alg.solve(A, kspace, regularizers=prox.Wavelet(axes=(-1, -2), weight=0.005),
                  solver="fista")

``tests/test_solve.py`` holds that against ``bart pics`` on the same problem
and requires the two to agree exactly, because anything less would mean this
package has an answer of its own.

Two shorter ways at the same computation stay where they are:
:meth:`bartorch.linop.LinearOperator.lstsq` for a plain least-squares solve,
and :func:`bartorch.tools.pics` where the problem is one the tool can name.
"""

from __future__ import annotations

from bartorch.alg.solve import ALGORITHMS, solve

__all__ = ["ALGORITHMS", "solve"]

"""The scaling ``pics`` puts around its solve.

``pics`` does three things to its data before iterating and one of them is a
number: it divides the k-space by an estimate of its own size, so that a
regularization weight means the same thing whatever the data was scaled by
when it arrived.  The estimate is not part of the solve and the tool makes it
outside one, which is why it is a function here rather than an argument to
:func:`bartorch.alg.solve` -- a reconstruction assembled in Python makes it
the same way, and then hands the solver data it has already scaled.

Two estimates, as in ``pics``: from the k-space centre when the encoding is
Cartesian, and from the adjoint image when there is a trajectory.
"""

from __future__ import annotations

import torch

from bartorch._lib import library
from bartorch._operator import as_operand
from bartorch.core.graph import _ensure_ready, _lock

__all__ = ["data_scaling"]


def data_scaling(
    y: torch.Tensor,
    *,
    A=None,
    percentile: float | None = None,
    compat: bool = False,
) -> float:
    """The number ``pics`` divides its data by before solving.

    Parameters
    ----------
    y : torch.Tensor
        The data, already conditioned the way the solve will see it -- which
        for a Cartesian encoding means after the sampling pattern and
        ``tools.fftmod(..., inverse=True)``, because that is where ``pics``
        makes the estimate too.  In the layout BART reads, with the coils on
        their own axis: the shape :func:`bartorch.tools.pics` takes, not the
        squeezed one the operator takes.  The estimate is read off the k-space
        centre and a coil axis in the wrong place is a different centre.
    A : LinearOperator, optional
        The encoding.  Given one, the estimate is the one ``pics`` makes for a
        trajectory: the spread of ``|A^H y|`` read off its order statistics.
        Without one it is the k-space-centre estimate, which is what
        ``bart estscaling`` computes and what ``pics`` uses for a Cartesian
        encoding.
    percentile : float, optional
        Take this percentile of the sorted magnitudes instead of the rule BART
        picks between the 90th and the largest.  ``pics`` has no flag for it;
        ``estscaling -p`` does.
    compat : bool
        Take the median, which is what BART's older estimate did.  Only the
        ``A`` branch has it.

    Returns
    -------
    float
        The scaling.  Zero means the estimate failed, which ``pics`` answers by
        warning and using one.

    Examples
    --------
    The shape of an assembled ``pics``::

        y = bt.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
        scale = alg.data_scaling(y)
        x = alg.solve(A, (y * (1.0 / scale)).squeeze(1), regularizers=...)

    The solution is in the scaled domain, which is where ``pics`` leaves its
    own; multiplying by ``scale`` is what undoes it, and the tool does not.

    Notes
    -----
    ``pics -w`` is the tool's way of supplying a scaling rather than estimating
    one, and ``-w 1`` is how one asks it not to scale at all -- which is what
    makes a solve assembled here comparable to the tool without going through
    this function.
    """
    import bartorch.tools as bt

    if A is None:
        estimate = bt.estscaling(y, **({} if percentile is None else {"percentile": percentile}))
        return float(estimate.reshape(-1)[0].abs())

    adjoint = A.adjoint(y)
    image = as_operand(adjoint, tuple(adjoint.shape), "the adjoint image")
    _ensure_ready()
    with _lock:
        return float(
            library().bartorch_scaling_norm(
                image.numel(),
                image.data_ptr(),
                1.0,
                int(compat),
                -1.0 if percentile is None else float(percentile),
            )
        )

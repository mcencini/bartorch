"""The data scaling ``pics`` applies before it solves."""

from __future__ import annotations

import torch

from bartorch._dispatch import _ensure_ready, _lock
from bartorch._lib import library
from bartorch._operator import as_operand

__all__ = ["data_scaling"]


def data_scaling(
    y: torch.Tensor,
    *,
    A=None,
    percentile: float | None = None,
    compat: bool = False,
) -> float:
    """The number ``pics`` divides its data by before solving.

    Dividing by it makes a regularization weight independent of the data's
    overall scale.

    Parameters
    ----------
    y : torch.Tensor
        Data as the solve will see it: for a Cartesian encoding, after the
        sampling pattern and ``fftmod(..., inverse=True)``.  Coils on their
        own axis, in the layout :func:`bartorch.tools.pics` takes rather than
        the squeezed one an operator takes: the estimate is read off the
        k-space centre, which a misplaced coil axis moves.
    A : LinearOperator, optional
        The encoding.  With one, the estimate ``pics`` makes for a trajectory:
        the spread of ``|A^H y|`` from its order statistics.  Without one,
        the k-space-centre estimate of ``bart estscaling``, which ``pics``
        uses for a Cartesian encoding.
    percentile : float, optional
        Take this percentile of the sorted magnitudes instead of BART's rule
        (``estscaling -p``; ``pics`` has no flag for it).
    compat : bool
        Take the median, as BART's older estimate did.  Only with ``A``.

    Returns
    -------
    float
        The scaling.  Zero means the estimate failed; ``pics`` then warns and
        uses one.

    Notes
    -----
    The solution of a solve on scaled data is scaled too; ``pics`` leaves it
    so.  ``pics -w 1`` disables the scaling, which makes a solve assembled
    without this function comparable to the tool.

    Examples
    --------
    >>> y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    >>> scale = optim.data_scaling(y)
    >>> x = optim.FISTA(prox.Wavelet((-1, -2), 0.01))((y / scale).squeeze(1), A)
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

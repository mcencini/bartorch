"""Normalization of the data before a solve."""

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
    """Normalization factor for the data of a reconstruction.

    Dividing the data by it makes a regularization weight independent of the
    data's overall scale, and is the normalization BART's own reconstructions
    apply before solving.

    Parameters
    ----------
    y : torch.Tensor
        Data as the solve will see it: for a Cartesian encoding, after the
        sampling pattern and ``fftmod(..., inverse=True)``.  Coils must be on
        their own axis in BART's layout rather than the squeezed layout an
        operator takes, because the estimate is read off the k-space centre and
        a misplaced coil axis moves it.
    A : LinearOperator, default=None
        The encoding.  With one, the estimate used for a non-Cartesian
        acquisition: the spread of ``|A^H y|`` from its order statistics.
        Without one, the k-space-centre estimate of
        :func:`bartorch.tools.estscaling`, used for a Cartesian acquisition.
    percentile : float, default=None
        Take this percentile of the sorted magnitudes instead of BART's
        rule.
    compat : bool, default=False
        Take the median, as BART's older estimate did.  Only with ``A``.

    Returns
    -------
    float
        The scaling.  Zero means the estimate failed, and a scaling of one
        should be used instead.

    Notes
    -----
    The solution of a solve on scaled data is scaled by the same factor, and
    is conventionally left so rather than divided back.

    Examples
    --------
    >>> y = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
    >>> scale = optim.data_scaling(y)
    >>> x = optim.FISTA(priors.Wavelet((-1, -2), 0.01))((y / scale).squeeze(1), A)
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

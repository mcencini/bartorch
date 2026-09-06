"""conjgrad_solve — standalone CG solver via BART's lsqr + iter_conjgrad."""

from __future__ import annotations

import torch

from bartorch.lib.linops import BartLinop


def conjgrad_solve(
    op: BartLinop,
    y: torch.Tensor,
    *,
    maxiter: int = 30,
    lam: float = 0.0,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Solve ``(A^H A + lam * I) x = A^H y`` using conjugate gradients.

    The entire CG iteration runs in BART's C library (``lsqr + iter_conjgrad``)
    with no Python callbacks in the inner loop.

    Parameters
    ----------
    op : BartLinop
        The encoding operator ``A``.
    y : torch.Tensor
        Measured data matching ``op.oshape``, complex64.
    maxiter : int, optional
        Maximum number of CG iterations (default 30).
    lam : float, optional
        Tikhonov regularisation weight (default 0.0).
    tol : float, optional
        Convergence tolerance (default 1e-6).

    Returns
    -------
    torch.Tensor
        Reconstructed image with shape ``op.ishape``, complex64.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> import bartorch.lib as bl
    >>> ksp  = bt.phantom([64, 64], kspace=True, ncoils=4)
    >>> sens = bt.ecalib(ksp, calib_size=16, maps=1)
    >>> E    = bl.encoding_op(sens)
    >>> img  = bl.conjgrad_solve(E, ksp, maxiter=15, lam=1e-4)
    """
    return op.solve(y, maxiter=maxiter, lam=lam, tol=tol)

"""BartLinop — persistent BART linear operator Python wrapper."""

from __future__ import annotations

import torch


class BartLinop:
    """Persistent BART linear operator.

    Wraps a ``BartLinopHandle`` from the C++ extension and exposes a
    Pythonic interface for forward/adjoint/normal application and CG solve.

    Instances are created by :func:`bartorch.lib.encoding_op` and are
    **persistent**: the underlying BART operator is constructed once and
    kept alive for the lifetime of this object.

    Parameters
    ----------
    _handle : BartLinopHandle
        Low-level C++ handle returned by ``_bartorch_ext.create_encoding_op``.
    """

    def __init__(self, _handle: object) -> None:
        self._handle = _handle

    @property
    def ishape(self) -> tuple[int, ...]:
        """Domain (input) shape in C-order."""
        return tuple(self._handle.ishape)

    @property
    def oshape(self) -> tuple[int, ...]:
        """Codomain (output) shape in C-order."""
        return tuple(self._handle.oshape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward operator: ``y = A x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor matching :attr:`ishape`, complex64.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :attr:`oshape`, complex64.
        """
        return self._handle.apply(x.to(torch.complex64), 0)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint operator: ``x = A^H y``.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor matching :attr:`oshape`, complex64.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :attr:`ishape`, complex64.
        """
        return self._handle.apply(y.to(torch.complex64), 1)

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normal operator: ``z = A^H A x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor matching :attr:`ishape`, complex64.

        Returns
        -------
        torch.Tensor
            Output tensor with shape :attr:`ishape`, complex64.
        """
        return self._handle.apply(x.to(torch.complex64), 2)

    def solve(
        self,
        y: torch.Tensor,
        *,
        maxiter: int = 30,
        lam: float = 0.0,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """CG solve ``(A^H A + lam * I) x = A^H y`` entirely in C.

        The entire conjugate-gradient iteration runs in BART's C library
        (``lsqr + iter_conjgrad``) with no Python callbacks in the inner loop.

        Parameters
        ----------
        y : torch.Tensor
            Measured data matching :attr:`oshape`, complex64.
        maxiter : int, optional
            Maximum number of CG iterations (default 30).
        lam : float, optional
            Tikhonov regularisation weight (default 0.0).
        tol : float, optional
            Convergence tolerance (default 1e-6).

        Returns
        -------
        torch.Tensor
            Reconstructed image with shape :attr:`ishape`, complex64.
        """
        return self._handle.solve(y.to(torch.complex64), maxiter, lam, tol)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def __repr__(self) -> str:
        return f"BartLinop(ishape={self.ishape}, oshape={self.oshape})"

"""Nonlinear inverse problems by BART's iteratively regularized Gauss-Newton."""

from __future__ import annotations

import torch

from bartorch._dispatch import BartError, _lock, _on_device
from bartorch._lib import library
from bartorch._operator import as_operand

__all__ = ["IRGNM"]


class IRGNM:
    """Iteratively regularized Gauss-Newton for ``F(x) = y``.

    Each step solves the problem linearized at the current point, with a
    Tikhonov term towards the regularization centre, by conjugate gradients;
    the weight is then divided by ``redu``, down to ``alpha_min``.

    Parameters
    ----------
    iterations : int
        Gauss-Newton steps.
    alpha : float
        Initial Tikhonov weight.
    alpha_min : float
        Lower bound of the weight.
    redu : float
        Factor the weight is divided by after each step.
    cg_maxiter : int
        Conjugate-gradient iterations per step.
    cg_tol : float
        Conjugate-gradient tolerance per step.
    """

    def __init__(
        self,
        *,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        redu: float = 2.0,
        cg_maxiter: int = 30,
        cg_tol: float = 0.0,
    ):
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_min)
        self.redu = float(redu)
        self.cg_maxiter = int(cg_maxiter)
        self.cg_tol = float(cg_tol)

    def __call__(
        self, y: torch.Tensor, F, x0: torch.Tensor, xref: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fit ``F(x) = y`` starting from ``x0``.

        Parameters
        ----------
        y : torch.Tensor
            Data of ``F.oshape``.
        F : NonlinearOperator or LinearOperator
            The forward model; a linear one is converted with
            :meth:`~bartorch.linop.LinearOperator.to_nonlinear`.
        x0 : torch.Tensor
            Starting point of ``F.ishape``, and the regularization centre
            unless ``xref`` is given.
        xref : torch.Tensor, optional
            Regularization centre of ``F.ishape``.

        Returns
        -------
        torch.Tensor
            Complex64 solution of ``F.ishape``.
        """
        from bartorch.linop.base import LinearOperator

        if isinstance(F, LinearOperator):
            F = F.to_nonlinear()
        op = F._bart()
        y = as_operand(y, op.oshape, "y")
        x = as_operand(x0, op.ishape, "x0").clone()
        ref = as_operand(xref, op.ishape, "xref") if xref is not None else None
        with _lock, _on_device(op.device or y.device):
            code = library().bartorch_irgnm(
                op._h.ptr,
                self.iterations,
                self.alpha,
                self.alpha_min,
                self.redu,
                self.cg_maxiter,
                self.cg_tol,
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    def __repr__(self) -> str:
        return (
            f"IRGNM(iterations={self.iterations}, alpha={self.alpha}, "
            f"alpha_min={self.alpha_min}, redu={self.redu})"
        )

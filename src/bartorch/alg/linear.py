"""Iterations for a linear problem."""

from __future__ import annotations

import torch

from bartorch.alg.base import Algorithm

__all__ = ["ConjugateGradient"]


def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The real inner product two complex tensors have as real vectors."""
    return torch.vdot(a.flatten(), b.flatten()).real


class ConjugateGradient(Algorithm):
    """Conjugate gradients on the normal equations of ``A x = y``.

    Solves ``(A^H A + lambda) x = A^H y``, applying the normal operator once
    per step -- which for a non-Cartesian encoding is one multiply against a
    point spread function rather than a transform each way.

    BART's own conjugate gradients is :meth:`bartorch.linop.LinearOperator.lstsq`
    and is faster: it never returns to Python between steps.  This one exists
    because it is written in torch, so a fixed number of its steps is a layer.

    Parameters
    ----------
    A : LinearOperator
        The encoding.
    y : torch.Tensor
        The data, of ``A.oshape``.
    lambda_ : float
        Tikhonov weight added to the normal operator.
    x0 : torch.Tensor, optional
        Starting point; without one the iteration starts at zero.
    max_iter, tol : int, float
        As :class:`~bartorch.alg.Algorithm` takes them.

    Examples
    --------
    >>> x = ConjugateGradient(A, kspace, lambda_=0.01, max_iter=30).run()
    """

    def __init__(self, A, y, lambda_: float = 0.0, x0=None, max_iter: int = 30, tol: float = 1e-6):
        self.A = A
        self.lambda_ = float(lambda_)
        x = torch.zeros(A.ishape, dtype=torch.complex64, device=y.device) if x0 is None else x0
        super().__init__(x, max_iter=max_iter, tol=tol)
        self.b = A.A_adjoint(y)
        self.r = self.b - self._normal(self.x)
        self.p = self.r
        self.rr = _inner(self.r, self.r)

    def _normal(self, x: torch.Tensor) -> torch.Tensor:
        normal = self.A.normal(x) if not x.requires_grad else self.A.A_adjoint(self.A(x))
        return normal + self.lambda_ * x if self.lambda_ else normal

    def _step(self) -> None:
        before = self.x
        Ap = self._normal(self.p)
        step = self.rr / torch.clamp(_inner(self.p, Ap), min=1e-30)
        self.x = self.x + step * self.p
        self.r = self.r - step * Ap
        rr = _inner(self.r, self.r)
        self.p = self.r + (rr / torch.clamp(self.rr, min=1e-30)) * self.p
        self.rr = rr
        self._record(before)

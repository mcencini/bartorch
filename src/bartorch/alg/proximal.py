"""Iterations for a problem with a part that is not differentiated."""

from __future__ import annotations

import torch

from bartorch.alg.base import Algorithm
from bartorch.prox.base import Prox

__all__ = ["GradientDescent", "ProximalGradient"]


class GradientDescent(Algorithm):
    """Gradient descent on ``|| A x - y ||^2 / 2``.

    The gradient is ``A^H (A x - y)``, which is one application of the normal
    operator and one of the adjoint against the data.
    """

    def __init__(self, A, y, step: float | None = None, x0=None, max_iter: int = 100, tol=1e-6):
        self.A = A
        self.y = y
        x = torch.zeros(A.ishape, dtype=torch.complex64, device=y.device) if x0 is None else x0
        super().__init__(x, max_iter=max_iter, tol=tol)
        self.step = float(step) if step is not None else max_eigenvalue_step(A, x)
        self.adjoint_y = A.A_adjoint(y)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        normal = self.A.normal(x) if not x.requires_grad else self.A.A_adjoint(self.A(x))
        return normal - self.adjoint_y

    def _step(self) -> None:
        before = self.x
        self.x = self.x - self.step * self._gradient(self.x)
        self._record(before)


class ProximalGradient(Algorithm):
    """Proximal gradient descent on ``|| A x - y ||^2 / 2 + g(x)``.

    ISTA, and FISTA when accelerated.  One application of the normal operator
    and one proximal step per iteration.

    Parameters
    ----------
    A : LinearOperator
        The encoding.
    y : torch.Tensor
        The data.
    g : Prox
        The proximal operator of the penalty; :class:`bartorch.prox.Zero` for
        none, so that one class covers both.
    step : float, optional
        The gradient step.  Without one, an estimate of one over the largest
        eigenvalue of the normal operator, by the power method.
    accelerate : bool
        Take Nesterov's momentum, which is what makes this FISTA rather than
        ISTA.
    x0 : torch.Tensor, optional
        Starting point.
    max_iter, tol : int, float
        As :class:`~bartorch.alg.Algorithm` takes them.

    Examples
    --------
    >>> from bartorch import alg, prox
    >>> g = prox.L1(A.ishape, weight=0.01)
    >>> x = alg.ProximalGradient(A, kspace, g, max_iter=80).run()

    Notes
    -----
    BART's own FISTA is ``bartorch.tools.pics(..., solver="fista")``, and it is
    faster.  This one is a torch graph: unrolled over a few steps with a
    learned ``g``, it is the body of a reconstruction network.
    """

    def __init__(
        self,
        A,
        y,
        g: Prox,
        step: float | None = None,
        accelerate: bool = True,
        x0=None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.A = A
        self.y = y
        self.g = g
        self.accelerate = bool(accelerate)
        x = torch.zeros(A.ishape, dtype=torch.complex64, device=y.device) if x0 is None else x0
        super().__init__(x, max_iter=max_iter, tol=tol)
        self.step = float(step) if step is not None else max_eigenvalue_step(A, x)
        self.adjoint_y = A.A_adjoint(y)
        self.z = self.x
        self.t = 1.0

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        normal = self.A.normal(x) if not x.requires_grad else self.A.A_adjoint(self.A(x))
        return normal - self.adjoint_y

    def _step(self) -> None:
        before = self.x
        stepped = self.z - self.step * self._gradient(self.z)
        self.x = self.g(self.step, stepped)
        if self.accelerate:
            t = (1.0 + (1.0 + 4.0 * self.t * self.t) ** 0.5) / 2.0
            self.z = self.x + ((self.t - 1.0) / t) * (self.x - before)
            self.t = t
        else:
            self.z = self.x
        self._record(before)


def max_eigenvalue_step(A, like: torch.Tensor, iterations: int = 15) -> float:
    """One over the largest eigenvalue of ``A^H A``, by the power method.

    The step a gradient method can take without diverging.  Estimated rather
    than asked for, because an operator does not know its own norm; a few
    applications of the normal operator are cheap beside the iteration it
    makes safe.
    """
    with torch.no_grad():
        v = torch.randn(A.ishape, dtype=torch.complex64, device=like.device)
        v = v / v.flatten().norm()
        value = torch.tensor(1.0, device=like.device)
        for _ in range(iterations):
            w = A.normal(v)
            value = w.flatten().norm()
            if value <= 0:
                return 1.0
            v = w / value
    return float(1.0 / value)

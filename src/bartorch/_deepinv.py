"""The ``deepinv`` adapter, :func:`to_deepinv`.

``deepinv`` is imported on first use rather than at import time, so a script
that only builds operators does not pay for it.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["to_deepinv"]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """``apply`` over a leading batch axis, when ``x`` has one more axis than ``shape``.

    The batch is walked rather than folded into the operator, whose data
    (trajectory, sensitivities) does not repeat over it.
    """
    if x.ndim == len(shape) + 1:
        return torch.stack([apply(item) for item in x])
    return apply(x)


@functools.lru_cache(maxsize=1)
def _physics_class() -> type:
    """The ``LinearPhysics`` subclass, built once, when ``deepinv`` is present."""
    try:
        from deepinv.physics import LinearPhysics
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "bartorch.to_deepinv() hands an operator over as a deepinv "
            "LinearPhysics, and deepinv is a dependency of this package -- an "
            "environment without it is a broken one rather than a lean one"
        ) from exc

    class BartPhysics(LinearPhysics):
        """A ``deepinv.physics.LinearPhysics`` over a bartorch operator.

        ``A`` and ``A_adjoint`` are the operator's and differentiate through its
        autograd functions; ``A_dagger`` is :class:`bartorch.optim.CG`.

        Parameters
        ----------
        op : LinearOperator
        maxiter, lambda_, tol : int, float, float
            ``A_dagger``'s defaults, overridable per call.
        **kwargs
            Passed to ``LinearPhysics``, for instance a ``noise_model``.
        """

        def __init__(self, op, maxiter: int = 30, lambda_: float = 0.0, tol: float = 1e-6, **kw):
            super().__init__(
                A=lambda x, **_: _batched(op, x, op.ishape),
                A_adjoint=lambda y, **_: _batched(op.A_adjoint, y, op.oshape),
                **kw,
            )
            self.op = op
            self.maxiter = maxiter
            self.lambda_ = lambda_
            self.tol = tol

        def A_dagger(self, y, **kwargs):  # noqa: N802
            """Least-squares solution by :class:`bartorch.optim.CG`."""
            from bartorch.optim import CG

            solver = CG(
                lambda_=kwargs.pop("lambda_", self.lambda_),
                maxiter=kwargs.pop("maxiter", self.maxiter),
                tol=kwargs.pop("tol", self.tol),
            )
            return _batched(lambda item: solver(item, self.op), y, self.op.oshape)

        def __repr__(self) -> str:
            return f"BartPhysics({self.op!r})"

    return BartPhysics


def to_deepinv(op, **kwargs):
    """A :class:`~bartorch.linop.LinearOperator` as a ``deepinv.physics.LinearPhysics``.

    A tensor with one more axis than the operator's shape is treated as a
    batch and applied item by item.  ``A_dagger`` solves by
    :class:`bartorch.optim.CG`.

    Parameters
    ----------
    op : LinearOperator
        Kept alive as long as the physics.
    **kwargs
        ``maxiter``, ``lambda_`` and ``tol`` for ``A_dagger``, overridable per
        call, and anything ``LinearPhysics`` takes.

    Returns
    -------
    deepinv.physics.LinearPhysics

    Examples
    --------
    >>> physics = bartorch.to_deepinv(linop.NoncartesianSense(maps, (128, 128), traj=traj))
    >>> physics.A_dagger(kspace[None]).shape       # a batch of one
    torch.Size([1, 128, 128])
    """
    return _physics_class()(op, **kwargs)

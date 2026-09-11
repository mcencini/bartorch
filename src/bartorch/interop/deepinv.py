"""bartorch operators as ``deepinv`` physics.

``deepinv`` is not a dependency of this package, and a
:class:`~bartorch.linop.LinearOperator` is not a ``LinearPhysics``.  Making it
one would put that import in the path of every operator and tie this package's
releases to theirs, so what an operator carries instead is the same three
methods under ``deepinv``'s names -- ``A``, ``A_adjoint``, ``A_dagger`` -- and
:func:`as_physics` for when a real subclass is wanted.

The one thing that has to be translated is the batch: ``deepinv`` works on
tensors with an axis in front for the examples in a batch, which a BART
operator does not have, so the adapter walks it.

The class is built the first time it is asked for, so importing this module
without ``deepinv`` installed is not an error.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["as_physics"]


def _batched(apply, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """*apply* over a leading batch axis, if the tensor has one.

    An operator's own shape is fixed, so anything with one axis more than the
    operator expects is a batch of them.  The axis is walked rather than
    folded into the operator, because an operator carries a trajectory and a
    bank of sensitivities that the batch does not multiply.
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
            "deepinv is required for LinearOperator.to_deepinv(); "
            "install it with `pip install 'bartorch[deepinv]'`"
        ) from exc

    class BartPhysics(LinearPhysics):
        """A ``deepinv.physics.LinearPhysics`` over a bartorch operator.

        ``A`` and ``A_adjoint`` are the operator's, so the encoding runs inside
        BART and differentiates through :mod:`bartorch.linop.autograd`;
        ``A_dagger`` is BART's conjugate gradients on the normal equations,
        which for a non-Cartesian operator applies the normal as one multiply
        against a point spread function and never returns to Python between
        iterations.

        Parameters
        ----------
        op : LinearOperator
            What the physics is.
        maxiter, lambda_, tol : int, float, float
            What ``A_dagger`` solves with by default; each can be overridden
            per call.
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
            """The least-squares solution, by BART's conjugate gradients."""
            solve = functools.partial(
                self.op.lstsq,
                lambda_=kwargs.pop("lambda_", self.lambda_),
                maxiter=kwargs.pop("maxiter", self.maxiter),
                tol=kwargs.pop("tol", self.tol),
            )
            return _batched(solve, y, self.op.oshape)

        def __repr__(self) -> str:
            return f"BartPhysics({self.op!r})"

    return BartPhysics


def as_physics(op, **kwargs):
    """Wrap a :class:`~bartorch.linop.LinearOperator` as a ``LinearPhysics``.

    Parameters
    ----------
    op : LinearOperator
        The operator to hand over.  It stays alive as long as the physics.
    **kwargs
        ``maxiter``, ``lambda_`` and ``tol`` for ``A_dagger``, and anything
        ``LinearPhysics`` itself takes.

    Returns
    -------
    deepinv.physics.LinearPhysics

    Examples
    --------
    >>> A = Sense(maps, (8, 128, 128), traj=traj)
    >>> physics = A.to_deepinv()
    >>> physics.A_dagger(kspace[None]).shape       # a batch of one
    torch.Size([1, 1, 128, 128])
    """
    return _physics_class()(op, **kwargs)

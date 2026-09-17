"""Projection onto convex sets: ``italgos.c``'s ``pocs``, and the sweep it repeats."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import torch
from torch import nn

from bartorch.optim.blocks import _prox

__all__ = ["POCS", "POCSBlock"]


def _projections(projections: Iterable) -> list:
    """The sets to project onto, each as a callable on a tensor.

    A :mod:`bartorch.priors` term is taken as its proximal operator at
    ``mu = 1``, which is what ``pocs`` calls a projection with.
    """
    made = []
    for projection in projections:
        if hasattr(projection, "prox"):
            term = projection
            made.append(lambda x, term=term: _prox(term, x, 1.0, tuple(x.shape)))
        elif callable(projection):
            made.append(projection)
        else:
            raise TypeError(
                f"a projection is a callable on a tensor or a bartorch.priors term, "
                f"not {type(projection).__name__}"
            )
    if not made:
        raise ValueError("a sweep of no projections computes nothing")
    return made


class POCSBlock(nn.Module):
    """One sweep of the projections, ``italgos.c``'s ``pocs``.

    Every projection is applied in turn, in place, and the sweep is the whole
    of a step: ``pocs`` takes no step size, keeps no momentum and reads no
    residual, so the state is the iterate alone.  Repeating the sweep is the
    method, which is what :class:`POCS` does.

    The projections carry the data and the encoding -- ``pocsense``'s are the
    measured samples, the range of the coil sensitivities and a sparsity
    threshold -- so ``A`` is not read.  It is accepted, and ignored, so that
    the block has the calling convention the others have and
    :class:`bartorch.learning.Unrolled` can stack it.

    Parameters
    ----------
    projections : sequence
        The sets to project onto, applied in the order given.  Each is a
        callable mapping a tensor to a tensor, or a :mod:`bartorch.priors`
        term, which is taken as its proximal operator at ``mu = 1``.

    Examples
    --------
    >>> block = POCSBlock([consistency, sense, sparsity])
    >>> state = block.start(kspace)
    >>> state = block(state)
    >>> block.output(state)
    """

    @dataclasses.dataclass(frozen=True)
    class State:
        x: torch.Tensor
        k: int = 0

    def __init__(self, projections: Sequence):
        super().__init__()
        self.projections = _projections(projections)
        modules = [p for p in projections if isinstance(p, nn.Module)]
        if modules:
            self.held = nn.ModuleList(modules)

    def start(self, y: torch.Tensor, A=None, x0: torch.Tensor | None = None) -> State:
        """The run's state.

        ``pocs_recon2`` clears its result and lets the first projection put
        the data in, so a run without ``x0`` starts at zero of ``y``'s shape.
        """
        y = torch.as_tensor(y)
        if x0 is None:
            return self.State(torch.zeros_like(y))
        return self.State(torch.as_tensor(x0).clone())

    def forward(self, state: State, A=None) -> State:
        x = state.x
        for project in self.projections:
            x = project(x)
        return dataclasses.replace(state, x=x, k=state.k + 1)

    def output(self, state: State, A=None) -> torch.Tensor:
        """The iterate.  ``pocs`` leaves it where the last projection put it."""
        return state.x


class POCS:
    """Projection onto convex sets, looping :class:`POCSBlock`.

    ``maxiter`` sweeps of the projections, each applied in turn.  There is no
    stopping test: ``pocs`` runs its count out.

    Parameters
    ----------
    projections : sequence
        The sets to project onto, as :class:`POCSBlock` takes them.
    maxiter : int
        Sweeps; ``pocsense``'s default is fifty.

    Examples
    --------
    >>> POCS([consistency, sense, sparsity], maxiter=50)(kspace)
    """

    def __init__(self, projections: Sequence, *, maxiter: int = 50):
        self.projections = list(projections)
        self.maxiter = int(maxiter)

    def _block(self) -> POCSBlock:
        return POCSBlock(self.projections)

    def __call__(self, y: torch.Tensor, A=None, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Sweep the projections, starting from ``x0`` or from zero of ``y``'s shape."""
        block = self._block()
        state = block.start(y, A, x0)
        for _ in range(self.maxiter):
            state = block(state, A)
        return block.output(state, A)

"""The iteration every algorithm here is."""

from __future__ import annotations

import abc

import torch

__all__ = ["Algorithm"]


class Algorithm(abc.ABC):
    """An iteration, one step at a time.

    Written as a step rather than as a loop so that the loop belongs to the
    caller: :meth:`run` is the usual one, and a network that unrolls a fixed
    number of steps drives :meth:`update` itself.

    Attributes
    ----------
    x : torch.Tensor
        The current iterate.
    iteration : int
        How many steps have been taken.
    max_iter : int
        How many :meth:`run` will take at most.
    tol : float
        The relative change at which :meth:`done` reports convergence.
    """

    def __init__(self, x: torch.Tensor, max_iter: int = 100, tol: float = 1e-6):
        self.x = x
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.iteration = 0
        #: The relative change of the last step, which :meth:`done` reads.
        self.residual = float("inf")

    def update(self) -> None:
        """Take one step, leaving the result in :attr:`x`, and count it."""
        self._step()
        self.iteration += 1

    @abc.abstractmethod
    def _step(self) -> None:
        """One step.  Every concrete algorithm is this and nothing else."""

    def done(self) -> bool:
        """Whether the iteration has converged or run out of steps."""
        return self.iteration >= self.max_iter or self.residual <= self.tol

    def run(self) -> torch.Tensor:
        """Iterate until :meth:`done`, and return the iterate.

        Under ``torch.no_grad`` this is a solver; with a gradient wanted it is
        an unrolled network of :attr:`max_iter` steps, so keep that number
        small when the graph is going to be kept.
        """
        while not self.done():
            self.update()
        return self.x

    def _record(self, before: torch.Tensor) -> None:
        """Note how far the last step moved, relative to where it started."""
        with torch.no_grad():
            moved = (self.x - before).detach().flatten().norm()
            scale = torch.clamp(before.detach().flatten().norm(), min=1e-30)
            self.residual = float(moved / scale)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(iteration={self.iteration}, residual={self.residual:.3g})"

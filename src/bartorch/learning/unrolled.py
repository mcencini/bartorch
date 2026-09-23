"""Unrolled networks formed by repeated application of an iteration block."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ["Unrolled"]


def _fields(state) -> list[str]:
    """Names of the state fields holding a tensor, or a non-empty tuple of tensors."""
    names = []
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        if isinstance(value, torch.Tensor) or (
            isinstance(value, tuple)
            and 0 != len(value)
            and all(isinstance(t, torch.Tensor) for t in value)
        ):
            names.append(field.name)
    return names


def _tensors(state, names) -> list[torch.Tensor]:
    made: list[torch.Tensor] = []
    for name in names:
        value = getattr(state, name)
        made.extend([value] if isinstance(value, torch.Tensor) else list(value))
    return made


def _restored(state, names, values):
    """``state`` with its tensor fields replaced by ``values``, in ``_tensors`` order."""
    fields, at = {}, 0
    for name in names:
        value = getattr(state, name)
        count = 1 if isinstance(value, torch.Tensor) else len(value)
        taken = list(values[at : at + count])
        at += count
        fields[name] = taken[0] if isinstance(value, torch.Tensor) else tuple(taken)
    return dataclasses.replace(state, **fields)


class Unrolled(nn.Module):
    """Unrolled network formed by applying an iteration block ``iterations`` times.

    The block implements one step of a BART iteration and this module
    implements the loop over it.  With every parameter frozen the stack
    reproduces the corresponding solver bit for bit; calling
    ``requires_grad_()`` on a step size, a penalty weight or a denoiser's
    weights makes the stack a trainable network.  MoDL is this module applied
    to an :class:`~bartorch.optim.ADMMBlock` whose single term is a
    :class:`~bartorch.priors.ImplicitPrior`.

    A single block shared by every iteration gives the weight sharing usual in
    unrolled networks; a sequence of blocks gives each iteration its own
    parameters.

    ``detach`` and ``checkpoint`` select how the backward pass is taken.
    Neither alters the value the network computes:

    * end to end, the default, records the whole stack.  Memory grows as the
      iteration count times the storage of one step.
    * ``detach=True`` starts each iteration from a detached state, so that the
      graph never spans two iterations.  Combined with a loss on each image
      yielded by :meth:`steps`, this is greedy per-iteration training, whose
      memory is independent of the iteration count.
    * ``checkpoint=True`` retains the states between iterations and recomputes
      the interior of a step during the backward pass.  The gradient is the
      end-to-end one and each block is applied twice.

    These correspond to the stages in which a large unrolled network is
    trained: a denoiser pretrained in isolation, then greedy per-iteration
    training, then end-to-end fine-tuning with gradient checkpointing.
    :class:`bartorch.optim.FixedPoint` is an alternative with bounded memory:
    it drives the block to its fixed point and differentiates there, with no
    iteration count to unroll.

    Parameters
    ----------
    block : nn.Module or sequence of nn.Module
        One of :mod:`bartorch.optim`'s iteration blocks, or a sequence of
        them, one per iteration.
    iterations : int, default=None
        Number of applications of a single shared block.  Omitted for a
        sequence, whose length gives the count.
    detach : bool, default=False
        Whether each iteration starts from a detached state.
    checkpoint : bool, default=False
        Whether the interior of an iteration is recomputed during the backward
        pass rather than stored.

    Notes
    -----
    Checkpointing recomputes a step, and the gradient is correct only if the
    recomputation reproduces it.  PyTorch's random state is restored for the
    recomputation, so dropout in a denoiser is reproduced.  A BART term that
    draws random shifts from BART's own generator, such as
    :class:`~bartorch.priors.Wavelet` or
    :class:`~bartorch.priors.LocallyLowRank` with ``randshift=True``, draws
    new shifts, and the gradient then does not correspond to the forward
    pass.  This is not checked: ``checkpoint=True`` is valid only for steps
    that are deterministic or draw from PyTorch's generator.

    Examples
    --------
    >>> block = optim.ADMMBlock(priors.ImplicitPrior(denoiser), rho=0.05, cg_maxiter=10)
    >>> block.rho.requires_grad_()
    >>> model = learning.Unrolled(block, iterations=10, checkpoint=True)
    >>> model(kspace, A).abs().sub(target).square().mean().backward()
    """

    def __init__(
        self,
        block,
        iterations: int | None = None,
        *,
        detach: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()
        shared = isinstance(block, nn.Module) and not isinstance(
            block, (nn.ModuleList, nn.Sequential)
        )
        if shared:
            if iterations is None:
                raise ValueError("one block is shared by every iteration, so say how many")
            self.blocks = nn.ModuleList([block])
            self.iterations = int(iterations)
        else:
            self.blocks = nn.ModuleList(block)
            if 0 == len(self.blocks):
                raise ValueError("a stack of no blocks computes nothing")
            if iterations is not None and int(iterations) != len(self.blocks):
                raise ValueError(
                    f"a block per iteration is {len(self.blocks)} of them, not {int(iterations)}"
                )
            self.iterations = len(self.blocks)
        if self.iterations < 1:
            raise ValueError(f"a stack runs at least one iteration, not {self.iterations}")
        self.detach = bool(detach)
        self.checkpoint = bool(checkpoint)

    def block(self, k: int) -> nn.Module:
        """The block applied at iteration ``k``: the shared one, or its own."""
        return self.blocks[k % len(self.blocks)]

    def steps(self, y: torch.Tensor, A, x0: torch.Tensor | None = None):
        """Yield the image after each iteration in turn.

        A loss taken on each yielded image, with ``detach`` set, gives greedy
        per-iteration training.  :meth:`forward` returns the last of them.
        """
        state = self.blocks[0].start(y, A, x0)
        names = _fields(state)
        for k in range(self.iterations):
            block = self.block(k)
            if self.detach:
                state = _restored(state, names, [t.detach() for t in _tensors(state, names)])
            state = self._step(block, state, A, names)
            yield block.output(state, A)

    def forward(self, y: torch.Tensor, A, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct ``y``, returning the image left by the last iteration.

        Parameters
        ----------
        y : torch.Tensor
            Measured data, with or without a leading batch axis.
        A : LinearOperator
            Encoding operator, shared by every item of a batch.
        x0 : torch.Tensor, default=None
            Starting point of the iteration; zero by default, as in BART.
        """
        image = None
        for image in self.steps(y, A, x0):
            pass
        return image

    def _step(self, block, state, A, names):
        """One application of ``block``, checkpointed where that was requested."""
        tensors = _tensors(state, names)
        learning = any(p.requires_grad for p in block.parameters())
        if not (
            self.checkpoint
            and torch.is_grad_enabled()
            and (learning or any(t.requires_grad for t in tensors))
        ):
            return block(state, A)

        made: dict = {}

        def run(*values):
            step = block(_restored(state, names, values), A)
            made["state"] = step
            return tuple(_tensors(step, names))

        out = checkpoint(run, *tensors, use_reentrant=False)
        return _restored(made["state"], names, out)

"""An iteration block applied a fixed number of times, as a network to train."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ["Unrolled"]


def _fields(state) -> list[str]:
    """The state's fields holding a tensor, or a non-empty tuple of them."""
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
    """``state`` with its tensor fields read back from ``values``, in ``_tensors``' order."""
    fields, at = {}, 0
    for name in names:
        value = getattr(state, name)
        count = 1 if isinstance(value, torch.Tensor) else len(value)
        taken = list(values[at : at + count])
        at += count
        fields[name] = taken[0] if isinstance(value, torch.Tensor) else tuple(taken)
    return dataclasses.replace(state, **fields)


class Unrolled(nn.Module):
    """A block from :mod:`bartorch.optim` applied ``iterations`` times, as a network.

    The block is the iteration's step and the loop is this module, so what is
    unrolled is BART's own iteration: with every parameter frozen the stack
    reproduces the solver it was built from, and ``requires_grad_()`` on a step
    size, a penalty weight or a denoiser's weights is what makes it a network.
    MoDL is this module over an :class:`~bartorch.optim.ADMMBlock` whose only
    term is a :class:`~bartorch.priors.ImplicitPrior`.

    One block shared by every iteration is the weight sharing an unrolled
    network usually means; a sequence of blocks gives each iteration its own.

    ``detach`` and ``checkpoint`` are what makes a deep stack trainable, and
    neither changes what the network computes:

    * end to end, the default, records the whole stack.  Memory grows with the
      iteration count times what one step holds.
    * ``detach=True`` starts each iteration from a detached state, so the graph
      never spans two of them.  Together with a loss on each of :meth:`steps`'
      images this is greedy per-iteration training, whose memory does not grow
      with the count at all.
    * ``checkpoint=True`` keeps the states between iterations and recomputes a
      step's interior in the backward pass.  The gradient is the end-to-end
      one; each block is applied twice.

    The three are the stages a large unrolled network is trained in --
    a denoiser pretrained on its own, then per iteration, then the whole stack
    fine-tuned with checkpointing -- and are set by construction rather than
    by rebuilding the model.
    :class:`bartorch.optim.FixedPoint` is the other route to a bounded memory:
    it drives the block to its fixed point and differentiates there, with no
    iteration count to unroll.

    Parameters
    ----------
    block : nn.Module or sequence of nn.Module
        One of :mod:`bartorch.optim`'s blocks, or several.
    iterations : int, optional
        How many times a single block is applied; the length of a sequence,
        which is then not given here.
    detach : bool
        Whether each iteration starts from a detached state.
    checkpoint : bool
        Whether an iteration's interior is recomputed in the backward pass
        rather than stored.  A term drawing its own random shifts -- a wavelet
        threshold -- draws again on the recomputation, so checkpointing is for
        a stack whose steps are deterministic, which a denoiser's is.

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
        """The block iteration ``k`` applies: the shared one, or its own."""
        return self.blocks[k % len(self.blocks)]

    def steps(self, y: torch.Tensor, A, x0: torch.Tensor | None = None):
        """The image after each iteration, yielded in turn.

        A loss on each of these is what greedy per-iteration training is;
        :meth:`forward` is the last of them.
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
        """The image the last iteration leaves.

        Parameters
        ----------
        y : torch.Tensor
            The measured data, with a leading batch axis or without one.
        A : LinearOperator
            The encoding, shared by every item of a batch.
        x0 : torch.Tensor, optional
            Where the run starts; zero by default, where BART starts it.
        """
        image = None
        for image in self.steps(y, A, x0):
            pass
        return image

    def _step(self, block, state, A, names):
        """One application, recomputed in the backward pass where that was asked for."""
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

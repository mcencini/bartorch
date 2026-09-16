"""Nonlinear operators defined by Python functions.

One argument or many.  A function of several tensors becomes an ``nlop`` of
several inputs, which lets a denoiser's weights be an *argument* of a
BART graph rather than something the function closed over -- and so what lets
a gradient reach them when the graph is BART's to apply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import (
    Built,
    Shape,
    callback,
    dims,
    generic_callback,
    pair_callback,
)
from bartorch.nlop.base import NonlinearOperator, _built

__all__ = ["TorchOperator", "Parameters"]


def _shapes(shape) -> tuple[Shape, ...]:
    """``shape`` as a tuple of shapes, whether it was one or several.

    ``(4,)`` is one shape and ``[(4,), ()]`` is two; ``()`` is the scalar
    shape rather than no arguments at all, because an operator has at least
    one of each.
    """
    if not isinstance(shape, Sequence):
        raise TypeError(f"a shape is a tuple of ints, or a sequence of those, not {shape!r}")
    if all(isinstance(s, int) for s in shape):
        return (tuple(shape),)
    return tuple(tuple(s) for s in shape)


def _flat(shapes: Sequence[Shape]) -> list[int]:
    """Every shape's dimension vector, laid end to end at DIMS."""
    out: list[int] = []
    for shape in shapes:
        out.extend(dims(shape))
    return out


class _Callback(NonlinearOperator):
    """A nonlinear operator implemented by Python functions on tensors; see
    :meth:`NonlinearOperator.from_callbacks`.

    ``forward`` evaluates the operator and fixes the point at which
    ``derivative`` and ``adjoint`` are taken until the next forward call,
    which is how BART's solvers use them.

    Parameters
    ----------
    oshape, ishape : tuple of int, or a sequence of them
        Codomain and domain shapes, C order.  A sequence of shapes makes an
        operator of that many outputs or inputs.
    forward : callable
        ``forward(x)`` for one input, ``forward(*xs)`` for several; it returns
        one tensor per output.  It fixes the linearisation point.
    derivative, adjoint : callable
        With one argument each way, ``derivative(dx)`` and ``adjoint(dy)``.
        With several, ``derivative(o, i, dx)`` and ``adjoint(o, i, dy)``: the
        derivative of output ``o`` by input ``i``, and the adjoint of it.

    Examples
    --------
    >>> NonlinearOperator.from_callbacks(
    ...     (4,), (4,), lambda x: 2 * x, lambda d: 2 * d, lambda v: 2 * v
    ... )

    Two inputs, so that the second can carry a weight a gradient reaches:

    >>> NonlinearOperator.from_callbacks((4,), [(4,), ()], scale, d_scale, adj_scale)
    """

    def __init__(
        self,
        oshape,
        ishape,
        forward: Callable,
        derivative: Callable,
        adjoint: Callable,
    ):
        self.oshapes, self.ishapes = _shapes(oshape), _shapes(ishape)
        self.forward_fn = forward
        self.derivative_fn = derivative
        self.adjoint_fn = adjoint
        super().__init__()

    @property
    def _many(self) -> bool:
        return 1 != len(self.ishapes) or 1 != len(self.oshapes)

    def _create(self) -> Built:
        if not self._many:
            return self._create_one()

        fwd = generic_callback(self.forward_fn, self.oshapes, self.ishapes, "forward")
        der = pair_callback(self.derivative_fn, self.oshapes, self.ishapes, "derivative")
        adj = pair_callback(self.adjoint_fn, self.oshapes, self.ishapes, "adjoint", adjoint=True)
        ptr = self._under_lock(
            library().bartorch_nlop_callback_generic,
            len(self.oshapes),
            DIMS,
            _marshal.longs(_flat(self.oshapes)),
            len(self.ishapes),
            DIMS,
            _marshal.longs(_flat(self.ishapes)),
            fwd,
            der,
            adj,
            None,
            _marshal.null_release(),
        )
        keep = (fwd, der, adj, self.forward_fn, self.derivative_fn, self.adjoint_fn)
        return _built(ptr, self.ishapes, self.oshapes, keep=keep)

    def _create_one(self) -> Built:
        ishape, oshape = self.ishape, self.oshape
        fwd = callback(self.forward_fn, ishape, oshape, "forward")
        der = callback(self.derivative_fn, ishape, oshape, "derivative")
        adj = callback(self.adjoint_fn, oshape, ishape, "adjoint")
        ptr = self._under_lock(
            library().bartorch_nlop_callback,
            DIMS,
            dims(oshape),
            DIMS,
            dims(ishape),
            fwd,
            der,
            adj,
            None,
            _marshal.null_release(),
        )
        keep = (fwd, der, adj, self.forward_fn, self.derivative_fn, self.adjoint_fn)
        return Built(ptr, ishape, oshape, keep=keep)


class TorchOperator(_Callback):
    """A nonlinear operator from a differentiable torch function.

    The derivative is torch's forward-mode Jacobian-vector product, and its
    adjoint the reverse-mode vector-Jacobian product, both at the last
    evaluated point.

    Parameters
    ----------
    fn : callable
        A differentiable function on tensors.  With several input shapes it
        takes that many arguments, and it may return several tensors.
    ishape, oshape : tuple of int, or a sequence of them
        Domain and codomain, C order.

    Examples
    --------
    >>> F = TorchOperator(lambda p: p[0] * torch.exp(-t / p[1]), (2,), t.shape)
    >>> nlop.IRGNM()(measured, F, x0=torch.tensor([1.0, 20.0]))

    A denoiser whose weights are an argument, so that they train through a
    graph BART applies:

    >>> prior = TorchOperator(lambda x, w: w * x, [state, ()], state)
    >>> nlop.chain(cell, prior, output=0, input=0)

    Notes
    -----
    A real argument is carried in the real part of a complex one: BART's
    operators are complex throughout, so a weight arrives as ``w + 0j`` and
    its gradient comes back complex.  Take the real part of it.
    """

    def __init__(self, fn: Callable, ishape, oshape):
        self.fn = fn
        ishapes, oshapes = _shapes(ishape), _shapes(oshape)
        state: dict = {}

        def forward(*xs: torch.Tensor):
            tracked = tuple(x.detach().clone().requires_grad_(True) for x in xs)
            with torch.enable_grad():
                made = fn(*tracked)
            made = (made,) if isinstance(made, torch.Tensor) else tuple(made)
            state["xs"], state["ys"] = tracked, made
            return tuple(y.detach() for y in made) if 1 != len(made) else made[0].detach()

        def derivative(o: int, i: int, dx: torch.Tensor) -> torch.Tensor:
            # The tangent is zero in every argument but `i`, which is what
            # makes this the derivative of output `o` by input `i` alone.
            xs = tuple(x.detach() for x in state["xs"])
            tangents = tuple(
                dx.detach().clone() if at == i else torch.zeros_like(x) for at, x in enumerate(xs)
            )
            with torch.enable_grad():
                _, jvp = torch.func.jvp(fn, xs, tangents)
            jvp = (jvp,) if isinstance(jvp, torch.Tensor) else tuple(jvp)
            return jvp[o]

        def adjoint(o: int, i: int, dy: torch.Tensor) -> torch.Tensor:
            xs, ys = state["xs"], state["ys"]
            with torch.enable_grad():
                (g,) = torch.autograd.grad(
                    ys[o],
                    xs[i],
                    grad_outputs=dy.detach().clone(),
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=True,
                )
            return g

        if 1 == len(ishapes) == len(oshapes):
            # The one-argument forms, so that an operator of one input is the
            # same operator it was before any of this.
            super().__init__(
                oshapes[0],
                ishapes[0],
                lambda x: forward(x),
                lambda dx: derivative(0, 0, dx),
                lambda dy: adjoint(0, 0, dy),
            )
        else:
            super().__init__(oshapes, ishapes, forward, derivative, adjoint)

    def _bundle(self):
        from bartorch.nlop.bundle import from_torch

        return from_torch(self, self.fn)


class Parameters:
    """A module's parameters as one argument of an operator.

    :class:`TorchOperator` differentiates by its *arguments*, so a denoiser whose
    weights are to be trained inside a graph BART applies has to take them
    rather than close over them.  A module keeps its parameters as several
    real tensors of several shapes; this is the one complex vector BART can
    carry, and the way back.

    The weights ride in the real part.  BART's operators are complex
    throughout, so the vector is complex64 and its imaginary half is an exact
    null direction: nothing reads it, and the returned gradient is complex with
    the answer in its real part.  An optimizer over the packed vector therefore
    works directly on it, so the packed vector is the object to hold as a
    ``torch.nn.Parameter``.

    Parameters
    ----------
    module : torch.nn.Module
        Read for the names and shapes of its parameters.  It is not kept
        differentiably: what the operator differentiates is the vector.

    Attributes
    ----------
    shape : tuple of int
        What to give :class:`TorchOperator` as the weights' shape.

    Examples
    --------
    >>> weights = nlop.Parameters(denoiser)
    >>> prior = nlop.TorchOperator(
    ...     lambda x, w: torch.func.functional_call(denoiser, weights.unpack(w), (x,)),
    ...     [state, weights.shape],
    ...     state,
    ... )
    >>> trained = torch.nn.Parameter(weights.pack())
    >>> torch.optim.Adam([trained], lr=1e-3)

    and afterwards ``weights.load(trained)`` puts them back in the module.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.names = [name for name, _ in module.named_parameters()]
        self.shapes = [tuple(p.shape) for _, p in module.named_parameters()]
        self.sizes = [p.numel() for _, p in module.named_parameters()]
        if not self.names:
            raise ValueError(f"{type(module).__name__} has no parameters to train")
        self.shape: Shape = (sum(self.sizes),)

    def pack(self) -> torch.Tensor:
        """The module's parameters as they stand, as one complex vector."""
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in self.module.parameters()])
        return flat.to(torch.complex64)

    def unpack(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """``weights`` as the mapping ``torch.func.functional_call`` takes."""
        if weights.numel() != self.shape[0]:
            raise ValueError(
                f"{type(self.module).__name__} has {self.shape[0]} parameters, "
                f"not {weights.numel()}"
            )
        flat = weights.reshape(-1)
        flat = flat.real if flat.is_complex() else flat
        out, at = {}, 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            out[name] = flat[at : at + size].reshape(shape)
            at += size
        return out

    def load(self, weights: torch.Tensor) -> None:
        """Write ``weights`` back into the module, after training."""
        made = self.unpack(weights.detach())
        with torch.no_grad():
            for name, parameter in self.module.named_parameters():
                parameter.copy_(made[name].to(parameter.dtype))

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"Parameters({type(self.module).__name__}, {self.shape[0]})"

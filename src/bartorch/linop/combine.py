"""Putting operators side by side: the only functions in :mod:`bartorch.linop`.

:func:`stack`, :func:`concatenate`, :func:`block_diag` and :func:`block` each
build one ``linop_stack_cod`` or ``linop_stack``, so a stack is a single
operator that a solver drives in its own loop rather than a Python list walked
per iteration.
"""

from __future__ import annotations

from collections.abc import Sequence

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import Built
from bartorch.linop.base import LinearOperator

__all__ = ["block", "block_diag", "concatenate", "hstack", "stack"]


def _operators(ops, what: str) -> list[LinearOperator]:
    ops = list(ops)
    if not ops:
        raise ValueError(f"{what} needs at least one operator")
    for op in ops:
        if not isinstance(op, LinearOperator):
            raise TypeError(f"{what} takes linear operators, got {type(op).__name__}")
    return ops


def _bart_dim(axis: int, ndim: int) -> int:
    """A C-order axis as BART's dimension index."""
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for {ndim} axes")
    return ndim - 1 - (axis % ndim)


class _StackCodomain(LinearOperator):
    """``linop_stack_cod``: one domain, codomains laid end to end."""

    def __init__(self, ops: Sequence[LinearOperator], bart_dim: int, oshape, ishape):
        self.ops = [op._bart() for op in ops]
        self.bart_dim = bart_dim
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        super().__init__()

    def _create(self) -> Built:
        handles = _marshal.handles(op._h.ptr for op in self.ops)
        device = next((op.device for op in self.ops if op.device is not None), None)
        ptr = self._under_lock(
            library().bartorch_linop_stack_cod,
            len(self.ops),
            handles,
            self.bart_dim,
            device=device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=tuple(self.ops), device=device)

    def __repr__(self) -> str:
        return f"stack({', '.join(repr(op) for op in self.ops)})"


class _BlockDiagonal(LinearOperator):
    """``linop_stack``: separate inputs to separate outputs."""

    def __init__(
        self, a: LinearOperator, b: LinearOperator, cod_dim: int, dom_dim: int, oshape, ishape
    ):
        self.a, self.b = a._bart(), b._bart()
        self.cod_dim, self.dom_dim = cod_dim, dom_dim
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        super().__init__()

    def _create(self) -> Built:
        device = self.a.device or self.b.device
        ptr = self._under_lock(
            library().bartorch_linop_stack,
            self.cod_dim,
            self.dom_dim,
            self.a._h.ptr,
            self.b._h.ptr,
            device=device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.a, self.b), device=device)

    def __repr__(self) -> str:
        return f"block_diag({self.a!r}, {self.b!r})"


def concatenate(ops: Sequence[LinearOperator], axis: int = 0) -> LinearOperator:
    """Apply every operator to the same input and lay the results end to end.

    ``torch.cat`` of what they return, and what pylops calls ``VStack``: the
    operators share a domain, and their codomains agree except along ``axis``,
    where the sizes add up.

    Parameters
    ----------
    ops : sequence of LinearOperator
        At least one.  All must have the same domain.
    axis : int, default=0
        Which axis of the codomain grows, as an index into it.

    Examples
    --------
    >>> A = concatenate([FFT(shape, axes=-1), Identity(shape)])
    >>> A.oshape == (2 * shape[0],) + shape[1:]
    True
    """
    ops = _operators(ops, "concatenate")
    first = ops[0]
    ndim = len(first.oshape)
    axis %= ndim

    for op in ops[1:]:
        if op.ishape != first.ishape:
            raise ValueError(
                f"concatenate needs one domain for all of them: {op.ishape} against {first.ishape}"
            )
        if len(op.oshape) != ndim:
            raise ValueError(f"codomains differ in rank: {op.oshape} against {first.oshape}")
        if any(m != n for i, (m, n) in enumerate(zip(op.oshape, first.oshape)) if i != axis):
            raise ValueError(
                f"codomains must agree off axis {axis}: {op.oshape} against {first.oshape}"
            )

    if 1 == len(ops):
        return ops[0]

    oshape = list(first.oshape)
    oshape[axis] = sum(op.oshape[axis] for op in ops)
    return _StackCodomain(ops, _bart_dim(axis, ndim), tuple(oshape), first.ishape)


def stack(ops: Sequence[LinearOperator], axis: int = 0) -> LinearOperator:
    """Apply every operator to the same input and put the results on a new axis.

    ``torch.stack`` of what they return: unlike :func:`concatenate` the
    codomains must agree everywhere, and the result gains an axis of length
    ``len(ops)``.

    Parameters
    ----------
    ops : sequence of LinearOperator
        At least one.  All must have the same domain and the same codomain.
    axis : int, default=0
        Where the new axis goes in the result.

    Examples
    --------
    >>> A = stack([FFT(shape, axes=-1), Identity(shape)])
    >>> A.oshape == (2,) + shape
    True
    """
    from bartorch.linop.shape import Permute, Reshape

    ops = _operators(ops, "stack")
    first = ops[0]
    for op in ops[1:]:
        if op.oshape != first.oshape or op.ishape != first.ishape:
            raise ValueError(
                f"stack needs the same shapes throughout: {op.ishape}->{op.oshape} "
                f"against {first.ishape}->{first.oshape}"
            )

    # A new leading axis is a BART dimension the codomain does not use, which
    # every operator has: they are built at DIMS dimensions and the ones past
    # the shape are one.  Stacking there grows it to len(ops), which read back
    # in C order is a new leading axis.
    ndim = len(first.oshape)
    if ndim >= DIMS:
        raise ValueError(f"a new axis would take this past BART's {DIMS} dimensions")

    widened = [Reshape((1, *op.oshape), op.oshape) @ op for op in ops]
    out = concatenate(widened, axis=0)

    axis %= ndim + 1
    if 0 == axis:
        return out
    order = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim + 1))
    return Permute(out.oshape, order) @ out


def hstack(ops: Sequence[LinearOperator], axis: int = 0) -> LinearOperator:
    """Split the input between the operators and add up what they return.

    ``[A B] @ [x; y] = A x + B y``, which pylops calls ``HStack``: the
    operators share a codomain, and their domains agree except along ``axis``.

    BART has no constructor for this, and it needs none: the adjoint of
    stacking the codomains is stacking the domains, so this is
    ``concatenate([op.H for op in ops], axis).H`` and stays one BART operator.

    Parameters
    ----------
    ops : sequence of LinearOperator
        At least one.  All must have the same codomain.
    axis : int, default=0
        Which axis of the domain grows, as an index into it.
    """
    ops = _operators(ops, "hstack")
    return concatenate([op.H for op in ops], axis).H


def block_diag(
    ops: Sequence[LinearOperator], axis: int = 0, domain_axis: int | None = None
) -> LinearOperator:
    """Give each operator its own part of the input and its own part of the output.

    BART's ``linop_stack``, folded over the sequence.  The domains are laid end
    to end along ``domain_axis`` and the codomains along ``axis``; both default
    to the first, and each side must agree off the axis it grows along.

    Parameters
    ----------
    ops : sequence of LinearOperator
        At least one.
    axis : int, default=0
        Which axis of the codomain grows.
    domain_axis : int, default=None
        Which axis of the domain grows; ``axis`` when left out.
    """
    ops = _operators(ops, "block_diag")
    out = ops[0]
    for op in ops[1:]:
        cod_ndim, dom_ndim = len(out.oshape), len(out.ishape)
        cod_axis = axis % cod_ndim
        dom_axis = (axis if domain_axis is None else domain_axis) % dom_ndim

        oshape, ishape = list(out.oshape), list(out.ishape)
        oshape[cod_axis] += op.oshape[cod_axis]
        ishape[dom_axis] += op.ishape[dom_axis]

        out = _BlockDiagonal(
            out,
            op,
            _bart_dim(cod_axis, cod_ndim),
            _bart_dim(dom_axis, dom_ndim),
            tuple(oshape),
            tuple(ishape),
        )
    return out


def block(rows: Sequence[Sequence[LinearOperator]], axis: int = 0) -> LinearOperator:
    """A block matrix of operators.

    Every operator in a row shares a codomain and every operator in a column
    shares a domain; the row is :func:`hstack` and the rows together are
    :func:`concatenate`.

    Parameters
    ----------
    rows : sequence of sequence of LinearOperator
        The blocks, row by row.  Rows must be the same length.
    axis : int, default=0
        Which axis grows, in the domain within a row and in the codomain
        between rows.
    """
    rows = [list(row) for row in rows]
    if not rows or not rows[0]:
        raise ValueError("block needs at least one operator")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("every row of a block needs the same number of operators")
    return concatenate([hstack(row, axis) for row in rows], axis)

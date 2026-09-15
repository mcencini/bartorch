"""Matching a composition against the encoding form, and lowering it.

Composing builds a description: :class:`Chain` is one encoding with an
element-wise factor on either side of it, :class:`Sum` is an addition of such
chains, and :class:`Contract` is a sum whose terms share an encoding, their
factors stacked along a leading axis.  :func:`lower` walks the description
once and returns a single encoding with the factors folded into its form, so
that what a solver drives is one operator and each application is one call
into the library.  Where the description does not match, :func:`materialise`
builds the same thing as BART's plain chain of operators -- correct, and
slower -- and the plan the operator reports says which of the two happened.

:func:`describe` reads an already-composed operator back into a description,
so a composition written with ``@`` and ``+`` is matched by the same walk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from bartorch.linop.base import LinearOperator

__all__ = ["Chain", "Contract", "Sum", "build", "describe", "lower", "lowered", "materialise"]


@dataclass(frozen=True)
class Chain:
    """One encoding with an element-wise factor on either side of it.

    ``image`` multiplies the image before the encoding and ``kspace`` the
    samples after it, each broadcast over the encoding's own shape; either
    may be absent.
    """

    encoding: LinearOperator
    image: torch.Tensor | None = None
    kspace: torch.Tensor | None = None


@dataclass(frozen=True)
class Sum:
    """An addition of chains, which is a contraction when they share an encoding."""

    terms: tuple[Chain, ...]


@dataclass(frozen=True)
class Contract:
    """Terms sharing one encoding, with the factors stacked along a leading axis.

    ``image`` is ``(terms, *image factor)`` and ``kspace`` ``(terms, *k-space
    factor)``, each broadcast over the encoding's shape on the axes after the
    first.  This is the shape a fit hands its coefficients back in, so a
    caller that has them already does not take them apart to say so.
    """

    encoding: LinearOperator
    kspace: torch.Tensor
    image: torch.Tensor

    @property
    def terms(self) -> tuple[Chain, ...]:
        return tuple(
            Chain(self.encoding, self.image[term], self.kspace[term])
            for term in range(int(self.kspace.shape[0]))
        )


def describe(op: LinearOperator) -> Sum | None:
    """``op`` as a description, or ``None`` where it is not one.

    A product of diagonals around an encoding is a :class:`Chain`; a sum of
    those is a :class:`Sum`.  Anything else -- a transform that is not one of
    the three, an operation that is not element-wise on either side -- has no
    description, and is left as it stands.
    """
    from bartorch.linop.base import _Add, _Compose
    from bartorch.linop.basic import Diagonal
    from bartorch.linop.sense import NoncartesianSense

    if isinstance(op, _Add):
        left, right = describe(op.a), describe(op.b)
        if left is None or right is None:
            return None
        return Sum((*left.terms, *right.terms))

    image = kspace = None
    while isinstance(op, _Compose):
        if isinstance(op.a, Diagonal) and kspace is None:
            kspace, op = op.a.diag, op.b
        elif isinstance(op.b, Diagonal) and image is None:
            image, op = op.b.diag, op.a
        else:
            return None

    if not isinstance(op, NoncartesianSense):
        return None
    return Sum((Chain(op, image, kspace),))


def stacked(description: Sum) -> Contract | None:
    """The sum as a contraction over its terms, or ``None`` where it is not one.

    The terms have to share an encoding, and each factor has to have the same
    shape in every term: two terms whose weights lie on different axes are
    not one contraction, whatever they add up to.
    """
    terms = description.terms
    encoding = terms[0].encoding
    if any(term.encoding is not encoding for term in terms):
        return None

    def stack(factors, shape):
        widened = [
            torch.ones((1,) * len(shape), dtype=torch.complex64) if f is None else f
            for f in factors
        ]
        if len({tuple(f.shape) for f in widened}) != 1:
            return None
        return torch.stack(widened)

    kspace = stack([term.kspace for term in terms], encoding.oshape)
    image = stack([term.image for term in terms], encoding.ishape)
    if kspace is None or image is None:
        return None
    return Contract(encoding, kspace, image)


def lower(description: Sum | Contract) -> LinearOperator | None:
    """The description as one encoding with its factors folded in, or ``None``.

    ``None`` says the description is outside the form: the terms do not share
    an encoding, a factor varies along an axis the form cannot hold it on, or
    the encoding already carries a contraction.
    """
    from bartorch.linop.mri import contracted

    if isinstance(description, Sum):
        if len(description.terms) == 1:
            only = description.terms[0]
            if only.image is None and only.kspace is None:
                return only.encoding
        contraction = stacked(description)
        if contraction is None:
            return None
    else:
        contraction = description

    return contracted(contraction.encoding, contraction.kspace, contraction.image)


def materialise(description: Sum | Contract) -> LinearOperator:
    """The description as BART's plain chain of operators.

    One term is the two diagonals chained onto the encoding; several are
    ``linop_plus`` over those, which is still a single BART operator and so
    still one call per application.  What it is not is one transform: each
    term applies the encoding itself.
    """
    from bartorch.linop.base import _Add, _Compose
    from bartorch.linop.basic import Diagonal

    # Built without matching: this is the description's fallback, so a node
    # that lowered itself back into the contraction would defeat the point.
    terms = description.terms
    out: LinearOperator | None = None
    for term in terms:
        built = term.encoding
        if term.image is not None:
            diagonal = Diagonal(_widened(term.image, term.encoding.ishape), built.ishape)
            built = _Compose(built, diagonal, match=False)
        if term.kspace is not None:
            diagonal = Diagonal(_widened(term.kspace, built.oshape), built.oshape)
            built = _Compose(diagonal, built, match=False)
        out = built if out is None else _Add(out, built, match=False)

    # The encoding inside still runs where its own plan says; what is not
    # fused is the sum, and the plan says so rather than reporting the
    # encoding's own and leaving the sum invisible.
    inner = terms[0].encoding.plan
    if inner is not None:
        out._plan = replace(inner, contraction="chained", terms=len(terms))
    return out


def _widened(factor: torch.Tensor, shape) -> torch.Tensor:
    """``factor`` given the rank of ``shape``, with ones where it is to broadcast."""
    got = tuple(factor.shape)
    return factor.reshape((1,) * (len(shape) - len(got)) + got)


def build(description: Sum | Contract) -> LinearOperator:
    """The description lowered where it matches the form, and chained where it does not."""
    matched = lower(description)
    return materialise(description) if matched is None else matched


def lowered(op: LinearOperator) -> LinearOperator | None:
    """``op`` as one encoding, or ``None`` where it is not one.

    What a composition asks before it builds itself: a description the form
    holds becomes a single encoding with the factors and the terms folded in,
    and anything else is left to the chain the composition stands for.
    """
    description = describe(op)
    if description is None:
        return None
    return lower(description)

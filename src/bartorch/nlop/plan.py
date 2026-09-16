"""Matching a nonlinear composition against the coil model, and lowering it.

A product of two unknowns behind a linear encoding is the model ``noir`` fits, and
the rewriting that pays is to apply the encoding once as its normal rather than
twice as a pair, as ``noir2_join`` already does off the grid.  See
``docs/design/nonlinear-fusion.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from bartorch.linop.base import LinearOperator
from bartorch.nlop.base import NonlinearOperator

__all__ = ["Plan"]


@dataclass(frozen=True)
class Plan:
    """Lowered form of a nonlinear model, and how a Gauss-Newton step applies its encoding.

    Attributes
    ----------
    bundle : str or None
        Where the derivative came from: ``"declared"``, ``"chain rule"``,
        ``"linear"``, ``"torch"``, or ``None`` for a model that has none.
    domain : str
        ``"normal"`` where the step works against ``E^H E`` and the data has
        been through ``E^H``, ``"paired"`` where the encoding is applied
        forward and adjoint.
    encoding : bartorch.linop.form.Plan or None
        The linear part's own plan, where it has one.
    fused : bool
        False where the model is not a coil composition, or where the rewrite
        was declined because the encoding's normal is no cheaper than the pair.
    """

    bundle: str | None = None
    domain: str = "paired"
    encoding: object | None = None
    fused: bool = False

    def __repr__(self) -> str:
        parts = [f"bundle={self.bundle}", f"domain={self.domain}", f"fused={self.fused}"]
        if self.encoding is not None:
            parts.append(f"encoding={self.encoding!r}")
        return f"Plan({', '.join(parts)})"


@dataclass(frozen=True)
class Coils:
    """A product of two unknowns with one linear encoding after it.

    ``lowered`` is set where the encoding is already applied as its normal, as
    ``noir2``'s non-Cartesian model is.
    """

    product: NonlinearOperator
    encoding: LinearOperator
    lowered: bool


def describe(F) -> Coils | None:
    """``F`` as a coil composition, or ``None`` where it is not one.

    A model that writes itself out -- :class:`~bartorch.nlop.NonlinearSense`
    does -- is read through that, so BART's own model and one built here from
    :func:`~bartorch.nlop.CoilSense` are the same description.
    """
    from bartorch.nlop.base import FromLinear, _Chain2
    from bartorch.nlop.bundle import Asymmetric

    written = getattr(F, "_composition", None)
    if written is not None:
        F = written()

    if not isinstance(F, _Chain2) or 0 != F.output or 0 != F.input:
        return None
    product, last = F.a, F.b
    if 2 != len(product.ishapes) or 1 != len(product.oshapes):
        return None
    if not isinstance(last, FromLinear):
        return None
    if isinstance(last, Asymmetric):
        return None if last.source is None else Coils(product, last.source, True)
    return Coils(product, last.op, False)


def lower(description: Coils) -> NonlinearOperator | None:
    """The composition with its encoding applied once as its normal, or ``None``.

    Taken wherever the composition matches: ``linop_get_normal`` is the
    encoding's own normal where it has one -- a point spread function for a
    NUFFT, the transform's own where no k-space factor survives -- and the two
    applications where it has not, which is the pair's cost anyway.  The
    trade-off is where the data lives; :meth:`IRGNMBlock.start` prepares it and
    :attr:`Plan.domain` reports it.

    ``None`` says it has already been taken, as it has in BART's own
    non-Cartesian model.
    """
    from bartorch.linop.basic import Identity
    from bartorch.nlop.base import chain
    from bartorch.nlop.bundle import Asymmetric

    if description.lowered:
        return None
    encoding = description.encoding
    stage = Asymmetric(encoding.gram(), Identity(encoding.ishape), source=encoding)
    return chain(description.product, stage, output=0, input=0)


def build(F, *, fuse: bool = True) -> tuple[NonlinearOperator, LinearOperator | None, Plan]:
    """``F`` as a step should assemble over it, what prepares its data, and the plan.

    The operator comes back unchanged where there is nothing to rewrite; the
    preparation is ``E^H`` where the step works in the normal-equation domain
    and ``None`` where it does not.  ``fuse=False`` declines the rewrite, which
    is the reference the fused answer is held against, and a model BART already built in
    that domain is past declining.
    """
    description = describe(F)
    bundle = F.bundle
    source = None if bundle is None else bundle.source
    if description is None:
        return F, None, Plan(bundle=source)

    encoding = description.encoding.plan
    if description.lowered:
        return F, description.encoding.H, Plan(source, "normal", encoding, True)

    made = None if not fuse else lower(description)
    if made is None:
        return F, None, Plan(source, "paired", encoding, False)
    return made, description.encoding.H, Plan(made.bundle.source, "normal", encoding, True)

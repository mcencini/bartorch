"""A regularization term, as the description BART's own configure step takes.

Nothing here computes a proximal step.  BART builds the proximal operator and
the transform that goes with it -- the wavelet transform under a wavelet
threshold, the gradient under total variation -- in ``opt_reg_configure``, and
a term is what that function reads: which kind, over which axes, joined along
which, with what weight.  Filling that table from an object rather than from a
string is the only difference between this and ``pics -R``.
"""

from __future__ import annotations

import abc

from bartorch._operator import axes_flags

__all__ = ["Regularizer"]


class Regularizer(abc.ABC):
    """One of BART's regularization terms.

    Attributes
    ----------
    kind : str
        BART's own letter for the term, which is what ``pics -R`` writes.
    weight : float
        The term's weight.
    axes : tuple of int
        The axes it works over, as indices into the image's shape.  Empty
        where the term has none of its own.
    joint_axes : tuple of int
        The axes it is joined along, which is BART's second flag.
    count : int
        The count an NIHT term takes, and zero for every other.
    """

    kind: str = ""
    weight: float = 0.0
    axes: tuple[int, ...] = ()
    joint_axes: tuple[int, ...] = ()
    count: int = 0

    def flags(self, ndim: int) -> tuple[int, int]:
        """The two bitmasks BART reads, for an image of *ndim* axes.

        An axis is an index into the image's shape here and a bit position in
        BART's own order there, which is the conversion the rest of the
        package makes at every boundary.
        """
        return (
            axes_flags(self.axes, ndim) if self.axes else 0,
            axes_flags(self.joint_axes, ndim) if self.joint_axes else 0,
        )

    def __repr__(self) -> str:
        parts = [f"weight={self.weight}"] if self.weight else []
        if self.axes:
            parts.insert(0, f"axes={self.axes}")
        if self.joint_axes:
            parts.append(f"joint_axes={self.joint_axes}")
        if self.count:
            parts.append(f"count={self.count}")
        return f"{type(self).__name__}({', '.join(parts)})"

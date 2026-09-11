"""A regularization term, holding the operator BART builds for it.

Nothing here computes a proximal step.  ``opt_reg_configure`` makes the
proximal operator and the transform that goes with it -- the wavelet transform
under a wavelet threshold, the gradient under total variation -- and a term
here is the pair it made, built once against the shape it works on and handed
to the solver as it stands.  Solving twice with the same term builds nothing
the second time.
"""

from __future__ import annotations

import abc
import weakref

from bartorch import _marshal
from bartorch._lib import library
from bartorch._operator import axes_flags
from bartorch.core.graph import BartError, _ensure_ready, _lock

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

    def build(
        self,
        shape: tuple[int, ...],
        *,
        block: int = 8,
        wavelet: str = "dau2",
        randshift: bool = True,
        overlapping_blocks: bool = False,
    ) -> int:
        """The BART operator for this term over an image of *shape*, built once.

        Parameters
        ----------
        shape : tuple of int
            The image's shape, C order.
        block : int
            Block size for a locally low-rank term, which is ``pics -b``.
        wavelet : str
            Wavelet family for a wavelet term, which is ``pics --wavelet``.
        randshift : bool
            Cycle-spin the transform by a random shift, which is what the tool
            does unless it is given ``pics -n``.
        overlapping_blocks : bool
            Fully overlapping blocks for a locally low-rank term, which is
            ``pics -N``.  It replaces the random shift rather than joining it,
            as it does for the tool.

        Returns
        -------
        int
            The handle the solver is given.  It belongs to this object and
            lives as long as it does.
        """
        shift_mode = 2 if overlapping_blocks else (1 if randshift else 0)
        key = (tuple(shape), block, wavelet, shift_mode)
        if not hasattr(self, "_handles"):
            self._handles = {}
        if key in self._handles:
            return self._handles[key]

        _ensure_ready()
        xflags, jflags = self.flags(len(shape))
        out = _marshal.out_pointer()
        with _lock:
            code = library().bartorch_prox_create(
                self.kind.encode(),
                xflags,
                jflags,
                float(self.weight),
                int(self.count),
                int(block),
                wavelet.encode(),
                shift_mode,
                _marshal.padded_dims(tuple(shape)),
                _marshal.by_reference(out),
            )
        if code != 0:
            said = library().bartorch_solve_error(code).decode(errors="replace")
            raise BartError(f"{self!r} could not be built: {said}")

        handle = out.value
        self._handles[key] = handle
        weakref.finalize(self, _release, handle)
        return handle

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


def _release(handle: int) -> None:
    """Give one built term back to BART."""
    with _lock:
        library().bartorch_prox_free(handle)

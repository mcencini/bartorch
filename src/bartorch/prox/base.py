"""The regularization term base class."""

from __future__ import annotations

import abc
import weakref

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock
from bartorch._lib import library
from bartorch._operator import axes_flags

__all__ = ["Regularizer"]


class Regularizer(abc.ABC):
    """One of BART's regularization terms.

    A term holds the proximal operator and transform BART's
    ``opt_reg_configure`` builds for it, per image shape, and hands them to a
    solver as they are: solving twice with a term builds nothing the second
    time.

    Attributes
    ----------
    kind : str
        BART's letter for the term, as ``pics -R`` writes it.
    axes : tuple of int
        Axes the term works over, as indices into the image's shape.
    joint_axes : tuple of int
        Axes along which the term acts jointly; BART's second bitmask.
    count : int
        Entries an NIHT term keeps; zero for every other term.
    """

    kind: str = ""
    weight: float = 0.0
    axes: tuple[int, ...] = ()
    joint_axes: tuple[int, ...] = ()
    count: int = 0

    def build(self, shape: tuple[int, ...]) -> int:
        """The BART operator for this term over an image of C-order ``shape``.

        Built on first use for each shape.  The returned handle is owned by
        this term and freed with it.
        """
        shape = tuple(shape)
        if not hasattr(self, "_handles"):
            self._handles = {}
        if shape in self._handles:
            return self._handles[shape]

        _ensure_ready()
        xflags, jflags = self.flags(len(shape))
        block, family, shift_mode = self._options()
        out = _marshal.out_pointer()
        with _lock:
            code = library().bartorch_prox_create(
                self.kind.encode(),
                xflags,
                jflags,
                float(self.weight),
                int(self.count),
                int(block),
                family.encode(),
                shift_mode,
                _marshal.padded_dims(shape),
                _marshal.by_reference(out),
            )
        if code != 0:
            said = library().bartorch_solve_error(code).decode(errors="replace")
            raise BartError(f"{self!r} could not be built: {said}")

        handle = out.value
        self._handles[shape] = handle
        weakref.finalize(self, _release, handle)
        return handle

    def _options(self) -> tuple[int, str, int]:
        """Block size, wavelet family and shift mode for ``opt_reg_configure``.

        Only the wavelet and locally low-rank terms read them.  Shift mode 0
        is no shift, 1 the random cycle spinning ``pics`` does unless ``-n``,
        2 fully overlapping blocks (``pics -N``).
        """
        return 8, "dau2", 1

    def flags(self, ndim: int) -> tuple[int, int]:
        """BART's bitmasks for :attr:`axes` and :attr:`joint_axes`, for an ``ndim``-axis image."""
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
    with _lock:
        library().bartorch_prox_free(handle)

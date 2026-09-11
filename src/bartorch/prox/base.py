"""The regularization term base class."""

from __future__ import annotations

import abc
import weakref
from collections.abc import Iterable

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
        Axes along which the term acts jointly.
    count : int
        Entries an NIHT term keeps; zero for every other term.
    """

    kind: str = ""
    weight: float = 0.0
    axes: tuple[int, ...] = ()
    joint_axes: tuple[int, ...] = ()
    count: int = 0
    #: Whether the term adds variables to the optimization, which BART counts
    #: across the whole set of terms, so that it cannot be built alone.
    _extends: bool = False

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
        xflags, jflags = self._flags(len(shape))
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

    def _settings(self) -> dict[str, object]:
        """The settings a command line gives once for all its terms, as this term needs them.

        Keys are those of :data:`_SHARED_DEFAULTS`.
        """
        return {}

    def _flags(self, ndim: int) -> tuple[int, int]:
        """BART's bitmasks for :attr:`axes` and :attr:`joint_axes`, for an ``ndim``-axis image."""
        return (
            axes_flags(self.axes, ndim) if self.axes else 0,
            axes_flags(self.joint_axes, ndim) if self.joint_axes else 0,
        )

    def _argument(self, ndim: int) -> str:
        """This term as a ``-R`` argument, for an ``ndim``-axis image."""
        x, j = self._flags(ndim)
        if self.kind == "Q":
            return f"Q:{self.weight!r}"
        if self.kind == "S":
            return "S"
        if self.kind in ("I", "R1", "R2"):
            return f"{self.kind}:{j}:{self.weight!r}"
        if self.kind in ("H", "N"):
            return f"{self.kind}:{x}:{j}:{self.count}"
        return f"{self.kind}:{x}:{j}:{self.weight!r}"

    def __repr__(self) -> str:
        parts = [f"weight={self.weight}"] if self.weight else []
        if self.axes:
            parts.insert(0, f"axes={self.axes}")
        if self.joint_axes:
            parts.append(f"joint_axes={self.joint_axes}")
        if self.count:
            parts.append(f"count={self.count}")
        return f"{type(self).__name__}({', '.join(parts)})"


#: BART's value for each setting a command line gives once for all its terms.
_SHARED_DEFAULTS: dict[str, object] = {
    "randshift": True,
    "overlapping": False,
    "block": 8,
    "family": "dau2",
    "alpha": (1.0, 3.0**0.5),
    "gamma": (1.0, 1.0),
}


def _as_terms(regularizers) -> list[Regularizer]:
    """``regularizers`` -- None, one term or an iterable of them -- as a list of terms."""
    if isinstance(regularizers, str):
        raise TypeError(
            f"a regularizer is a term from bartorch.prox, not the string {regularizers!r}; "
            "`prox.Wavelet(axes=(-1, -2), weight=...)` is what `-R W:3:0:...` says"
        )
    if regularizers is None:
        return []
    if isinstance(regularizers, Regularizer):
        return [regularizers]
    terms = list(regularizers) if isinstance(regularizers, Iterable) else [regularizers]
    for term in terms:
        if not isinstance(term, Regularizer):
            raise TypeError(f"a regularizer is a term from bartorch.prox, not {term!r}")
    return terms


def _command_line(
    terms: list[Regularizer], ndim: int | None, command: str, kinds: Iterable[str] | None = None
) -> tuple[list[str], dict[str, object]]:
    """``-R`` arguments for ``terms`` over an ``ndim``-axis image, and their shared settings.

    With ``ndim`` None only negative axes are accepted.  The settings returned
    are those that differ from BART's defaults.

    Raises
    ------
    TypeError
        ``command`` does not take one of the terms (``kinds`` lists those it takes).
    ValueError
        Two terms need different values of one shared setting.
    """
    arguments: list[str] = []
    shared: dict[str, object] = {}
    for term in terms:
        if kinds is not None and term.kind not in kinds:
            raise TypeError(f"{command} does not take {type(term).__name__} terms")
        arguments.append(term._argument(ndim))
        for name, value in term._settings().items():
            if shared.setdefault(name, value) != value:
                raise ValueError(
                    f"{command} sets {name} once for every term, "
                    f"and these terms ask for {shared[name]!r} and {value!r}"
                )
    return arguments, {k: v for k, v in shared.items() if v != _SHARED_DEFAULTS[k]}


def _release(handle: int) -> None:
    with _lock:
        library().bartorch_prox_free(handle)

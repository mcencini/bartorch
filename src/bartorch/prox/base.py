"""The regularization term base class."""

from __future__ import annotations

import abc
import weakref
from collections.abc import Iterable

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import as_operand, axes_flags

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

    def prox_shape(self, image_shape: tuple[int, ...]) -> tuple[int, ...]:
        """The C-order shape this term's proximal operator works on.

        The image's, for a term that carries its own transform inside the
        proximal operator -- which is most of them, wavelets included.  A term
        whose transform is in front instead answers with the transform's
        codomain: total variation thresholds the components of a gradient, and
        those are an axis of their own.
        """
        image_shape = tuple(image_shape)
        handle = self.build(image_shape)
        out = _marshal.wide_dim_vector()

        with _lock:
            rank = library().bartorch_prox_domain(handle, DIMS + 1, out)
        if rank < 0:
            raise BartError(f"{self!r} would not say what it works on (code {rank})")

        shape = [int(out[i]) for i in range(rank)][::-1]
        while len(shape) > len(image_shape) and 1 == shape[0]:
            shape.pop(0)
        return tuple(shape)

    def prox(
        self, x: torch.Tensor, gamma: float = 1.0, *, image_shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """``prox_{gamma f}(x)``, the operator BART's solvers call.

        What a solver does between its gradient steps, reached on its own so
        that an iteration written outside the library calls the same operator
        the library would have.

        Parameters
        ----------
        x : tensor
            Of :meth:`prox_shape`, which for most terms is the image's.
        gamma : float
            The step the proximal operator is taken at.  The term's own weight
            is already in the operator, so this is only the step.
        image_shape : tuple of int, optional
            The image the term was configured for, when that is not what ``x``
            is shaped like -- which is the case for total variation, whose
            proximal operator works on the components of a gradient.  By
            default ``x``'s own shape, which is right for every other term.

        Returns
        -------
        torch.Tensor

        Notes
        -----
        This is the proximal operator alone.  A term may also carry a
        transform in front of it -- see :meth:`transform` -- and then what a
        solver computes is ``prox(transform(x))``.  BART's own ``iter2_ist``
        applies the proximal operator to the image and ignores the transform,
        which is why BART's IST and FISTA cannot take a total-variation term:
        its proximal operator is not shaped like an image.

        Examples
        --------
        >>> prox.Wavelet((-1, -2), 0.01).prox(image, gamma=0.95)
        """
        image_shape = tuple(x.shape) if image_shape is None else tuple(image_shape)
        shape = self.prox_shape(image_shape)
        if tuple(x.shape) != shape:
            raise ValueError(
                f"over an image of {image_shape}, {self!r} works on {shape}, not {tuple(x.shape)}"
            )

        handle = self.build(image_shape)
        src = as_operand(x, shape, "x")
        out = torch.empty_like(src)

        with _lock, _on_device(src.device):
            code = library().bartorch_prox_apply(
                handle, float(gamma), out.data_ptr(), src.data_ptr()
            )
        if code != 0:
            said = library().bartorch_solve_error(code).decode(errors="replace")
            raise BartError(f"{self!r} could not be applied: {said}")
        return out

    def transform(self, image_shape: tuple[int, ...]):
        """The operator BART puts in front of this term's proximal operator.

        The identity for a term that carries its transform inside the proximal
        operator instead.  The shapes do not say which arrangement a term is:
        the Laplace term's transform is a real convolution whose codomain is
        shaped like the image, so a caller that guessed from the shape would
        quietly leave it out.

        Returns
        -------
        LinearOperator
        """
        handle = self.build(tuple(image_shape))
        lib = library()
        with _lock:
            ptr = lib.bartorch_prox_transform(handle)
        if not ptr:
            raise BartError(f"{self!r} has no transform to give")

        try:
            ishape = _handle_shape(lib.bartorch_linop_domain, ptr, len(image_shape))
            oshape = _handle_shape(lib.bartorch_linop_codomain, ptr, len(image_shape))
        except BartError:
            lib.bartorch_linop_free(ptr)
            raise
        return _Transform(ptr, ishape, oshape)

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


def _handle_shape(query, ptr: int, min_ndim: int) -> tuple[int, ...]:
    """The C-order shape BART records for an operator handle.

    The query fills as many entries as it is given and returns the rank it
    has, which is how a rank past BART's sixteen is noticed rather than
    quietly truncated.
    """
    vector = _marshal.dim_vector()
    rank = query(ptr, DIMS, vector)
    if rank > DIMS:
        raise BartError(
            f"this term's transform works at rank {rank}, past BART's {DIMS}, which an "
            "operator here cannot hold; total variation is the one that does -- its "
            "gradient puts the components on an axis of their own beyond the image's -- "
            "and its proximal operator is still reachable through prox()"
        )
    return tuple(_marshal.shape_from_dims(vector, min_ndim))


class _Transform:
    """The operator a term carries, from a handle BART has already built."""

    def __new__(cls, ptr: int, ishape, oshape):
        from bartorch._operator import Built
        from bartorch.linop.base import LinearOperator

        class Held(LinearOperator):
            def __init__(self):
                self._held = (ptr, tuple(ishape), tuple(oshape))
                super().__init__()

            def _create(self):
                return Built(self._held[0], self._held[1], self._held[2])

            def __repr__(self) -> str:
                return f"<term transform {self._held[1]} -> {self._held[2]}>"

        return Held()


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

"""The regularizer base class."""

from __future__ import annotations

import abc
import weakref
from collections.abc import Iterable

import torch

from bartorch import _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import as_operand, axes_flags

__all__ = ["Regularizer", "frozen"]


class Regularizer(abc.ABC):
    """Regularization functional with a proximal operator built by BART.

    A regularizer represents :math:`g(G x)`: a linear transform :math:`G` and
    the proximal operator of a functional :math:`g` acting on its codomain.
    :math:`G` is the identity for terms that penalize the image directly, and
    for terms carrying their own transform inside the proximal operator, such
    as :class:`~bartorch.priors.Wavelet`; it is a genuine operator for terms
    such as :class:`~bartorch.priors.TotalVariation`, whose functional acts on
    finite differences.  Both come from BART's ``opt_reg_configure``.

    Only the alternating-direction and primal-dual iterations are given
    :math:`G`; see :meth:`transform_is_identity`.

    Both are built per image shape and cached, and handed to a solver as they
    are, so solving twice with the same term builds nothing the second time.

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
    #: Whether the term extends the optimization variable with auxiliary ones,
    #: which BART counts
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
        proximal operator, a wavelet term included.  A term
        whose transform is in front instead answers with the transform's
        codomain: total variation thresholds the components of a gradient, and
        those are an axis of their own.
        """
        image_shape = tuple(image_shape)
        handle = self.build(image_shape)
        out = _marshal.wide_dim_vector()

        with _lock:
            rank = library().bartorch_prox_domain(handle, len(out), out)
        if rank < 0:
            raise BartError(f"{self!r} would not say what it works on (code {rank})")

        shape = [int(out[i]) for i in range(rank)][::-1]
        while len(shape) > len(image_shape) and 1 == shape[0]:
            shape.pop(0)
        return tuple(shape)

    def prox(
        self, x: torch.Tensor, gamma: float = 1.0, *, image_shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """``prox_{gamma g}(x)``, the operator BART's solvers apply.

        Exposed directly so that an iteration written outside the library
        applies the same proximal operator the library would have.

        Raises a :class:`ValueError` for a tensor that requires a gradient: no
        backward pass is implemented for BART's proximal operators, and
        treating one as the identity would zero the gradient path through the
        regularizer.  Use :class:`~bartorch.priors.ImplicitPrior` for a
        differentiable proximal step, or :func:`frozen` to hold this one fixed.

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
        This is the proximal operator alone.  A term may also carry a linear
        transform in front of it -- see :meth:`transform` -- in which case a
        solver computes ``prox(transform(x))``.  BART's ``iter2_ist`` applies
        the proximal operator to the image and ignores the transform entirely,
        so IST and FISTA admit only terms whose transform is the identity; a
        total-variation term is excluded, its proximal operator not being
        shaped like an image.

        Examples
        --------
        >>> priors.Wavelet((-1, -2), 0.01).prox(image, gamma=0.95)
        """
        from bartorch.linop.base import _tracking

        if _tracking(x):
            raise ValueError(
                f"{self!r} is one of BART's proximal operators and has no backward pass: "
                "`operator_p_fun_t` is (data, mu, dst, src), with nowhere to carry one, "
                "so an iteration containing this term cannot be differentiated.  Use a "
                "denoiser in place of the term -- `optim.admm(y, A, denoiser)` differentiates "
                "end to end -- or `priors.frozen(term)` to hold this one fixed in the graph"
            )

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

    def apply_transform(
        self, x: torch.Tensor, image_shape: tuple[int, ...] | None = None, mode: str = "forward"
    ) -> torch.Tensor:
        """This term's transform applied to ``x``, without making an operator of it.

        :meth:`transform` cannot answer for the gradient family -- their
        components live on an axis past BART's sixteen -- and those are exactly
        the terms an alternating-direction solver is for.  This applies the
        transform over the shapes :meth:`prox_shape` reports instead, which a
        tensor can hold at any rank.

        Parameters
        ----------
        x : tensor
            The image for ``"forward"``, the proximal operator's domain for
            ``"adjoint"``, the image for ``"normal"``.
        image_shape : tuple of int, optional
            What the term was configured for; by default ``x``'s own shape,
            which is right whenever the transform starts from the image.
        mode : {"forward", "adjoint", "normal"}

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Recorded for autograd when ``x`` carries a gradient: the backward pass
        is the transpose, which is the other of ``forward`` and ``adjoint``,
        and ``normal`` itself.  An unrolled network differentiates through the
        transform this way; the proximal operator behind it is BART's and has
        no implemented backward pass, which is why
        :class:`~bartorch.priors.ImplicitPrior` exists.
        """
        from bartorch.linop.base import _tracking

        if _tracking(x):
            from bartorch.priors.autograd import apply_transform

            return apply_transform(
                self, x, tuple(image_shape if image_shape is not None else x.shape), mode
            )
        return self._transform_apply(x, image_shape, mode)

    def _transform_apply(
        self, x: torch.Tensor, image_shape: tuple[int, ...] | None = None, mode: str = "forward"
    ) -> torch.Tensor:
        """:meth:`apply_transform`, without recording for autograd."""
        modes = {"forward": 0, "adjoint": 1, "normal": 2}
        if mode not in modes:
            raise ValueError(f"mode is forward, adjoint or normal, not {mode!r}")

        if image_shape is None:
            image_shape = tuple(x.shape)
        image_shape = tuple(image_shape)

        # forward: image -> the proximal operator's domain.  adjoint: back.
        # normal: image to image, which is the pair of them.
        transformed = self.prox_shape(image_shape)
        want, oshape = {
            "forward": (image_shape, transformed),
            "adjoint": (transformed, image_shape),
            "normal": (image_shape, image_shape),
        }[mode]
        if tuple(x.shape) != want:
            raise ValueError(f"the {mode} takes {want}, not {tuple(x.shape)}")

        handle = self.build(image_shape)
        src = as_operand(x, want, "x")
        out = torch.empty(oshape, dtype=torch.complex64, device=src.device)

        with _lock, _on_device(src.device):
            code = library().bartorch_prox_transform_apply(
                handle, modes[mode], out.data_ptr(), src.data_ptr()
            )
        if code != 0:
            raise BartError(f"{self!r} could not apply its transform ({mode})")
        return out

    def transform(self, image_shape: tuple[int, ...]):
        """The operator BART puts in front of this term's proximal operator.

        The identity for a term that carries its transform inside the proximal
        operator instead.  The shapes do not say which arrangement a term is:
        the Laplace term's transform is a real convolution whose codomain is
        shaped like the image, so a caller inferring the arrangement from the
        shape would omit it without noticing.

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

    def rewind(self, image_shape: tuple[int, ...]) -> None:
        """Put this term's own random generator back to where it started.

        A wavelet threshold spins its transform by a shift drawn from a
        generator of its own, seeded at one when BART makes the operator.  The
        tool builds a fresh operator per run; a term here is kept, so a solve
        rewinds it and a reused term answers as the tool does.  Every solver
        does this before it iterates, whether the loop is BART's or written
        out here; a term with no such generator is left alone.
        """
        handle = self.build(tuple(image_shape))
        with _lock:
            code = library().bartorch_prox_rewind(handle)
        if code != 0:
            raise BartError(f"{self!r} could not be rewound")

    def transform_is_identity(self, image_shape: tuple[int, ...]) -> bool:
        """Whether the term's linear transform is the identity, as BART decides it.

        Decided by ``linop_is_identity`` on the operator, not by comparing
        shapes: the Laplace term's transform is a convolution on the image's
        own shape, and total variation's has a rank BART cannot hand over at
        all.  ``iter2_chambolle_pock`` asks this of the first term before
        treating it as the primal proximal step rather than a dual one.

        Returns
        -------
        bool
        """
        handle = self.build(tuple(image_shape))
        with _lock:
            answer = library().bartorch_prox_transform_is_identity(handle)
        if answer < 0:
            raise BartError(f"{self!r} could not be asked about its transform")
        return 1 == answer

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

    def _check(self, ndim: int | None) -> None:
        """Raise if this term cannot be built over an ``ndim``-axis image.

        BART states some of a term's preconditions as assertions, which end
        the process rather than return; and a term with auxiliary variables is
        built by `opt_reg_configure`, deep inside the solve, where there is
        nothing left to catch.  Whatever can be decided from the rank is
        decided here instead, while an exception still reaches the caller.
        Most terms have nothing to say and this does nothing.
        """

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
    has, so a rank past BART's sixteen is detected rather than truncated
    without notice.
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
            f"a regularizer is a term from bartorch.priors, not the string {regularizers!r}; "
            "`priors.Wavelet(axes=(-1, -2), weight=...)` states what `-R W:3:0:...` does"
        )
    if regularizers is None:
        return []
    if isinstance(regularizers, Regularizer):
        return [regularizers]
    terms = list(regularizers) if isinstance(regularizers, Iterable) else [regularizers]
    for term in terms:
        if not isinstance(term, Regularizer) and not _term_shaped(term):
            hint = "; a denoiser goes in priors.ImplicitPrior" if callable(term) else ""
            raise TypeError(f"a regularizer is a term from bartorch.priors, not {term!r}{hint}")
    return terms


def _term_shaped(thing) -> bool:
    """Whether something answers the four questions an iteration asks a term.

    :class:`~bartorch.priors.ImplicitPrior` does.  BART cannot take one, so a
    solver holding one has no library route.
    """
    return all(
        callable(getattr(thing, name, None))
        for name in ("prox", "prox_shape", "apply_transform", "rewind")
    )


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
        term._check(ndim)
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


class _Frozen:
    """A term whose proximal step is detached, making it a constant in the graph.

    Everything but :meth:`prox` is the wrapped term's own, and :meth:`prox`
    detaches its input first.  Registered as a :class:`Regularizer` because as
    far as BART is concerned it is one -- detaching changes no numbers -- so a
    solve containing it still has a library route and still reproduces the
    library's output bit for bit.
    """

    def __init__(self, term):
        self.term = term

    def __getattr__(self, name):
        # `term` itself is never delegated: asking for it before __init__ has
        # set it -- which is what unpickling does -- would recur forever.
        if "term" == name:
            raise AttributeError(name)
        return getattr(self.term, name)

    def prox(self, x, gamma: float = 1.0, *, image_shape=None):
        return self.term.prox(x.detach(), gamma, image_shape=image_shape)

    def __repr__(self) -> str:
        return f"frozen({self.term!r})"


Regularizer.register(_Frozen)


class _Penalty(Regularizer):
    """One penalty of a set BART configured together, over the image and the unknowns behind it.

    Its proximal operator detaches first when every term the set came from was
    :func:`frozen`.
    """

    kind = "penalty"

    def __init__(self, handle: int, shape: tuple[int, ...], frozen: bool):
        self._handles = {tuple(shape): handle}
        self._frozen = bool(frozen)
        weakref.finalize(self, _release, handle)

    def build(self, shape: tuple[int, ...]) -> int:
        shape = tuple(shape)
        if shape not in self._handles:
            raise ValueError(f"this penalty walks {next(iter(self._handles))}, not {shape}")
        return self._handles[shape]

    def prox(self, x, gamma: float = 1.0, *, image_shape=None):
        return super().prox(x.detach() if self._frozen else x, gamma, image_shape=image_shape)

    def __repr__(self) -> str:
        return f"penalty over {next(iter(self._handles))}"


def frozen(term: Regularizer) -> Regularizer:
    """``term`` with its proximal step detached, for a differentiated solve.

    No backward pass is implemented for BART's proximal operators, so
    :meth:`Regularizer.prox` raises on a tensor that requires a gradient rather
    than contributing an incorrect one: soft thresholding is not the identity,
    and differentiating as though it were zeroes the whole gradient path
    through the regularizer.

    This wrapper detaches the proximal step's input instead.  The term
    thresholds exactly as before, and the gradient obtained is the one the
    iteration has with this term held fixed.  Its use is the mixed solve, where
    a differentiable denoiser occupies one term and a BART term another, and
    the gradient is required only for the denoiser.

    Examples
    --------
    >>> optim.admm(y, A, [denoiser, priors.frozen(priors.Wavelet((-1, -2), 0.01))])
    """
    return _Frozen(term)

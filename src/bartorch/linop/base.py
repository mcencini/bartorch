"""The linear operator base class and operator algebra.

The classes that combine operators are private.  Composition, addition,
scaling and the adjoint are reached through ``@``, ``+``, ``-``, ``*`` and
``.H``; what those return is an implementation detail, and naming it in a
type annotation or an isinstance check would be reading into the algebra a
structure it does not promise to keep.
"""

from __future__ import annotations

import math
from numbers import Number

import torch

from bartorch._dispatch import BartError
from bartorch._lib import library
from bartorch._operator import Built, Operator, Shape

__all__ = ["LinearOperator"]


def _tracking(x) -> bool:
    return isinstance(x, torch.Tensor) and x.requires_grad and torch.is_grad_enabled()


def _slicing(key, shape: tuple[int, ...]):
    """Where each axis's block starts, how wide it is, and which axes go away.

    What ``A[key]`` has to know to build the restriction, worked out from the
    same key a tensor would take.
    """
    keys = list(key) if isinstance(key, tuple) else [key]

    if sum(1 for k in keys if k is Ellipsis) > 1:
        raise IndexError("only one ... is allowed in an index")
    if Ellipsis in keys:
        at = keys.index(Ellipsis)
        keys[at : at + 1] = [slice(None)] * (len(shape) - (len(keys) - 1))
    if len(keys) > len(shape):
        raise IndexError(f"{len(keys)} indices for an operator with {len(shape)} axes")
    keys += [slice(None)] * (len(shape) - len(keys))

    start, block, drop = [], [], set()
    for axis, (k, size) in enumerate(zip(keys, shape)):
        if isinstance(k, (int,)) and not isinstance(k, bool):
            index = k + size if k < 0 else k
            if not 0 <= index < size:
                raise IndexError(f"index {k} is out of range for axis {axis} of size {size}")
            start.append(index)
            block.append(1)
            drop.add(axis)
        elif isinstance(k, slice):
            if k.step not in (None, 1):
                raise IndexError(
                    "a step other than one has no BART constructor behind it; "
                    "index the result instead"
                )
            begin, stop, _ = k.indices(size)
            if stop < begin:
                raise IndexError(f"axis {axis} would be empty, which is not an operator")
            start.append(begin)
            block.append(stop - begin)
        else:
            raise IndexError(f"an operator takes integers, slices and ..., not {type(k).__name__}")
    return tuple(start), tuple(block), drop


class LinearOperator(Operator):
    """A linear map between two C-order shapes, with an adjoint.

    A subclass is defined either by :meth:`_create`, which builds one of
    BART's operators, or in Python by :meth:`forward` and :meth:`adjoint`, and
    :meth:`normal` where a cheaper form exists.  BART reaches a Python-defined
    operator through callbacks, one crossing into Python per application.

    Both kinds compose into a single BART operator, are solved by
    :mod:`bartorch.optim`, and differentiate in torch.  The backward pass of
    ``A(x)`` is ``A.adjoint``, which for complex tensors is the conjugate
    Wirtinger gradient torch expects, not the transpose.

    Attributes
    ----------
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    device : torch.device or None
        Where the operator does its arithmetic, when that is not where its
        operands are.
    """

    #: Whether BART's closed-form pseudo-inverse for this operator has been
    #: checked against the damped normal equations it claims to solve.  BART
    #: offering one is not enough; see :meth:`pinv`.
    _exact_pinv: bool = False

    _free_name = "bartorch_linop_free"
    _domain_name = "bartorch_linop_domain"
    _codomain_name = "bartorch_linop_codomain"

    def __init__(self):
        if self._native:
            if not self._defers:
                self._build()
        elif (
            type(self).forward is LinearOperator.forward
            or type(self).adjoint is LinearOperator.adjoint
        ):
            raise TypeError(f"{type(self).__name__} must define _create, or forward and adjoint")

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A x``, without recording for autograd.

        ``out`` is written in place when given; for a BART-backed operator it
        must be a contiguous complex64 tensor of :attr:`oshape` on the input's
        device.
        """
        return self._apply(library().bartorch_linop_forward, x, self.ishape, self.oshape, out)

    def adjoint(self, y: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H y``, without recording for autograd."""
        return self._apply(library().bartorch_linop_adjoint, y, self.oshape, self.ishape, out)

    def normal(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A^H A x``.

        A BART-backed operator applies its own normal, which for a
        non-Cartesian encoding built with ``toeplitz=True`` is a convolution
        with a point spread function.  Otherwise ``adjoint(forward(x))``.
        """
        if self._native:
            return self._apply(library().bartorch_linop_normal, x, self.ishape, self.ishape, out)
        return self.adjoint(self.forward(x), out)

    def _as_callbacks(self) -> LinearOperator:
        from bartorch.linop.basic import Callback

        return Callback(self.oshape, self.ishape, self.forward, self.adjoint, self.normal)

    @property
    def plan(self):
        """The encoding form this operator was lowered into, or ``None``.

        An MRI encoding reports what the planner chose for it -- the
        transform, the element-wise factors on each side of it, the
        contraction, what is streamed, how the normal is applied, and which
        executor path ran it.  The algebra carries the plan through, so a
        composition with one encoding in it reports that encoding's plan;
        anything else has none.
        """
        if self._defers and "_plan" not in self.__dict__:
            # A composition works out its plan by lowering, and lowering is
            # what building does, so ask for the handle rather than report
            # the plan of an encoding the sum around it may have changed.
            self._h  # noqa: B018
        own = getattr(self, "_plan", None)
        if own is not None:
            return own
        # By value rather than by part: an adjoint and a normal each hold the
        # operator twice, once as it was given and once as the handle, and two
        # references to one encoding are still one plan.
        found = {
            part.plan
            for part in (getattr(self, name, None) for name in ("a", "b", "op", "source"))
            if isinstance(part, LinearOperator) and part.plan is not None
        }
        return found.pop() if len(found) == 1 else None

    # --- operator algebra ---------------------------------------------------
    #
    # Every one of these is a BART constructor applied to BART operators, so
    # the result is a single operator that BART's solvers drive in their own
    # loop.  Nothing here computes with a tensor.

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """``self @ other`` applies ``other`` first, as one BART operator."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return _Compose(self, other)

    def __add__(self, other: LinearOperator) -> LinearOperator:
        """``self + other``, as one BART operator; the shapes must agree."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return _Add(self, other)

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        """``self - other``, which is ``self + (-other)``."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return _Add(self, -other)

    def __neg__(self) -> LinearOperator:
        """``-self``, BART's scale by minus one chained onto it."""
        return _Scale(-1.0, self.oshape) @ self

    def __mul__(self, other) -> LinearOperator:
        """``c * self`` for a real or complex number ``c``.

        Composition is ``@``, not ``*``: an operator on either side is
        refused rather than taken as composition.
        """
        if isinstance(other, LinearOperator):
            raise TypeError("compose operators with @, not *")
        if not isinstance(other, Number):
            return NotImplemented
        return _Scale(other, self.oshape) @ self

    __rmul__ = __mul__

    def __truediv__(self, other) -> LinearOperator:
        """``self / c``, the scale by its reciprocal."""
        if not isinstance(other, Number) or isinstance(other, LinearOperator):
            return NotImplemented
        return _Scale(1.0 / complex(other), self.oshape) @ self

    def __pow__(self, power: int) -> LinearOperator:
        """``self ** n``, ``n`` applications chained; the operator must be square."""
        if not isinstance(power, int) or isinstance(power, bool):
            return NotImplemented
        if self.ishape != self.oshape:
            raise ValueError(
                f"a power needs a square operator, and this one maps {self.ishape} to {self.oshape}"
            )
        if power < 0:
            raise ValueError("a negative power would be an inverse, which BART does not build")
        from bartorch.linop.basic import Identity

        out: LinearOperator = Identity(self.ishape)
        for _ in range(power):
            out = out @ self
        return out

    def __getitem__(self, key) -> LinearOperator:
        """``A[key]``: the operator whose output is ``A(x)[key]``.

        The restriction is BART's, not a view taken afterwards -- an
        :class:`~bartorch.linop.Extract` chained onto this operator, with a
        :class:`~bartorch.linop.Reshape` after it where an integer index drops
        an axis the way it does for a tensor.  So a slice of an operator is
        still one BART operator, and its adjoint puts the block back and
        leaves the rest zero.

        Integers and slices, one per axis or fewer, with ``...`` standing for
        the axes not named.  A step other than one has no BART constructor
        behind it and is refused rather than emulated.

        Examples
        --------
        >>> coil = CartesianSense(...)  # (coils, y, x)
        >>> first = coil[0]            # (y, x), the first coil
        >>> middle = coil[:, 8:24]     # (coils, 16, x)
        """
        from bartorch.linop.shape import Extract, Reshape

        start, block, drop = _slicing(key, self.oshape)
        out: LinearOperator = Extract(start, block, self.oshape) @ self
        if drop:
            kept = tuple(n for axis, n in enumerate(block) if axis not in drop)
            out = Reshape(kept or (1,), block) @ out
        return out

    @property
    def H(self) -> LinearOperator:  # noqa: N802  (the mathematical name)
        """``A^H``, from BART's own adjoint constructor.

        The result is a BART operator rather than a Python wrapper, so
        ``A.H @ B`` is one operator and its normal is ``A A^H``.
        """
        return _Adjoint(self)

    @property
    def T(self) -> LinearOperator:  # noqa: N802  (the mathematical name)
        """``A^T``, the adjoint without the conjugation, as ``conj(A).H``."""
        return self.conj().H

    def conj(self) -> LinearOperator:
        """``conj(A)``: conjugate the input, apply, conjugate the output."""
        from bartorch.linop.basic import Conj

        return Conj(self.oshape) @ self @ Conj(self.ishape)

    def gram(self) -> LinearOperator:
        """``A^H A`` as an operator.

        BART's own, so an encoding built with ``toeplitz=True`` gives the
        point-spread convolution rather than the two applications.
        """
        return _Normal(self)

    def cogram(self) -> LinearOperator:
        """``A A^H`` as an operator."""
        return _Normal(self.H)

    def opnorm(self) -> float:
        """The spectral norm, by BART's power iteration on ``A^H A``.

        BART starts the iteration from its process-global generator, so this
        returns a slightly different number each time it is called.  It is an
        estimate: what a step size or a Lipschitz constant needs, not an exact
        singular value.
        """
        op = self._bart()
        largest = self._under_lock(library().bartorch_linop_maxeigen, op._h.ptr, device=op.device)
        if largest < 0.0:
            raise BartError("the power iteration did not converge on this operator")
        return math.sqrt(largest)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``A x``, recorded for autograd when ``x`` requires a gradient.

        Parameters
        ----------
        x : torch.Tensor
            Array of :attr:`ishape`.
        out : torch.Tensor, optional
            Contiguous complex64 array of :attr:`oshape` to write into, so that
            repeated applications reuse one buffer.  Not allowed when ``x``
            requires a gradient.
        """
        if _tracking(x):
            if out is not None:
                raise ValueError(
                    "out= writes in place, which autograd cannot record; "
                    "drop it, or detach the input"
                )
            from bartorch.linop.autograd import apply_forward

            return apply_forward(self, x)
        return self.forward(x, out)

    # pyxu's names for the shapes, so that an operator can stand in for one of
    # its LinOps, the way the deepinv names below let it stand in for a
    # LinearPhysics.  ishape and oshape stay the ones this library uses.
    #
    # There is deliberately no flat ``.shape``: an operator here maps a shape
    # to a shape, not a vector of length N to one of length M.  The flat sizes
    # are ``dim_size`` and ``codim_size``.

    @property
    def dim_shape(self) -> tuple[int, ...]:
        """The domain, under pyxu's name for it; the same as :attr:`ishape`."""
        return self.ishape

    @property
    def codim_shape(self) -> tuple[int, ...]:
        """The codomain, under pyxu's name for it; the same as :attr:`oshape`."""
        return self.oshape

    @property
    def dim_size(self) -> int:
        """How many elements the domain holds."""
        return math.prod(self.ishape)

    @property
    def codim_size(self) -> int:
        """How many elements the codomain holds."""
        return math.prod(self.oshape)

    @property
    def dim_rank(self) -> int:
        """How many axes the domain has."""
        return len(self.ishape)

    @property
    def codim_rank(self) -> int:
        """How many axes the codomain has."""
        return len(self.oshape)

    def to_nonlinear(self):
        """The same operator as a :class:`~bartorch.nlop.NonlinearOperator`."""
        from bartorch.nlop.base import FromLinear

        return FromLinear(self)

    def pinv(self, y: torch.Tensor, damp: float = 0.0, **kwargs) -> torch.Tensor:
        """``(A^H A + damp I)^-1 A^H y``, the damped least-squares solution.

        Solved in closed form where BART has one that has been checked, and by
        :class:`bartorch.optim.CG` otherwise; the same quantity either way.

        BART offers the closed form through a ``norm_inv``, which only
        `linops/sum.c` carries, and that routine divides by a count
        `linop_sum_create` overwrites afterwards -- so it answers for a
        differently scaled operator than the one it is attached to.  A class
        therefore opts in through :attr:`_exact_pinv` once a test holds the
        closed form and the solver to the same answer, and only
        :class:`~bartorch.linop.ScaledSum` does.  Everything else takes the
        solver, chains, sums and adjoints included, since those drop the
        ``norm_inv`` regardless.

        Parameters
        ----------
        y : torch.Tensor
            Array of :attr:`oshape`.
        damp : float
            The Tikhonov weight, ``CG``'s ``lambda_``.
        **kwargs
            Passed to :class:`~bartorch.optim.CG`, with ``x0`` as the warm
            start.  Refused when the closed form applies, because there is
            then no solver for them to configure.
        """
        op = self._bart()
        lib = library()

        if self._exact_pinv and lib.bartorch_linop_has_pseudo_inv(op._h.ptr):
            if kwargs:
                raise TypeError(
                    f"{type(self).__name__} has a closed-form pseudo-inverse in BART, so "
                    f"there is no solver to configure; drop {sorted(kwargs)}"
                )

            def solve(ptr, dst, src):
                return lib.bartorch_linop_pseudo_inv(ptr, float(damp), dst, src)

            return op._apply(solve, y, self.oshape, self.ishape)

        from bartorch.optim import CG

        x0 = kwargs.pop("x0", None)
        return CG(damp, **kwargs)(y, self, x0)

    # deepinv's names for the same operations, so that an operator can stand
    # in for a LinearPhysics without deepinv being imported.

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A x``, under ``deepinv``'s name."""
        return self(x)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A^H y``, under ``deepinv``'s name, recorded for autograd."""
        if _tracking(y):
            from bartorch.linop.autograd import apply_adjoint

            return apply_adjoint(self, y)
        return self.adjoint(y)

    def A_adjoint_A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """``A^H A x``, under ``deepinv``'s name, recorded for autograd.

        The recording entry point for :meth:`normal`, which is the raw one.
        A gradient step inside an unrolled network applies ``A^H A``, so
        calling ``normal`` there would leave the data term out of the graph.
        """
        if _tracking(x):
            from bartorch.linop.autograd import apply_normal

            return apply_normal(self, x)
        return self.normal(x)

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802
        """The pseudo-inverse, under ``deepinv``'s name.  See :meth:`pinv`."""
        return self.pinv(y, **kwargs)


class _Composition(LinearOperator):
    """A product or a sum, kept as a description until something needs it.

    Its shapes come from its operands, so composing validates and reports
    without building anything.  The handle is built on first use, and the
    planner is offered the whole description first: an encoding with factors
    and terms around it is lowered into one encoding, and what a solver then
    drives is that rather than the chain this stands for.
    """

    _defers = True

    def __init__(
        self,
        a: LinearOperator,
        b: LinearOperator,
        ishape: Shape,
        oshape: Shape,
        match: bool = True,
    ):
        self.a, self.b = a._bart(), b._bart()
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.device = self.a.device or self.b.device
        self._match = bool(match)
        super().__init__()

    def _chained(self) -> Built:
        """This composition as BART's own operator over the two operands."""
        raise NotImplementedError

    def _create(self) -> Built:
        return self._chained()

    def _build(self) -> None:
        """The lowered encoding where the planner has one, else the plain chain."""
        from dataclasses import replace

        from bartorch.linop import plan

        description = plan.describe(self) if self._match else None
        lowered = None if description is None else plan.lower(description)
        if lowered is None:
            super()._build()
            # A sum the form could not hold is a transform per term, and
            # saying so is the whole point of reporting a plan: without this
            # the search would answer with one term's own plan, which is the
            # silent fallback rather than a report of it.
            if description is not None and len(description.terms) > 1:
                inner = description.terms[0].encoding.plan
                if inner is not None:
                    self._plan = replace(inner, contraction="chained", terms=len(description.terms))
            return

        # The lowered operator answers from here on, and the two share its
        # handle rather than the pointer: one owner frees it once, whichever
        # of them the caller lets go of first.
        self._lowered = lowered
        self._h = lowered._h
        self._plan = lowered.plan
        if tuple(lowered.ishape) != self.ishape or tuple(lowered.oshape) != self.oshape:
            raise ValueError(
                f"the planner lowered {self.ishape}->{self.oshape} into "
                f"{tuple(lowered.ishape)}->{tuple(lowered.oshape)}"
            )


class _Compose(_Composition):
    """``a @ b`` as one BART operator; ``b`` is applied first."""

    def __init__(self, a: LinearOperator, b: LinearOperator, match: bool = True):
        super().__init__(a, b, b.ishape, a.oshape, match)

    def _chained(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_chain,
            self.b._h.ptr,
            self.a._h.ptr,
            device=self.device,
        )
        return Built(ptr, self.b.ishape, self.a.oshape, keep=(self.a, self.b), device=self.device)

    def __repr__(self) -> str:
        return f"({self.a!r} @ {self.b!r})"


class _Add(_Composition):
    """``a + b`` as one BART operator; the two must have the same shapes."""

    def __init__(self, a: LinearOperator, b: LinearOperator, match: bool = True):
        if tuple(a.ishape) != tuple(b.ishape) or tuple(a.oshape) != tuple(b.oshape):
            raise ValueError(
                f"a sum needs one pair of shapes, not {tuple(a.ishape)}->{tuple(a.oshape)} "
                f"and {tuple(b.ishape)}->{tuple(b.oshape)}"
            )
        super().__init__(a, b, a.ishape, a.oshape, match)

    def _chained(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_plus,
            self.a._h.ptr,
            self.b._h.ptr,
            device=self.device,
        )
        return Built(ptr, self.a.ishape, self.a.oshape, keep=(self.a, self.b), device=self.device)

    def __repr__(self) -> str:
        return f"({self.a!r} + {self.b!r})"


class _Adjoint(LinearOperator):
    """``A^H`` as one BART operator, from ``linop_get_adjoint``.

    BART builds this by swapping the operator's own forward and adjoint, so
    the result is a real BART operator: composing it needs no callbacks, and
    its normal is ``A A^H``.
    """

    def __init__(self, op: LinearOperator):
        self.source = op
        self.op = op._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_adjoint_op, self.op._h.ptr, device=self.op.device
        )
        return Built(ptr, self.op.oshape, self.op.ishape, keep=(self.op,), device=self.op.device)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        return self.source

    def __repr__(self) -> str:
        return f"{self.source!r}.H"


class _WithNormal(LinearOperator):
    """``a``, answering ``normal`` when it is asked for ``A^H A``.

    BART derives a normal by chaining the adjoint onto the forward, which is
    the two applications.  Where the product has a closed form -- a sampling
    pattern and a subspace basis collapse into a single kernel applied between
    the transforms, and the frames are never made -- this is how that form is
    attached.  Both sides are BART operators, so the result is one operator
    still.
    """

    def __init__(self, a: LinearOperator, normal: LinearOperator):
        if normal.ishape != a.ishape or normal.oshape != a.ishape:
            raise ValueError(
                f"a normal operator maps the domain to itself, so {a.ishape} to {a.ishape}, "
                f"not {normal.ishape} to {normal.oshape}"
            )
        self.a, self.normal = a._bart(), normal._bart()
        super().__init__()

    def _create(self) -> Built:
        device = self.a.device or self.normal.device
        ptr = self._under_lock(
            library().bartorch_linop_with_normal,
            self.a._h.ptr,
            self.normal._h.ptr,
            device=device,
        )
        return Built(ptr, self.a.ishape, self.a.oshape, keep=(self.a, self.normal), device=device)

    def __repr__(self) -> str:
        return f"{self.a!r}.with_normal({self.normal!r})"


class _Normal(LinearOperator):
    """``A^H A`` as one BART operator, from ``linop_get_normal``."""

    def __init__(self, op: LinearOperator):
        self.source = op
        self.op = op._bart()
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_normal_op, self.op._h.ptr, device=self.op.device
        )
        return Built(ptr, self.op.ishape, self.op.ishape, keep=(self.op,), device=self.op.device)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        # A normal operator is self-adjoint.
        return self

    def __repr__(self) -> str:
        return f"{self.source!r}.gram()"


class _Scale(LinearOperator):
    """Multiplication by one number, BART's ``linop_scale``.

    Private because ``c * A`` is how it is reached, and a scale on its own is
    ``c * Identity(shape)``.
    """

    def __init__(self, value, shape):
        self.value = complex(value)
        self._shape = tuple(shape)
        super().__init__()

    def _create(self) -> Built:
        from bartorch._lib import DIMS
        from bartorch._operator import dims

        ptr = self._under_lock(
            library().bartorch_linop_scale,
            DIMS,
            dims(self._shape),
            float(self.value.real),
            float(self.value.imag),
        )
        return Built(ptr, self._shape, self._shape)

    def __repr__(self) -> str:
        return f"{self.value:g}" if self.value.imag == 0 else f"{self.value}"

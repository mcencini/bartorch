"""BART operators on tensors, and operators defined in Python that BART can drive.

A :class:`LinearOperator` or :class:`NonlinearOperator` wraps one BART
operator.  BART's own building blocks come from the classmethods; an operator
written in Python, whether a torchsim signal model, an mrtoeplitz normal
operator or any autograd-differentiable function, enters through
:meth:`LinearOperator.from_callbacks`, :meth:`NonlinearOperator.from_callbacks`
or :meth:`NonlinearOperator.from_torch`.  Either kind chains with the other
and goes to BART's solvers without leaving C between iterations except for
the callbacks themselves.

Shapes are C order.  Buffers cross the boundary as pointers: a callback sees
the tensor BART is working on, and an application writes into a tensor
allocated here.
"""

from __future__ import annotations

import ctypes
import logging
import traceback
import weakref
from collections.abc import Callable
from typing import Any

import torch

from bartorch import _buffer, _cuda
from bartorch._lib import APPLY_FN, DIMS, library
from bartorch.core.graph import BartError, _ensure_ready, _lock, _on_device

__all__ = ["LinearOperator", "NonlinearOperator"]

_log = logging.getLogger("bartorch.ops")

Shape = tuple[int, ...]


def _dims(shape: Shape) -> ctypes.Array:
    """BART dimension vector, padded to DIMS, of a C-order shape."""
    rev = list(shape)[::-1]
    if len(rev) > DIMS:
        raise ValueError(f"BART supports at most {DIMS} dimensions, got {len(rev)}")
    return (ctypes.c_long * DIMS)(*(rev + [1] * (DIMS - len(rev))))


def _check_dims(query, ptr: int, shape: Shape, what: str) -> None:
    """Verify that BART's view of an operator matches the C-order shape recorded for it."""
    dims = (ctypes.c_long * DIMS)()
    query(ptr, DIMS, dims)
    bart = [int(dims[i]) for i in range(DIMS)]
    if bart != list(_dims(shape)):
        raise BartError(f"operator {what} is {bart[::-1]} in BART but {shape} was recorded")


def _flags(shape: Shape, full: Shape) -> int:
    """BART bitmask of the axes along which ``shape`` is not one, given the full shape."""
    if len(shape) != len(full):
        raise ValueError(f"expected {len(full)} axes, got {len(shape)}")
    bits = 0
    for axis, (d, f) in enumerate(zip(shape, full)):
        if d != 1 and d != f:
            raise ValueError(f"axis {axis} has size {d} but the operator has {f}")
        if d != 1:
            bits |= 1 << (len(full) - 1 - axis)
    return bits


def _axes_flags(axes, ndim: int) -> int:
    from bartorch.utils.flags import _axes_to_flags

    return _axes_to_flags(axes, ndim)


def _view(ptr: int, shape: Shape) -> torch.Tensor:
    """A complex64 tensor over BART's buffer, without a copy."""
    return _buffer.view(ptr, shape)


def _as_operand(x: Any, shape: Shape, what: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.device.type == "cuda" and not _cuda.available():
        raise ValueError(
            f"{what} is on a CUDA device, and this library has no CUDA support built in "
            "or no device is present; move it to the host with .cpu()"
        )
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError(f"{what} is on {x.device}; BART reaches the host and CUDA devices")
    if tuple(x.shape) != tuple(shape):
        try:
            x = x.reshape(shape)
        except RuntimeError as exc:
            raise ValueError(f"{what} has shape {tuple(x.shape)}, expected {shape}") from exc
    return x.to(torch.complex64).contiguous()


def _callback(fn: Callable[[torch.Tensor], torch.Tensor], ishape: Shape, oshape: Shape, name: str):
    def cb(_ctx, dst, src):
        try:
            with torch.no_grad():
                x = _view(src, ishape)
                y = fn(x)
                _view(dst, oshape).copy_(y.reshape(oshape).to(torch.complex64))
            return 0
        except Exception:
            _log.error("%s callback failed:\n%s", name, traceback.format_exc())
            return -1

    return APPLY_FN(cb)


class _Handle:
    """Owns one BART operator handle and everything it must outlive."""

    def __init__(self, ptr: int, free, keep: tuple = ()):
        if not ptr:
            # An operator has no return code to carry a message, so BART's own
            # is read from where the error catcher left it.
            said = library().bartorch_last_error().decode(errors="replace").strip()
            raise BartError(said or "BART could not create the operator")
        self.ptr = ptr
        self._keep = keep
        self._finalizer = weakref.finalize(self, _Handle._release, ptr, free, keep)

    @staticmethod
    def _release(ptr, free, keep):
        with _lock:
            free(ptr)
        del keep


class LinearOperator:
    """A BART linear operator between C-order shapes.

    Parameters
    ----------
    handle : internal
        Use the classmethods to construct one.
    """

    def __init__(self, handle: _Handle, ishape: Shape, oshape: Shape):
        self._h = handle
        self.ishape: Shape = tuple(ishape)
        self.oshape: Shape = tuple(oshape)
        _check_dims(library().bartorch_linop_domain, handle.ptr, self.ishape, "domain")
        _check_dims(library().bartorch_linop_codomain, handle.ptr, self.oshape, "codomain")

    # --- construction ---------------------------------------------------

    @classmethod
    def _create(cls, ptr: int, ishape: Shape, oshape: Shape, keep: tuple = ()) -> LinearOperator:
        return cls(_Handle(ptr, library().bartorch_linop_free, keep), ishape, oshape)

    @classmethod
    def from_callbacks(
        cls,
        oshape: Shape,
        ishape: Shape,
        forward: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
        normal: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> LinearOperator:
        """A linear operator implemented by Python functions on tensors.

        Parameters
        ----------
        oshape, ishape : tuple of int
            Codomain and domain shapes, C order.
        forward, adjoint : callable
            Map a tensor of ``ishape`` to ``oshape`` and back.  Each receives
            a view of BART's buffer and returns a new tensor.
        normal : callable, optional
            ``adjoint(forward(x))`` directly, when a cheaper form exists, such
            as an mrtoeplitz kernel.
        """
        _ensure_ready()
        oshape, ishape = tuple(oshape), tuple(ishape)
        fwd = _callback(forward, ishape, oshape, "forward")
        adj = _callback(adjoint, oshape, ishape, "adjoint")
        nrm = _callback(normal, ishape, ishape, "normal") if normal is not None else None
        with _lock:
            ptr = library().bartorch_linop_callback(
                DIMS,
                _dims(oshape),
                DIMS,
                _dims(ishape),
                fwd,
                adj,
                ctypes.cast(nrm, ctypes.c_void_p) if nrm else None,
                None,
                None,
            )
        return cls._create(ptr, ishape, oshape, (fwd, adj, nrm, forward, adjoint, normal))

    @classmethod
    def fft(
        cls, shape: Shape, axes, inverse: bool = False, centered: bool = True
    ) -> LinearOperator:
        """BART's unitary Fourier transform along ``axes``, centred by default."""
        _ensure_ready()
        shape = tuple(shape)
        flags = _axes_flags(axes, len(shape))
        with _lock:
            ptr = library().bartorch_linop_fft(
                DIMS, _dims(shape), flags, int(inverse), int(centered)
            )
        return cls._create(ptr, shape, shape)

    @classmethod
    def diagonal(cls, diag: torch.Tensor, shape: Shape) -> LinearOperator:
        """Pointwise multiplication by ``diag``, broadcast over the axes where it is one."""
        _ensure_ready()
        shape = tuple(shape)
        d = _as_operand(diag, tuple(diag.shape), "diag")
        flags = _flags(tuple(d.shape), shape)
        with _lock, _on_device(d.device):
            ptr = library().bartorch_linop_cdiag(DIMS, _dims(shape), flags, d.data_ptr())
        return cls._create(ptr, shape, shape, (d,))

    @classmethod
    def multiply_sum(cls, tensor: torch.Tensor, ishape: Shape, oshape: Shape) -> LinearOperator:
        """Multiply by ``tensor`` and sum over the axes absent from ``oshape``.

        This is the coil model: with sensitivities of shape ``(coils, y, x)``,
        ``ishape (1, y, x)`` and ``oshape (coils, y, x)`` it maps an image to
        coil images, and its adjoint combines coil images with the conjugate
        sensitivities.
        """
        _ensure_ready()
        t = _as_operand(tensor, tuple(tensor.shape), "tensor")
        ishape, oshape = tuple(ishape), tuple(oshape)
        with _lock, _on_device(t.device):
            ptr = library().bartorch_linop_fmac(
                DIMS, _dims(oshape), _dims(ishape), _dims(tuple(t.shape)), t.data_ptr()
            )
        return cls._create(ptr, ishape, oshape, (t,))

    @classmethod
    def sampling(cls, pattern: torch.Tensor, shape: Shape) -> LinearOperator:
        """Multiplication by a sampling pattern, broadcast over the axes where it is one."""
        _ensure_ready()
        shape = tuple(shape)
        p = _as_operand(pattern, tuple(pattern.shape), "pattern")
        with _lock, _on_device(p.device):
            ptr = library().bartorch_linop_sampling(
                _dims(shape), _dims(tuple(p.shape)), p.data_ptr()
            )
        return cls._create(ptr, shape, shape, (p,))

    @classmethod
    def nufft(
        cls,
        traj: torch.Tensor,
        image_shape: Shape,
        kspace_shape: Shape | None = None,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        toeplitz: bool = True,
        oversampling: float = 0.0,
        width: float = 0.0,
    ) -> LinearOperator:
        """BART's NUFFT from coil images to samples along ``traj``.

        Parameters
        ----------
        traj : tensor
            Trajectory of shape ``(..., samples, 3)`` in grid units, as
            ``bartorch.tools.traj`` produces.
        image_shape : tuple of int
            Coil-image shape, C order, for instance ``(coils, y, x)``.
        kspace_shape : tuple of int, optional
            Sample shape; by default the trajectory's shape with the
            coordinate axis replaced by the coil axes of ``image_shape``.
        weights : tensor, optional
            A diagonal in k-space the transform is multiplied by on the way
            out and its conjugate on the way back.
        basis : tensor, optional
            A subspace basis over frames and coefficients, which contracts the
            coefficients the images carry into the frames k-space has.  The
            normal is a point spread function over both, which is why these
            belong to the operator rather than to something chained onto it.
        toeplitz : bool
            Apply the normal operator through the Toeplitz embedding.
        oversampling, width : float
            Grid oversampling and kernel width; zero keeps BART's defaults.
        """
        _ensure_ready()
        t = _as_operand(traj, tuple(traj.shape), "traj")
        image_shape = tuple(image_shape)
        if kspace_shape is None:
            from bartorch import _finufft

            kspace_shape = _default_kspace_shape(
                tuple(t.shape), image_shape, _finufft.spatial_ndim(t)
            )
        w = None if weights is None else _as_operand(weights, tuple(weights.shape), "weights")
        b = None if basis is None else _as_operand(basis, tuple(basis.shape), "basis")
        with _lock, _on_device(t.device):
            ptr = library().bartorch_linop_nufft(
                DIMS,
                _dims(tuple(kspace_shape)),
                _dims(image_shape),
                _dims(tuple(t.shape)),
                t.data_ptr(),
                None if w is None else _dims(tuple(w.shape)),
                None if w is None else w.data_ptr(),
                None if b is None else _dims(tuple(b.shape)),
                None if b is None else b.data_ptr(),
                int(toeplitz),
                float(oversampling),
                float(width),
            )
        keep = tuple(x for x in (t, w, b) if x is not None)
        return cls._create(ptr, image_shape, tuple(kspace_shape), keep)

    @classmethod
    def sense(
        cls,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        traj: torch.Tensor | None = None,
        kspace_shape: Shape | None = None,
        kernels: bool = False,
        toeplitz: bool = True,
    ) -> LinearOperator:
        """Sensitivities and a transform, over one slab of coils at a time.

        The coils are independent until the sum that ends the adjoint, so this
        applies a slab of sensitivities, transforms that slab and sums it in.
        What is resident is a slab rather than the whole bank, and the
        transform is built for a slab, so the grid its Toeplitz normal
        convolves on shrinks with it.  ``bartorch.set_coil_batch`` is how many
        coils a slab holds.

        Parameters
        ----------
        sensitivities : tensor
            Coil sensitivities of shape ``(coils, *image_shape[1:])``, or the
            k-space kernels they band-limit to when ``kernels`` is set.
        image_shape : tuple of int
            Coil-image shape, C order, for instance ``(coils, y, x)``.
        traj : tensor, optional
            Trajectory in grid units; without one this is the Cartesian
            operator and the transform is the centred unitary FFT, which is
            ``bartorch.tools.fft(..., unitary=True)``.
        kspace_shape : tuple of int, optional
            Sample shape; by default the trajectory's, or the image's on a
            grid.
        kernels : bool
            Sensitivities left on the host are brought over a slab at a time
            when the operator is applied on a card, so the bank itself never
            has to fit; what crosses is one slab.

        kernels : bool
            Read ``sensitivities`` as kernels: the centre of each map's
            spectrum, which is all a smooth map carries.  A slab is padded
            back on to the image grid and transformed when it is needed, so a
            bank that would not fit is never held.  What the operator applies
            is then the maps band-limited to the kernel, which
            :func:`bartorch.maps_to_kernels` reports the error of.
        toeplitz : bool
            Apply the normal through the Toeplitz embedding.
        """
        _ensure_ready()
        image_shape = tuple(image_shape)
        if len(image_shape) < 3:
            raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")

        # BART reads the coils off a dimension of their own, which sits after
        # the three spatial ones, so a two-dimensional problem carries the
        # third as a singleton.  The caller need not write it out.
        coils, spatial = image_shape[0], image_shape[1:]
        if len(spatial) == 2:
            spatial = (1, *spatial)

        s = _as_operand(sensitivities, tuple(sensitivities.shape), "sensitivities")
        if s.shape[0] != coils:
            raise ValueError(f"{s.shape[0]} sensitivities for {coils} coils")
        sens_spatial = tuple(s.shape[1:])
        if len(sens_spatial) == 2:
            sens_spatial = (1, *sens_spatial)
            s = s.reshape(coils, *sens_spatial)
        t = None if traj is None else _as_operand(traj, tuple(traj.shape), "traj")

        max_shape = (coils, *spatial)

        if kspace_shape is None:
            if t is None:
                kspace_shape = max_shape
            else:
                # BART lays non-Cartesian samples out with the read axis a
                # singleton and the coils where they always are, so the coil
                # axis lines up with the sensitivities rather than landing on
                # the one that carries sets of maps.
                kspace_shape = (coils, *tuple(t.shape)[:-1], 1)
        kspace_shape = tuple(kspace_shape)

        # Where the operator is built follows the transform's own data, not
        # the sensitivities: a bank left on the host is the point.
        built_on = s.device if t is None else t.device

        with _lock, _on_device(built_on):
            ptr = library().bartorch_linop_sense(
                _dims(max_shape),
                _dims(kspace_shape),
                _dims((coils, *sens_spatial)),
                s.data_ptr(),
                int(kernels),
                None if t is None else _dims(tuple(t.shape)),
                None if t is None else t.data_ptr(),
                int(toeplitz),
            )
        keep = tuple(x for x in (s, t) if x is not None)
        return cls._create(ptr, spatial, kspace_shape, keep)

    # --- algebra --------------------------------------------------------

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """``self @ other`` applies ``other`` first."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        with _lock:
            ptr = library().bartorch_linop_chain(other._h.ptr, self._h.ptr)
        return LinearOperator._create(ptr, other.ishape, self.oshape, (self, other))

    def __add__(self, other: LinearOperator) -> LinearOperator:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        with _lock:
            ptr = library().bartorch_linop_plus(self._h.ptr, other._h.ptr)
        return LinearOperator._create(ptr, self.ishape, self.oshape, (self, other))

    # --- application ----------------------------------------------------

    def _apply(self, fn, x: torch.Tensor, ishape: Shape, oshape: Shape) -> torch.Tensor:
        x = _as_operand(x, ishape, "input")
        y = torch.empty(oshape, dtype=torch.complex64, device=x.device)
        with _lock, _on_device(x.device):
            if fn(self._h.ptr, y.data_ptr(), x.data_ptr()) != 0:
                raise BartError("operator application failed; see the log for BART's message")
        return y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_linop_forward, x, self.ishape, self.oshape)

    forward = __call__

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_linop_adjoint, y, self.oshape, self.ishape)

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_linop_normal, x, self.ishape, self.ishape)

    # --- solving --------------------------------------------------------

    def lstsq(
        self,
        y: torch.Tensor,
        lambda_: float = 0.0,
        maxiter: int = 30,
        tol: float = 1e-6,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve ``min ||A x - y||^2 + lambda ||x||^2`` by BART's conjugate gradients."""
        y = _as_operand(y, self.oshape, "y")
        if x0 is None:
            x = torch.zeros(self.ishape, dtype=torch.complex64, device=y.device)
        else:
            x = _as_operand(x0, self.ishape, "x0").clone()
        with _lock, _on_device(y.device):
            code = library().bartorch_lsqr(
                self._h.ptr,
                int(maxiter),
                float(lambda_),
                float(tol),
                int(x0 is not None),
                x.data_ptr(),
                y.data_ptr(),
            )
        if code != 0:
            raise BartError("least-squares solve failed; see the log for BART's message")
        return x

    def to_nonlinear(self) -> NonlinearOperator:
        with _lock:
            ptr = library().bartorch_nlop_from_linop(self._h.ptr)
        return NonlinearOperator._create(ptr, self.ishape, self.oshape, (self,))

    def __repr__(self) -> str:
        return f"LinearOperator({self.ishape} -> {self.oshape})"


def _default_kspace_shape(traj_shape: Shape, image_shape: Shape, ndim: int) -> Shape:
    # BART: traj dims [3, samples, spokes, ...], kspace dims [1, samples, spokes, coils, ...].
    # In C order the trajectory is (..., spokes, samples, 3) and the coil axes of
    # the image sit in front of its spatial ones -- two or three of them, which
    # only the trajectory can say, since (coils, y, x) and (z, y, x) are the same
    # shape.
    coils = tuple(image_shape[:-ndim])
    return coils + tuple(traj_shape[:-1]) + (1,)


class NonlinearOperator:
    """A BART nonlinear operator with a derivative and its adjoint."""

    def __init__(self, handle: _Handle, ishape: Shape, oshape: Shape):
        self._h = handle
        self.ishape: Shape = tuple(ishape)
        self.oshape: Shape = tuple(oshape)
        _check_dims(library().bartorch_nlop_domain, handle.ptr, self.ishape, "domain")
        _check_dims(library().bartorch_nlop_codomain, handle.ptr, self.oshape, "codomain")

    @classmethod
    def _create(cls, ptr: int, ishape: Shape, oshape: Shape, keep: tuple = ()) -> NonlinearOperator:
        return cls(_Handle(ptr, library().bartorch_nlop_free, keep), ishape, oshape)

    @classmethod
    def from_callbacks(
        cls,
        oshape: Shape,
        ishape: Shape,
        forward: Callable[[torch.Tensor], torch.Tensor],
        derivative: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
    ) -> NonlinearOperator:
        """A nonlinear operator implemented by Python functions on tensors.

        ``forward(x)`` evaluates the operator and fixes the point at which
        ``derivative(dx)`` and ``adjoint(dy)`` are taken until the next
        forward call, which is how BART's solvers use them.
        """
        _ensure_ready()
        oshape, ishape = tuple(oshape), tuple(ishape)
        fwd = _callback(forward, ishape, oshape, "forward")
        der = _callback(derivative, ishape, oshape, "derivative")
        adj = _callback(adjoint, oshape, ishape, "adjoint")
        with _lock:
            ptr = library().bartorch_nlop_callback(
                DIMS, _dims(oshape), DIMS, _dims(ishape), fwd, der, adj, None, None
            )
        return cls._create(ptr, ishape, oshape, (fwd, der, adj, forward, derivative, adjoint))

    @classmethod
    def from_torch(
        cls, fn: Callable[[torch.Tensor], torch.Tensor], ishape: Shape, oshape: Shape
    ) -> NonlinearOperator:
        """A nonlinear operator from a differentiable torch function.

        The derivative is the forward-mode Jacobian-vector product and its
        adjoint the reverse-mode vector-Jacobian product at the last point the
        operator was evaluated, so a torchsim signal model or any other
        autograd-differentiable map can be fitted by BART's Gauss-Newton
        solver.
        """
        ishape, oshape = tuple(ishape), tuple(oshape)
        state: dict[str, torch.Tensor] = {}

        def forward(x: torch.Tensor) -> torch.Tensor:
            xr = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                y = fn(xr)
            state["x"], state["y"] = xr, y
            return y.detach()

        def derivative(dx: torch.Tensor) -> torch.Tensor:
            x = state["x"].detach()
            with torch.enable_grad():
                _, jvp = torch.func.jvp(fn, (x,), (dx.detach().clone(),))
            return jvp

        def adjoint(dy: torch.Tensor) -> torch.Tensor:
            x, y = state["x"], state["y"]
            with torch.enable_grad():
                (g,) = torch.autograd.grad(
                    y, x, grad_outputs=dy.detach().clone(), retain_graph=True
                )
            return g

        return cls.from_callbacks(oshape, ishape, forward, derivative, adjoint)

    def __matmul__(self, other: NonlinearOperator | LinearOperator) -> NonlinearOperator:
        """``self @ other`` applies ``other`` first."""
        if isinstance(other, LinearOperator):
            other = other.to_nonlinear()
        if not isinstance(other, NonlinearOperator):
            return NotImplemented
        with _lock:
            ptr = library().bartorch_nlop_chain(other._h.ptr, self._h.ptr)
        return NonlinearOperator._create(ptr, other.ishape, self.oshape, (self, other))

    def __rmatmul__(self, other: LinearOperator) -> NonlinearOperator:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        return other.to_nonlinear() @ self

    def _apply(self, fn, x: torch.Tensor, ishape: Shape, oshape: Shape) -> torch.Tensor:
        x = _as_operand(x, ishape, "input")
        y = torch.empty(oshape, dtype=torch.complex64, device=x.device)
        with _lock, _on_device(x.device):
            if fn(self._h.ptr, y.data_ptr(), x.data_ptr()) != 0:
                raise BartError("operator application failed; see the log for BART's message")
        return y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(library().bartorch_nlop_apply, x, self.ishape, self.oshape)

    def derivative(self, dx: torch.Tensor) -> torch.Tensor:
        """The derivative at the last evaluated point applied to ``dx``."""
        return self._apply(library().bartorch_nlop_derivative, dx, self.ishape, self.oshape)

    def adjoint(self, dy: torch.Tensor) -> torch.Tensor:
        """The adjoint of the derivative at the last evaluated point applied to ``dy``."""
        return self._apply(library().bartorch_nlop_adjoint, dy, self.oshape, self.ishape)

    def irgnm(
        self,
        y: torch.Tensor,
        x0: torch.Tensor,
        iterations: int = 8,
        alpha: float = 1.0,
        alpha_min: float = 0.0,
        redu: float = 2.0,
        cgiter: int = 30,
        cgtol: float = 0.0,
        xref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit ``self(x) = y`` from ``x0`` by BART's iteratively regularised Gauss-Newton.

        Parameters
        ----------
        y : tensor
            Data of shape ``oshape``.
        x0 : tensor
            Starting point of shape ``ishape``; also the regularisation
            centre unless ``xref`` is given.
        iterations : int
            Gauss-Newton steps.
        alpha, alpha_min, redu : float
            Initial regularisation weight, its floor, and the factor it is
            divided by after each step.
        cgiter, cgtol : int, float
            Conjugate-gradient budget and tolerance for each linearised step.
        """
        y = _as_operand(y, self.oshape, "y")
        x = _as_operand(x0, self.ishape, "x0").clone()
        ref = _as_operand(xref, self.ishape, "xref") if xref is not None else None
        with _lock, _on_device(y.device):
            code = library().bartorch_irgnm(
                self._h.ptr,
                int(iterations),
                float(alpha),
                float(alpha_min),
                float(redu),
                int(cgiter),
                float(cgtol),
                x.data_ptr(),
                y.data_ptr(),
                ref.data_ptr() if ref is not None else None,
            )
        if code != 0:
            raise BartError("Gauss-Newton solve failed; see the log for BART's message")
        return x

    def __repr__(self) -> str:
        return f"NonlinearOperator({self.ishape} -> {self.oshape})"

"""Handle, lifetime and marshalling shared by linear and nonlinear operators."""

from __future__ import annotations

import logging
import traceback
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from bartorch import _buffer, _cuda, _marshal
from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import APPLY_FN, DIMS, library

__all__ = ["Built", "Operator", "Shape"]

_log = logging.getLogger("bartorch.operator")

Shape = tuple[int, ...]

#: The BART dimension vector of a C-order shape, padded to all sixteen axes.
dims = _marshal.padded_dims


@dataclass
class Built:
    """What :meth:`Operator._create` returns.

    Attributes
    ----------
    ptr : int
        The BART operator.  Zero means BART declined, and the reason is read
        from the error catcher.
    ishape, oshape : tuple of int
        Domain and codomain, C order.
    keep : tuple
        Objects the operator holds by pointer -- trajectory, sensitivities,
        callbacks.  They are released after the handle is freed.
    device : torch.device, optional
        Where the operator does its arithmetic, when that is not where its
        operands are.
    """

    ptr: int
    ishape: Shape
    oshape: Shape
    keep: tuple = ()
    device: torch.device | None = None


class _Handle:
    """Owns one BART operator handle and everything it must outlive."""

    def __init__(self, ptr: int, free, keep: tuple = ()):
        if not ptr:
            # An operator has no return code to carry a message, so BART's own
            # is read from where the error catcher left it.
            said = library().bartorch_last_error().decode(errors="replace").strip()
            raise BartError(said or "BART could not create the operator")
        self.ptr = ptr
        self._finalizer = weakref.finalize(self, _Handle._release, ptr, free, keep)

    @staticmethod
    def _release(ptr, free, keep):
        with _lock:
            free(ptr)
        del keep


def check_dims(query, ptr: int, shape: Shape, what: str) -> None:
    """Raise if BART's dimensions for an operator differ from the recorded C-order shape."""
    vector = _marshal.dim_vector()
    query(ptr, DIMS, vector)
    bart = [int(vector[i]) for i in range(DIMS)]
    if bart != list(dims(shape)):
        raise BartError(f"operator {what} is {bart[::-1]} in BART but {shape} was recorded")


def broadcast_flags(shape: Shape, full: Shape) -> int:
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


def axes_flags(axes, ndim: int) -> int:
    """BART bitmask of a C-order axis index or tuple of them."""
    from bartorch._flags import _axes_to_flags

    return _axes_to_flags(axes, ndim)


def as_operand(x: Any, shape: Shape, what: str) -> torch.Tensor:
    """A contiguous complex64 tensor of ``shape``, on a device BART can reach."""
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


def callback(fn: Callable[[torch.Tensor], torch.Tensor], ishape: Shape, oshape: Shape, name: str):
    """``fn`` as a BART apply callback.

    ``fn`` receives a view of BART's source buffer, without a copy, and its
    result is copied into BART's destination buffer as complex64.  An
    exception is logged and reported to BART as a failure.
    """

    def cb(_ctx, dst, src):
        try:
            with torch.no_grad():
                x = _buffer.view(src, ishape)
                y = fn(x)
                _buffer.view(dst, oshape).copy_(y.reshape(oshape).to(torch.complex64))
            return 0
        except Exception:
            _log.error("%s callback failed:\n%s", name, traceback.format_exc())
            return -1

    return APPLY_FN(cb)


class Operator:
    """Base of the linear and nonlinear operator classes.

    A subclass that defines :meth:`_create` is backed by a BART operator,
    built by :meth:`_build`.  One that does not is defined in Python and has
    no handle; :meth:`_bart` wraps its methods as callbacks whenever BART needs
    a handle, and the wrapper is not cached.
    """

    #: Library functions that free this kind of handle and report its shapes.
    _free_name: str = ""
    _domain_name: str = ""
    _codomain_name: str = ""

    ishape: Shape
    oshape: Shape
    device: torch.device | None = None

    @property
    def _native(self) -> bool:
        return type(self)._create is not Operator._create

    def _create(self) -> Built:
        """Build BART's operator from what the subclass recorded in ``__init__``.

        Everything the operator holds by pointer goes in :attr:`Built.keep`.
        """
        raise NotImplementedError

    def _build(self) -> None:
        _ensure_ready()
        built = self._create()
        lib = library()
        self._h = _Handle(built.ptr, getattr(lib, self._free_name), built.keep)
        self.ishape = tuple(built.ishape)
        self.oshape = tuple(built.oshape)
        self.device = built.device
        check_dims(getattr(lib, self._domain_name), self._h.ptr, self.ishape, "domain")
        check_dims(getattr(lib, self._codomain_name), self._h.ptr, self.oshape, "codomain")

    def _bart(self) -> Operator:
        """This operator if BART-backed, otherwise a new callback wrapper around it."""
        return self if self._native else self._as_callbacks()

    def _as_callbacks(self) -> Operator:
        raise NotImplementedError

    @staticmethod
    def _under_lock(fn, *args, device: torch.device | None = None) -> int:
        """Call one of BART's constructors under the lock, on ``device``."""
        _ensure_ready()
        with _lock, _on_device(device or torch.device("cpu")):
            return fn(*args)

    def _apply(
        self, fn, x: torch.Tensor, ishape: Shape, oshape: Shape, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self._native:
            raise NotImplementedError(
                f"{type(self).__name__} defines neither _create nor this method"
            )
        x = as_operand(x, ishape, "input")
        if out is None:
            y = torch.empty(oshape, dtype=torch.complex64, device=x.device)
        else:
            if (
                tuple(out.shape) != tuple(oshape)
                or out.dtype != torch.complex64
                or out.device != x.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    f"out must be a contiguous complex64 tensor of shape "
                    f"{tuple(oshape)} on {x.device}"
                )
            y = out
        with _lock, _on_device(self.device or x.device):
            if fn(self._h.ptr, y.data_ptr(), x.data_ptr()) != 0:
                raise BartError("operator application failed; see the log for BART's message")
        return y

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ishape} -> {self.oshape})"

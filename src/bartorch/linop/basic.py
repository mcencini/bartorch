"""BART's building-block linear operators, one class each.

Each is the BART constructor it names and nothing around it: the shapes are C
order, an axis argument is an index into that shape rather than a bitmask, and
what the operator holds by pointer is kept alive with it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from bartorch import _marshal
from bartorch._lib import DIMS, library
from bartorch._operator import (
    Built,
    Shape,
    as_operand,
    axes_flags,
    broadcast_flags,
    callback,
    dims,
)
from bartorch.linop.base import BartLinearOperator

__all__ = ["Callback", "Diagonal", "FFT", "MultiplySum", "Sampling"]


class FFT(BartLinearOperator):
    """BART's unitary Fourier transform along ``axes``, centred by default.

    Parameters
    ----------
    shape : tuple of int
        The shape it transforms, C order.
    axes : int or tuple of int
        Which axes to transform, as indices into ``shape``; negative indices
        count from the end.
    inverse : bool
        Transform the other way.
    centred : bool
        Put the zero frequency in the middle, which is BART's ``fftc``.

    Examples
    --------
    >>> F = FFT((8, 16), axes=(-1, -2))
    >>> F(x).shape
    torch.Size([8, 16])
    """

    def __init__(self, shape: Shape, axes, inverse: bool = False, centred: bool = True, **kwargs):
        # ``centered`` was the spelling before, and still is accepted.
        if "centered" in kwargs:
            centred = kwargs.pop("centered")
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        self.shape = tuple(shape)
        self.axes = axes
        self.inverse = bool(inverse)
        self.centred = bool(centred)
        super().__init__()

    def _create(self) -> Built:
        flags = axes_flags(self.axes, len(self.shape))
        ptr = self._under_lock(
            library().bartorch_linop_fft,
            DIMS,
            dims(self.shape),
            flags,
            int(self.inverse),
            int(self.centred),
        )
        return Built(ptr, self.shape, self.shape)


class Diagonal(BartLinearOperator):
    """Pointwise multiplication by ``diag``, broadcast over the axes where it is one.

    BART's ``cdiag``.

    Parameters
    ----------
    diag : tensor
        The diagonal.  Every axis is either the operator's size along that
        axis or one, and the ones are broadcast.
    shape : tuple of int
        The shape the operator works on, C order.
    """

    def __init__(self, diag: torch.Tensor, shape: Shape):
        self.shape = tuple(shape)
        self.diag = as_operand(diag, tuple(diag.shape), "diag")
        super().__init__()

    def _create(self) -> Built:
        flags = broadcast_flags(tuple(self.diag.shape), self.shape)
        ptr = self._under_lock(
            library().bartorch_linop_cdiag,
            DIMS,
            dims(self.shape),
            flags,
            self.diag.data_ptr(),
            device=self.diag.device,
        )
        return Built(ptr, self.shape, self.shape, keep=(self.diag,))


class Sampling(BartLinearOperator):
    """Multiplication by a sampling pattern, broadcast over the axes where it is one.

    Parameters
    ----------
    pattern : tensor
        Ones where a sample was taken and zeros where it was not.
    shape : tuple of int
        The k-space shape, C order.
    """

    def __init__(self, pattern: torch.Tensor, shape: Shape):
        self.shape = tuple(shape)
        self.pattern = as_operand(pattern, tuple(pattern.shape), "pattern")
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_sampling,
            dims(self.shape),
            dims(tuple(self.pattern.shape)),
            self.pattern.data_ptr(),
            device=self.pattern.device,
        )
        return Built(ptr, self.shape, self.shape, keep=(self.pattern,))


class MultiplySum(BartLinearOperator):
    """Multiply by a tensor and sum over the axes absent from the codomain.

    BART's ``fmac``.  This is the coil model: with sensitivities of shape
    ``(coils, y, x)``, ``ishape (1, y, x)`` and ``oshape (coils, y, x)`` it
    maps an image to coil images, and its adjoint combines coil images with
    the conjugate sensitivities.

    Parameters
    ----------
    tensor : tensor
        What to multiply by.
    ishape, oshape : tuple of int
        Domain and codomain, C order.  An axis the domain has and the codomain
        does not is summed over.
    """

    def __init__(self, tensor: torch.Tensor, ishape: Shape, oshape: Shape):
        self.tensor = as_operand(tensor, tuple(tensor.shape), "tensor")
        # ``Operator.__init__`` sets these again from what _create returns; a
        # concrete operator may name them itself so that _create can read them.
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(
            library().bartorch_linop_fmac,
            DIMS,
            dims(self.oshape),
            dims(self.ishape),
            dims(tuple(self.tensor.shape)),
            self.tensor.data_ptr(),
            device=self.tensor.device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.tensor,))


class Callback(BartLinearOperator):
    """A linear operator implemented by Python functions on tensors.

    What it is for is the other direction: an operator written here that
    BART's own solvers and BART's own composition can be given.  Each callback
    receives a view of BART's buffer, without a copy, and returns a tensor.

    Parameters
    ----------
    oshape, ishape : tuple of int
        Codomain and domain shapes, C order.
    forward, adjoint : callable
        Map a tensor of ``ishape`` to ``oshape`` and back.
    normal : callable, optional
        ``adjoint(forward(x))`` directly, when a cheaper form exists, such as
        an mrtoeplitz kernel.  Without one BART composes the two.
    """

    def __init__(
        self,
        oshape: Shape,
        ishape: Shape,
        forward: Callable[[torch.Tensor], torch.Tensor],
        adjoint: Callable[[torch.Tensor], torch.Tensor],
        normal: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.oshape, self.ishape = tuple(oshape), tuple(ishape)
        self.forward_fn, self.adjoint_fn, self.normal_fn = forward, adjoint, normal
        super().__init__()

    def _create(self) -> Built:
        ishape, oshape = self.ishape, self.oshape
        fwd = callback(self.forward_fn, ishape, oshape, "forward")
        adj = callback(self.adjoint_fn, oshape, ishape, "adjoint")
        nrm = (
            callback(self.normal_fn, ishape, ishape, "normal")
            if self.normal_fn is not None
            else _marshal.null_apply()
        )
        ptr = self._under_lock(
            library().bartorch_linop_callback,
            DIMS,
            dims(oshape),
            DIMS,
            dims(ishape),
            fwd,
            adj,
            nrm,
            None,
            _marshal.null_release(),
        )
        keep = (fwd, adj, nrm, self.forward_fn, self.adjoint_fn, self.normal_fn)
        return Built(ptr, ishape, oshape, keep=keep)

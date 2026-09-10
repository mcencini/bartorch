"""A tensor over one of BART's buffers, without a copy.

A callback is handed a raw pointer and the shape BART expects there.  Where
that memory is decides how it is wrapped: a host pointer through the buffer
protocol, a device pointer through the CUDA array interface, which torch reads
as a real pair per element and :func:`torch.view_as_complex` folds back.
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch import _marshal

Shape = tuple[int, ...]


class _DeviceArray:
    """The CUDA array interface over ``ptr``, as pairs of floats."""

    def __init__(self, ptr: int, numel: int):
        self.__cuda_array_interface__ = {
            "shape": (numel, 2),
            "typestr": "<f4",
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


def view(ptr: int, shape: Shape) -> torch.Tensor:
    """A complex64 tensor over BART's buffer at *ptr*, shaped as asked."""
    numel = 1
    for size in shape:
        numel *= size
    if numel == 0:
        return torch.empty(shape, dtype=torch.complex64)

    from bartorch._lib import library

    lib = library()
    if lib.bartorch_on_device(ptr):
        device = lib.bartorch_cuda_device()
        pairs = torch.as_tensor(
            _DeviceArray(ptr, numel), device=torch.device("cuda", max(device, 0))
        )
        return torch.view_as_complex(pairs).reshape(shape)

    buf = _marshal.float_buffer(ptr, 2 * numel)
    return torch.frombuffer(buf, dtype=torch.complex64).reshape(shape)


def real_view(ptr: int, numel: int) -> torch.Tensor:
    """A float32 tensor over *numel* floats at *ptr*, wherever they are."""
    if numel == 0:
        return torch.empty(0)

    from bartorch._lib import library

    lib = library()
    if lib.bartorch_on_device(ptr):
        device = lib.bartorch_cuda_device()
        pairs = torch.as_tensor(
            _DeviceArray(ptr, (numel + 1) // 2), device=torch.device("cuda", max(device, 0))
        )
        return pairs.reshape(-1)[:numel]

    buf = _marshal.float_buffer(ptr, numel)
    return torch.frombuffer(buf, dtype=torch.float32)


def as_pointer(x: Any) -> int:
    """The address a pointer argument carries, or zero."""
    return _marshal.address_of(x)

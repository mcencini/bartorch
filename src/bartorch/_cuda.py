"""Running BART on a CUDA device.

BART keeps its own streams and asks the driver whether a pointer is on a
device, so a torch CUDA tensor needs no registration and an array BART creates
comes from the allocator callback on the device the caller selected.  What is
left is ordering: :func:`ordered` holds BART's streams until the work already
queued on torch's current stream has run, and holds torch's stream until
BART's work has, so neither side synchronises the device.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from bartorch._lib import library

__all__ = [
    "available",
    "device",
    "device_count",
    "free_memory",
    "ordered",
    "set_streams",
    "streams",
    "use_memcache",
]


def built() -> bool:
    """Whether this library was compiled with BART's CUDA kernels."""
    return bool(library().bartorch_cuda_built())


def available() -> bool:
    """Whether BART can run on a device here: compiled in, and one present."""
    return built() and library().bartorch_cuda_device_count() > 0 and torch.cuda.is_available()


def device_count() -> int:
    return int(library().bartorch_cuda_device_count())


def device() -> int:
    """The device BART is running on, or -1 when it is on the host.

    BART is pointed at a card only for the duration of a call, so this is -1
    everywhere except inside :func:`ordered`.
    """
    return int(library().bartorch_cuda_device())


def free_memory() -> int:
    """Free device memory in bytes, or -1 when there is no device."""
    return int(library().bartorch_cuda_free_memory())


def streams() -> int:
    """Number of CUDA streams BART runs on; one by default."""
    return int(library().bartorch_cuda_get_streams())


def set_streams(n: int) -> None:
    """Set the number of CUDA streams BART runs on, 1 to 8.

    More than one overlaps transfers with arithmetic.
    """
    if library().bartorch_cuda_set_streams(int(n)) != 0:
        raise ValueError(f"BART takes between 1 and 8 streams, not {n}")


def use_memcache(enable: bool) -> None:
    """Whether BART keeps freed device memory for its own reuse.

    Off, memory goes back to the driver as soon as BART is done with it, where
    torch's allocator can take it, at the cost of slower allocation.  A SENSE
    operator whose caller keeps its arrays on the host returns the cache after
    each application either way.
    """
    library().bartorch_cuda_use_memcache(int(bool(enable)))


@contextmanager
def ordered(device: torch.device):
    """Run BART on *device*, ordered against torch's current stream.

    Entering points BART at the device and holds its streams until what torch
    has already queued has run; leaving holds torch's stream until BART's work
    has and puts BART back on the host.
    """
    lib = library()
    index = device.index if device.index is not None else torch.cuda.current_device()
    if lib.bartorch_cuda_enable(index) != 0:
        raise RuntimeError(f"BART could not use CUDA device {index}")
    stream = torch.cuda.current_stream(index)
    try:
        lib.bartorch_cuda_wait_for_stream(stream.cuda_stream)
        yield
        lib.bartorch_cuda_signal_stream(stream.cuda_stream)
    finally:
        lib.bartorch_cuda_enable(-1)

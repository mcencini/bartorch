"""Process-wide settings, build information and CUDA controls, exported by :mod:`bartorch`."""

from __future__ import annotations

from bartorch import _cuda
from bartorch._dispatch import (
    get_debug_level,
    set_copy_inputs,
    set_debug_level,
    set_num_threads,
)

__all__ = [
    "backend_sources",
    "bart_version",
    "build_info",
    "cuda_available",
    "get_debug_level",
    "set_copy_inputs",
    "set_cuda_streams",
    "set_debug_level",
    "set_num_threads",
    "use_cuda_memcache",
]


def bart_version() -> str:
    """Version string of the embedded BART."""
    from bartorch._lib import library

    return library().bartorch_bart_version().decode()


def build_info() -> str:
    """How the library was built: BART version, compiler, nested-function mode, CUDA."""
    from bartorch._lib import library

    return library().bartorch_build_info().decode()


def backend_sources() -> dict[str, str]:
    """The library serving each BLAS and LAPACK routine, and the FFT."""
    from bartorch import _backend
    from bartorch._dispatch import _ensure_ready

    _ensure_ready()
    return _backend.sources()


def cuda_available() -> bool:
    """Whether BART can run on a CUDA device: built with CUDA, and a device present."""
    return _cuda.available()


def set_cuda_streams(n: int) -> None:
    """Set the number of CUDA streams BART runs on, 1 to 8.

    More than one lets BART overlap its transfers with its arithmetic.
    """
    _cuda.set_streams(n)


def use_cuda_memcache(enable: bool) -> None:
    """Whether BART keeps freed device memory for its own reuse.

    Off, memory goes back to the driver as soon as BART is done with it, where
    torch's allocator can take it, at the cost of slower allocation.
    """
    _cuda.use_memcache(enable)

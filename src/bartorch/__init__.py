"""bartorch: the Berkeley Advanced Reconstruction Toolbox, in-process, on tensors.

Every BART tool is a function in :mod:`bartorch.tools` that takes and returns
``torch.Tensor`` objects.  Shapes follow C order, so the last axis is the one
BART calls the first; an axis argument is an index into that shape and a
bitmask is never needed.

The tools run inside the compiled library in this package, which carries all
of BART.  BLAS and LAPACK come from whatever the process already loaded,
which with torch installed is the library torch links; the FFT is pocketfft.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from bartorch import cuda, finufft
from bartorch.core.graph import (
    BartError,
    coil_batch,
    fold_maps,
    dispatch,
    get_debug_level,
    kernels_to_maps,
    maps_to_kernels,
    run_command,
    set_coil_batch,
    set_fold_maps,
    set_copy_inputs,
    set_debug_level,
    set_num_threads,
)

try:
    __version__ = version("bartorch")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


def bart_version() -> str:
    """The version string of the embedded BART."""
    from bartorch._lib import library

    return library().bartorch_bart_version().decode()


def build_info() -> str:
    """How the compiled library was built: BART version, compiler, nested-function mode."""
    from bartorch._lib import library

    return library().bartorch_build_info().decode()


def backend_sources() -> dict[str, str]:
    """Which library serves each BLAS and LAPACK routine, and the FFT."""
    from bartorch import _backend
    from bartorch.core.graph import _ensure_ready

    _ensure_ready()
    return _backend.sources()


__all__ = [
    "BartError",
    "cuda",
    "finufft",
    "__version__",
    "backend_sources",
    "bart_version",
    "build_info",
    "dispatch",
    "get_debug_level",
    "run_command",
    "coil_batch",
    "fold_maps",
    "kernels_to_maps",
    "maps_to_kernels",
    "set_coil_batch",
    "set_fold_maps",
    "set_copy_inputs",
    "set_debug_level",
    "set_num_threads",
]

"""The compiled library, loaded through ctypes.

``libbartorch`` is plain C reached through the ABI in ``csrc/include/bartorch.h``;
nothing here depends on the Python or torch version.  ``BARTORCH_LIBRARY``
overrides the search when the library lives outside the package directory.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

DIMS = 16

ALLOC_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long)
)
FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
LOG_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
)
APPLY_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
RELEASE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _library_names() -> list[str]:
    if sys.platform == "win32":
        return ["bartorch.dll", "libbartorch.dll"]
    if sys.platform == "darwin":
        return ["libbartorch.dylib"]
    return ["libbartorch.so"]


def _package_dirs() -> list[Path]:
    """Every directory the package's files are in.

    A wheel puts the library beside the Python sources.  An editable install
    leaves the sources where they are and adds the directory it built into to
    the package's ``__path__``, so that is where the library is then.
    """
    roots = [Path(__file__).resolve().parent]
    package = sys.modules.get(__package__ or "bartorch")
    for entry in getattr(package, "__path__", ()):
        root = Path(entry).resolve()
        if root not in roots:
            roots.append(root)
    return roots


def _find_library() -> Path:
    override = os.environ.get("BARTORCH_LIBRARY")
    if override:
        return Path(override)
    for root in _package_dirs():
        for name in _library_names():
            candidate = root / name
            if candidate.exists():
                return candidate
    raise ImportError(
        "bartorch: compiled library not found next to the package. "
        "Build it with `pip install .` or point BARTORCH_LIBRARY at libbartorch."
    )


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    lib.bartorch_bart_version.restype = c.c_char_p
    lib.bartorch_bart_version.argtypes = []
    lib.bartorch_build_info.restype = c.c_char_p
    lib.bartorch_build_info.argtypes = []
    lib.bartorch_set_allocator.restype = None
    lib.bartorch_set_allocator.argtypes = [ALLOC_FN, FREE_FN, c.c_void_p]
    lib.bartorch_set_log_handler.restype = None
    lib.bartorch_set_log_handler.argtypes = [LOG_FN, c.c_void_p]
    lib.bartorch_set_debug_level.restype = None
    lib.bartorch_set_debug_level.argtypes = [c.c_int]
    lib.bartorch_get_debug_level.restype = c.c_int
    lib.bartorch_get_debug_level.argtypes = []
    lib.bartorch_set_num_threads.restype = None
    lib.bartorch_set_num_threads.argtypes = [c.c_int]
    lib.bartorch_register.restype = c.c_int
    lib.bartorch_register.argtypes = [c.c_char_p, c.c_int, c.POINTER(c.c_long), c.c_void_p]
    lib.bartorch_exists.restype = c.c_int
    lib.bartorch_exists.argtypes = [c.c_char_p]
    lib.bartorch_lookup.restype = c.c_int
    lib.bartorch_lookup.argtypes = [c.c_char_p, c.c_int, c.POINTER(c.c_long), c.POINTER(c.c_void_p)]
    lib.bartorch_unlink.restype = c.c_int
    lib.bartorch_unlink.argtypes = [c.c_char_p]
    lib.bartorch_unlink_all.restype = c.c_int
    lib.bartorch_unlink_all.argtypes = []
    lib.bartorch_command.restype = c.c_int
    lib.bartorch_command.argtypes = [
        c.c_int,
        c.POINTER(c.c_char_p),
        c.c_char_p,
        c.c_size_t,
        c.c_char_p,
        c.c_size_t,
    ]
    lib.bartorch_backend_set.restype = c.c_int
    lib.bartorch_backend_set.argtypes = [c.c_char_p, c.c_void_p]
    lib.bartorch_backend_count.restype = c.c_int
    lib.bartorch_backend_count.argtypes = []
    lib.bartorch_backend_name.restype = c.c_char_p
    lib.bartorch_backend_name.argtypes = [c.c_int]
    lib.bartorch_backend_has_fallback.restype = c.c_int
    lib.bartorch_backend_has_fallback.argtypes = [c.c_int]

    P = c.c_void_p
    L = c.POINTER(c.c_long)
    lib.bartorch_linop_callback.restype = P
    lib.bartorch_linop_callback.argtypes = [c.c_int, L, c.c_int, L, APPLY_FN, APPLY_FN, P, P, P]
    lib.bartorch_linop_fft.restype = P
    lib.bartorch_linop_fft.argtypes = [c.c_int, L, c.c_ulong, c.c_int, c.c_int]
    lib.bartorch_linop_cdiag.restype = P
    lib.bartorch_linop_cdiag.argtypes = [c.c_int, L, c.c_ulong, P]
    lib.bartorch_linop_fmac.restype = P
    lib.bartorch_linop_fmac.argtypes = [c.c_int, L, L, L, P]
    lib.bartorch_linop_sampling.restype = P
    lib.bartorch_linop_sampling.argtypes = [L, L, P]
    lib.bartorch_linop_nufft.restype = P
    lib.bartorch_linop_nufft.argtypes = [
        c.c_int,
        L,
        L,
        L,
        P,
        L,
        P,
        L,
        P,
        c.c_int,
        c.c_float,
        c.c_float,
    ]
    lib.bartorch_linop_chain.restype = P
    lib.bartorch_linop_chain.argtypes = [P, P]
    lib.bartorch_linop_plus.restype = P
    lib.bartorch_linop_plus.argtypes = [P, P]
    for name in ("domain", "codomain"):
        for kind in ("linop", "nlop"):
            fn = getattr(lib, f"bartorch_{kind}_{name}")
            fn.restype = c.c_int
            fn.argtypes = [P, c.c_int, L]
    for name in ("forward", "adjoint", "normal"):
        fn = getattr(lib, f"bartorch_linop_{name}")
        fn.restype = c.c_int
        fn.argtypes = [P, P, P]
    for name in ("apply", "derivative", "adjoint"):
        fn = getattr(lib, f"bartorch_nlop_{name}")
        fn.restype = c.c_int
        fn.argtypes = [P, P, P]
    lib.bartorch_linop_free.restype = None
    lib.bartorch_linop_free.argtypes = [P]
    lib.bartorch_nlop_free.restype = None
    lib.bartorch_nlop_free.argtypes = [P]
    lib.bartorch_lsqr.restype = c.c_int
    lib.bartorch_lsqr.argtypes = [P, c.c_int, c.c_float, c.c_float, c.c_int, P, P]
    lib.bartorch_nlop_callback.restype = P
    lib.bartorch_nlop_callback.argtypes = [
        c.c_int,
        L,
        c.c_int,
        L,
        APPLY_FN,
        APPLY_FN,
        APPLY_FN,
        P,
        P,
    ]
    lib.bartorch_nlop_from_linop.restype = P
    lib.bartorch_nlop_from_linop.argtypes = [P]
    lib.bartorch_nlop_chain.restype = P
    lib.bartorch_nlop_chain.argtypes = [P, P]
    lib.bartorch_irgnm.restype = c.c_int
    lib.bartorch_irgnm.argtypes = [
        P,
        c.c_int,
        c.c_float,
        c.c_float,
        c.c_float,
        c.c_int,
        c.c_float,
        P,
        P,
        P,
    ]

    for name in ("built", "device_count", "device", "get_streams"):
        fn = getattr(lib, f"bartorch_cuda_{name}")
        fn.restype = c.c_int
        fn.argtypes = []
    for name in ("enable", "set_streams", "use_memcache"):
        fn = getattr(lib, f"bartorch_cuda_{name}")
        fn.restype = c.c_int
        fn.argtypes = [c.c_int]
    for name in ("wait_for_stream", "signal_stream"):
        fn = getattr(lib, f"bartorch_cuda_{name}")
        fn.restype = c.c_int
        fn.argtypes = [P]
    lib.bartorch_cuda_free_memory.restype = c.c_long
    lib.bartorch_cuda_free_memory.argtypes = []

    lib.bartorch_finufft_set.restype = c.c_int
    lib.bartorch_finufft_set.argtypes = [c.c_char_p, P]
    lib.bartorch_finufft_layout.restype = c.c_int
    lib.bartorch_finufft_layout.argtypes = [c.c_int, c.c_int, c.c_int, c.c_int, c.c_int]
    lib.bartorch_finufft_usable_on.restype = c.c_int
    lib.bartorch_finufft_usable_on.argtypes = [c.c_int]
    lib.bartorch_finufft_set_tolerance.restype = None
    lib.bartorch_finufft_set_tolerance.argtypes = [c.c_double]
    lib.bartorch_finufft_tolerance.restype = c.c_double
    lib.bartorch_finufft_tolerance.argtypes = []
    lib.bartorch_finufft_set_upsampling.restype = None
    lib.bartorch_finufft_set_upsampling.argtypes = [c.c_double]
    lib.bartorch_finufft_upsampling.restype = c.c_double
    lib.bartorch_finufft_upsampling.argtypes = []
    lib.bartorch_sense_set_coil_batch.restype = None
    lib.bartorch_sense_set_coil_batch.argtypes = [c.c_int]
    lib.bartorch_sense_coil_batch.restype = c.c_int
    lib.bartorch_sense_coil_batch.argtypes = []
    lib.bartorch_sense_counter.restype = c.c_long
    lib.bartorch_sense_counter.argtypes = [c.c_int]
    lib.bartorch_sense_reset_counters.restype = None
    lib.bartorch_sense_reset_counters.argtypes = []
    lib.bartorch_fft_set.restype = c.c_int
    lib.bartorch_fft_set.argtypes = [c.c_char_p, c.c_void_p]
    lib.bartorch_fft_usable.restype = c.c_int
    lib.bartorch_fft_usable.argtypes = []
    lib.bartorch_fft_counter.restype = c.c_long
    lib.bartorch_fft_counter.argtypes = [c.c_int]
    lib.bartorch_fft_reset_counters.restype = None
    lib.bartorch_fft_reset_counters.argtypes = []
    lib.bartorch_finufft_set_threads.restype = None
    lib.bartorch_finufft_set_threads.argtypes = [c.c_int]
    lib.bartorch_finufft_threads.restype = c.c_int
    lib.bartorch_finufft_threads.argtypes = []
    lib.bartorch_finufft_use_in_tools.restype = None
    lib.bartorch_finufft_use_in_tools.argtypes = [c.c_int]
    lib.bartorch_finufft_usable.restype = c.c_int
    lib.bartorch_finufft_usable.argtypes = []
    lib.bartorch_finufft_live_plans.restype = c.c_long
    lib.bartorch_finufft_live_plans.argtypes = []
    lib.bartorch_last_error.restype = c.c_char_p
    lib.bartorch_last_error.argtypes = []
    lib.bartorch_clear_error.restype = None
    lib.bartorch_clear_error.argtypes = []
    lib.bartorch_nufft_decline_reason.restype = c.c_int
    lib.bartorch_nufft_decline_reason.argtypes = []
    lib.bartorch_nufft_decline_text.restype = c.c_char_p
    lib.bartorch_nufft_decline_text.argtypes = []
    lib.bartorch_nufft_allow_fallback.restype = None
    lib.bartorch_nufft_allow_fallback.argtypes = [c.c_int]
    lib.bartorch_nufft_fallback_allowed.restype = c.c_int
    lib.bartorch_nufft_fallback_allowed.argtypes = []
    lib.bartorch_nufft_counter.restype = c.c_long
    lib.bartorch_nufft_counter.argtypes = [c.c_int]
    lib.bartorch_nufft_reset_counters.restype = None
    lib.bartorch_nufft_reset_counters.argtypes = []
    lib.bartorch_toeplitz_counter.restype = c.c_long
    lib.bartorch_toeplitz_counter.argtypes = [c.c_int]
    lib.bartorch_toeplitz_reset_counters.restype = None
    lib.bartorch_toeplitz_reset_counters.argtypes = []
    lib.bartorch_on_device.restype = c.c_int
    lib.bartorch_on_device.argtypes = [P]
    return lib


_lib: ctypes.CDLL | None = None
_path: Path | None = None


def library() -> ctypes.CDLL:
    """Return the loaded library, loading it on first use."""
    global _lib, _path
    if _lib is None:
        _path = _find_library()
        _lib = _bind(ctypes.CDLL(str(_path)))
    return _lib


def library_path() -> Path:
    library()
    assert _path is not None
    return _path

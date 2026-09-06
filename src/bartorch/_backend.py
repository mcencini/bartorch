"""Where BART's BLAS and LAPACK come from at runtime.

The compiled library carries reference BLAS and no LAPACK.  At import the
process is searched for Fortran-ABI routines that are already loaded: the
MKL, OpenBLAS or Accelerate that torch links.  Whatever is found is installed
into the library's backend table; every LAPACK routine still missing is served
by the NumPy callbacks in :mod:`bartorch._linalg`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from bartorch import _linalg
from bartorch._lib import library

_keepalive: list[object] = []
_sources: dict[str, str] = {}


def _torch_libraries() -> list[Path]:
    try:
        import torch
    except ImportError:
        return []
    libdir = Path(torch.__file__).resolve().parent / "lib"
    if sys.platform == "win32":
        names = ["torch_cpu.dll"]
    elif sys.platform == "darwin":
        names = ["libtorch_cpu.dylib"]
    else:
        names = ["libtorch_cpu.so"]
    return [libdir / n for n in names if (libdir / n).exists()]


def _candidate_handles() -> list[tuple[str, ctypes.CDLL]]:
    handles: list[tuple[str, ctypes.CDLL]] = []
    extra = os.environ.get("BARTORCH_BLAS_LIBRARY")
    if extra:
        handles.append((extra, ctypes.CDLL(extra)))
    for path in _torch_libraries():
        try:
            handles.append((str(path), ctypes.CDLL(str(path))))
        except OSError:
            pass
    if sys.platform == "darwin":
        accelerate = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        try:
            handles.append(("Accelerate", ctypes.CDLL(accelerate)))
        except OSError:
            pass
    if sys.platform != "win32":
        handles.append(("process", ctypes.CDLL(None)))
    return handles


def _lookup(handle: ctypes.CDLL, name: str) -> int | None:
    try:
        fn = getattr(handle, name)
    except AttributeError:
        return None
    return ctypes.cast(fn, ctypes.c_void_p).value


def install() -> dict[str, str]:
    """Fill the backend table; return the source chosen for each routine."""
    lib = library()
    names = [lib.bartorch_backend_name(i).decode() for i in range(lib.bartorch_backend_count())]
    fallback = {n: bool(lib.bartorch_backend_has_fallback(i)) for i, n in enumerate(names)}
    handles = _candidate_handles()
    _keepalive.extend(h for _, h in handles)
    numpy_cbs = _linalg.callbacks()
    chosen: dict[str, str] = {}
    for name in names:
        addr = None
        for source, handle in handles:
            addr = _lookup(handle, name)
            if addr is not None:
                chosen[name] = source
                break
        if addr is None and name in numpy_cbs:
            cb = numpy_cbs[name]
            _keepalive.append(cb)
            addr = ctypes.cast(cb, ctypes.c_void_p).value
            chosen[name] = "numpy"
        if addr is None:
            chosen[name] = "reference" if fallback[name] else "missing"
            continue
        lib.bartorch_backend_set(name.encode(), addr)
    _sources.clear()
    _sources.update(chosen)
    return dict(chosen)


def sources() -> dict[str, str]:
    """The source serving each BLAS and LAPACK routine, after install()."""
    return dict(_sources)

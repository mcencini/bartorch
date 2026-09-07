"""Where BART's BLAS and LAPACK come from at runtime.

The compiled library carries reference BLAS and no LAPACK; both are filled in
at import from libraries already present in the process.  Every entry is a
compiled Fortran-ABI routine, never a Python callback:

* the BLAS and LAPACK torch itself links, which is MKL on Linux and Windows
  and Accelerate on macOS;
* Accelerate directly, on macOS;
* SciPy's ``cython_blas`` and ``cython_lapack``, which publish the whole of
  BLAS and LAPACK as function pointers into their compiled OpenBLAS.

``BARTORCH_BLAS_LIBRARY`` points the search at a specific shared library
first, for a caller who wants a particular MKL, OpenBLAS or vendor build.
:func:`sources` reports what each routine resolved to.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from bartorch._lib import library

_keepalive: list[object] = []
_sources: dict[str, str] = {}

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
_PyCapsule_GetName.restype = ctypes.c_char_p
_PyCapsule_GetName.argtypes = [ctypes.py_object]


class _Provider:
    """One source of Fortran-ABI routines, looked up by symbol name."""

    def __init__(self, name: str):
        self.name = name

    def lookup(self, symbol: str) -> int | None:
        raise NotImplementedError


class _SharedLibrary(_Provider):
    """A loaded shared library, queried with dlsym."""

    def __init__(self, name: str, handle: ctypes.CDLL):
        super().__init__(name)
        self._handle = handle

    def lookup(self, symbol: str) -> int | None:
        try:
            fn = getattr(self._handle, symbol)
        except AttributeError:
            return None
        return ctypes.cast(fn, ctypes.c_void_p).value


class _CythonCapsules(_Provider):
    """SciPy's ``cython_blas`` / ``cython_lapack`` function-pointer tables.

    The capsules hold the addresses of the compiled routines themselves, so a
    call reaches OpenBLAS directly; the names carry no trailing underscore.
    """

    def __init__(self, name: str, table: dict):
        super().__init__(name)
        self._table = table

    def lookup(self, symbol: str) -> int | None:
        capsule = self._table.get(symbol.rstrip("_"))
        if capsule is None:
            return None
        return _PyCapsule_GetPointer(capsule, _PyCapsule_GetName(capsule))


def _torch_library() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    libdir = Path(torch.__file__).resolve().parent / "lib"
    if sys.platform == "win32":
        names = ["torch_cpu.dll"]
    elif sys.platform == "darwin":
        names = ["libtorch_cpu.dylib"]
    else:
        names = ["libtorch_cpu.so"]
    for name in names:
        if (libdir / name).exists():
            return libdir / name
    return None


def _open(name: str, path: str) -> _Provider | None:
    try:
        return _SharedLibrary(name, ctypes.CDLL(path))
    except OSError:
        return None


def _providers() -> list[_Provider]:
    """Sources of compiled routines, most preferred first."""
    found: list[_Provider | None] = []

    override = os.environ.get("BARTORCH_BLAS_LIBRARY")
    if override:
        found.append(_open(override, override))

    torch_lib = _torch_library()
    if torch_lib is not None:
        found.append(_open(torch_lib.name, str(torch_lib)))

    if sys.platform == "darwin":
        found.append(
            _open("Accelerate", "/System/Library/Frameworks/Accelerate.framework/Accelerate")
        )

    if sys.platform != "win32":
        found.append(_open("process", None))

    for module, label in (
        ("scipy.linalg.cython_blas", "scipy"),
        ("scipy.linalg.cython_lapack", "scipy"),
    ):
        try:
            table = __import__(module, fromlist=["__pyx_capi__"]).__pyx_capi__
        except (ImportError, AttributeError):
            continue
        found.append(_CythonCapsules(label, table))

    return [p for p in found if p is not None]


def install() -> dict[str, str]:
    """Fill the library's routine table; return the source chosen for each."""
    lib = library()
    names = [lib.bartorch_backend_name(i).decode() for i in range(lib.bartorch_backend_count())]
    fallback = {n: bool(lib.bartorch_backend_has_fallback(i)) for i, n in enumerate(names)}

    providers = _providers()
    _keepalive.extend(providers)

    chosen: dict[str, str] = {}
    for name in names:
        for provider in providers:
            address = provider.lookup(name)
            if address is not None:
                lib.bartorch_backend_set(name.encode(), address)
                chosen[name] = provider.name
                break
        else:
            chosen[name] = "reference" if fallback[name] else "missing"

    _sources.clear()
    _sources.update(chosen)
    return dict(chosen)


def sources() -> dict[str, str]:
    """The library serving each BLAS and LAPACK routine, after :func:`install`."""
    return dict(_sources)

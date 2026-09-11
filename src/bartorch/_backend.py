"""Where BART's BLAS and LAPACK come from at runtime.

The compiled library carries reference BLAS and no LAPACK; both are filled in
at import from libraries already present in the process.  Every entry is a
compiled Fortran-ABI routine, never a Python callback.  In order:

* MKL, when the ``mkl`` extra is installed.  It covers every routine BART
  calls and is the fastest of these on the work that leans on LAPACK: an
  ESPIRiT calibration, whose per-voxel eigendecompositions dominate it, takes
  about half as long as it does on the others.  There is no MKL wheel for
  macOS, so this is a Linux path;
* the BLAS and LAPACK torch itself links, which is MKL statically on Linux and
  Accelerate on macOS.  Only the routines torch calls are exported, thirteen
  of the thirty;
* Accelerate directly, on macOS;
* SciPy's ``cython_blas`` and ``cython_lapack``, which publish the whole of
  BLAS and LAPACK as function pointers into their compiled OpenBLAS, and cover
  whatever the others do not.

``BARTORCH_BLAS_LIBRARY`` puts one source first: ``mkl``, ``scipy``, ``torch``,
or the path to a shared library.  :func:`sources` reports what each routine
resolved to.

BART's FFT is filled from the same list.  It is planned through the FFTW guru
interface and executed by MKL's DFTI where one of these sources has it, which
on Linux is torch's own MKL and needs nothing installed; where none does,
which is macOS, it is executed by the transform compiled into the library.
``sources()["fft"]`` says which.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import sysconfig
from pathlib import Path

from bartorch._lib import library

_keepalive: list[object] = []
_sources: dict[str, str] = {}
_fft_symbols: dict[str, int] = {}

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


def _mkl_library() -> Path | None:
    """A full MKL in this environment, if one is installed.

    The ``mkl`` wheel puts its libraries under the install prefix rather than
    beside the package, and that prefix is not always ``sys.prefix``.
    """
    roots = {Path(sysconfig.get_paths()["data"]), Path(sys.prefix), Path(sys.base_prefix)}
    names = (
        "libmkl_rt.dylib",
        "libmkl_rt.so.2",
        "libmkl_rt.so.3",
        "libmkl_rt.so",
    )
    for root in sorted(roots):
        for sub in ("lib", "lib64"):
            for name in names:
                candidate = root / sub / name
                if candidate.exists():
                    return candidate
    return None


def _torch_library() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    libdir = Path(torch.__file__).resolve().parent / "lib"
    names = ["libtorch_cpu.dylib"] if sys.platform == "darwin" else ["libtorch_cpu.so"]
    for name in names:
        if (libdir / name).exists():
            return libdir / name
    return None


def _open(name: str, path: str) -> _Provider | None:
    try:
        return _SharedLibrary(name, ctypes.CDLL(path))
    except OSError:
        return None


def _mkl_providers() -> list[_Provider | None]:
    mkl = _mkl_library()
    return [_open("mkl", str(mkl))] if mkl is not None else []


def _torch_providers() -> list[_Provider | None]:
    found: list[_Provider | None] = []
    torch_lib = _torch_library()
    if torch_lib is not None:
        found.append(_open(torch_lib.name, str(torch_lib)))
    if sys.platform == "darwin":
        found.append(
            _open("Accelerate", "/System/Library/Frameworks/Accelerate.framework/Accelerate")
        )
    found.append(_open("process", None))
    return found


def _scipy_providers() -> list[_Provider | None]:
    found: list[_Provider | None] = []
    for module in ("scipy.linalg.cython_blas", "scipy.linalg.cython_lapack"):
        try:
            table = __import__(module, fromlist=["__pyx_capi__"]).__pyx_capi__
        except (ImportError, AttributeError):
            continue
        found.append(_CythonCapsules("scipy", table))
    return found


_NAMED = {"mkl": _mkl_providers, "torch": _torch_providers, "scipy": _scipy_providers}


def _providers() -> list[_Provider]:
    """Sources of compiled routines, most preferred first."""
    found: list[_Provider | None] = []

    override = os.environ.get("BARTORCH_BLAS_LIBRARY")
    if override and override not in _NAMED:
        found.append(_open(override, override))

    for name in _NAMED if not override else [override] + [n for n in _NAMED if n != override]:
        found.extend(_NAMED[name]())

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

    chosen["fft"] = _install_fft(providers)

    _sources.clear()
    _sources.update(chosen)
    return dict(chosen)


#: The DFTI entry points BART's FFT is executed with.  All or none: a table
#: with a hole in it would plan and then fail to transform.
_DFTI = (
    "DftiCreateDescriptor_s_md",
    "DftiSetValue",
    "DftiCommitDescriptor",
    "DftiComputeForward",
    "DftiComputeBackward",
    "DftiFreeDescriptor",
)


def _install_fft(providers: list[_Provider]) -> str:
    """Hand the library MKL's DFTI, if one of these sources has all of it."""
    lib = library()
    for provider in providers:
        found = {name: provider.lookup(name) for name in _DFTI}
        if any(address is None for address in found.values()):
            continue
        for name, address in found.items():
            if 0 != lib.bartorch_fft_set(name.encode(), address):
                break
        else:
            if lib.bartorch_fft_usable():
                _fft_symbols.clear()
                _fft_symbols.update(found)
                return provider.name
    return "built-in"


@contextlib.contextmanager
def built_in_fft():
    """The compiled-in transform, for as long as the block lasts.

    Not part of the package's surface.  What it is for is holding one
    implementation against the other, which is the only way to test that the
    description handed to DFTI is the one BART asked for: both are asked the
    same question and the answers have to agree.  A plan already built keeps
    its descriptor, and reaches it only while the table is filled, so this
    takes effect on plans made before it as well as after.
    """
    lib = library()
    was = bool(lib.bartorch_fft_usable())

    for name in _DFTI:
        lib.bartorch_fft_set(name.encode(), None)
    try:
        yield
    finally:
        if was:
            for name, address in _fft_symbols.items():
                lib.bartorch_fft_set(name.encode(), address)


def sources() -> dict[str, str]:
    """What serves each routine after :func:`install`.

    One entry per BLAS and LAPACK routine, plus ``fft`` for the transform
    behind BART's FFT.
    """
    return dict(_sources)

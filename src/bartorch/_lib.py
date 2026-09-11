"""Finding and loading ``libbartorch``.

``BARTORCH_LIBRARY`` overrides the search.  The signatures come from
:mod:`bartorch._abi`, which ``scripts/gen_abi.py`` generates from the C header.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from bartorch._abi import (  # noqa: F401  (re-exported: these are the ABI's vocabulary)
    ALLOC_FN,
    APPLY_FN,
    DIMS,
    FREE_FN,
    LOG_FN,
    LOG_LEVELS,
    RELEASE_FN,
    SYMBOLS,
    bind,
)


def _library_names() -> list[str]:
    # No Windows name: BART does not build there, so neither does this, and
    # WSL2 is a Linux install like any other.
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


_lib: ctypes.CDLL | None = None
_path: Path | None = None


def library() -> ctypes.CDLL:
    """Return the loaded library, loading it on first use."""
    global _lib, _path
    if _lib is None:
        _path = _find_library()
        _lib = bind(ctypes.CDLL(str(_path)))
    return _lib


def library_path() -> Path:
    library()
    assert _path is not None
    return _path

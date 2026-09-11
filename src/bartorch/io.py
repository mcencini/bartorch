"""Reading and writing BART CFL file pairs."""

from __future__ import annotations

import numpy as np

__all__ = ["readcfl", "writecfl"]

_BART_DIMS = 16


def readcfl(name: str) -> np.ndarray:
    """Read the CFL pair ``name.hdr`` / ``name.cfl`` into a NumPy array.

    Returns
    -------
    numpy.ndarray
        ``complex64`` with BART's dims as its shape, in Fortran order; trailing
        singleton dims are dropped.
    """
    with open(name + ".hdr") as f:
        lines = [ln.strip() for ln in f if not ln.startswith("#")]
    dims = [int(x) for x in lines[0].split()]

    while dims and dims[-1] == 1:
        dims.pop()

    n = int(np.prod(dims)) if dims else 1
    arr = np.fromfile(name + ".cfl", dtype=np.complex64)

    if arr.size != n:
        raise ValueError(f"CFL data size mismatch: header says {n} elements, file has {arr.size}.")

    return arr.reshape(dims, order="F")


def writecfl(name: str, array: np.ndarray) -> None:
    """Write ``array`` as the CFL pair ``name.hdr`` / ``name.cfl``.

    Its shape becomes BART's dims and its values are written as ``complex64`` in
    Fortran order.
    """
    array = np.asarray(array, dtype=np.complex64)
    dims = list(array.shape)
    padded = dims + [1] * (_BART_DIMS - len(dims))

    with open(name + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")

    array.ravel(order="F").tofile(name + ".cfl")

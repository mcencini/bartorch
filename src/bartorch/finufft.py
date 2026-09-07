"""FINUFFT underneath BART.

BART builds every non-Cartesian transform through ``nufft_create``, and that
is the seam: with FINUFFT in place, ``nufft``, ``pics``, ``nlinv`` and ``moba``
compute their forward and adjoint transforms with FINUFFT's type 2 and type 1
rather than with BART's Kaiser-Bessel gridder and oversampled FFT.  A
trajectory that varies across frames is part of that: the frames join the
point set rather than splitting it, so one plan serves them, and a subspace
basis contracts its coefficients away on the k-space side of the pair.  What
FINUFFT cannot serve -- weights that do not lie along k-space, an image that
varies along a sample axis -- is BART's own operator still, and
:func:`decline_reason` says which.

    >>> import bartorch
    >>> bartorch.finufft.enable()
    True
    >>> image = bartorch.tools.pics(kspace, maps, t=traj)

A transform is served by whichever library the data is on: FINUFFT on the
host, cuFINUFFT on a card, which :func:`used_on_device` reports.

:meth:`bartorch.ops.LinearOperator.finufft` is the same transform as an
operator, for use outside BART's tools.
"""

from bartorch._finufft import (
    available,
    cuda_available,
    decline_reason,
    normals_built,
    operators_built,
    reset_counters,
    tolerance,
    used_on_device,
)
from bartorch._finufft import use_in_tools as enable
from bartorch._finufft import used_in_tools as enabled

__all__ = [
    "available",
    "cuda_available",
    "decline_reason",
    "enable",
    "enabled",
    "normals_built",
    "operators_built",
    "reset_counters",
    "tolerance",
    "used_on_device",
]

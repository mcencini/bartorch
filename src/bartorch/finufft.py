"""FINUFFT underneath BART.

BART builds every non-Cartesian transform through ``nufft_create``, and that
is the seam: with FINUFFT in place, ``nufft``, ``pics``, ``nlinv`` and ``moba``
compute their forward and adjoint transforms with FINUFFT's type 2 and type 1
rather than with BART's Kaiser-Bessel gridder and oversampled FFT.  What
FINUFFT cannot serve -- a subspace basis, weights that do not lie along
k-space, a trajectory that changes across frames -- is BART's own operator
still, and :func:`decline_reason` says which.

    >>> import bartorch
    >>> bartorch.finufft.enable()
    True
    >>> image = bartorch.tools.pics(kspace, maps, t=traj)

:meth:`bartorch.ops.LinearOperator.finufft` is the same transform as an
operator, for use outside BART's tools.
"""

from bartorch._finufft import (
    available,
    cuda_available,
    decline_reason,
    operators_built,
    reset_counters,
    tolerance,
)
from bartorch._finufft import use_in_tools as enable
from bartorch._finufft import used_in_tools as enabled

__all__ = [
    "available",
    "cuda_available",
    "decline_reason",
    "enable",
    "enabled",
    "operators_built",
    "reset_counters",
    "tolerance",
]

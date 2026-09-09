"""The NUFFT underneath BART.

BART builds every non-Cartesian transform through ``nufft_create``, and that
is the seam: ``nufft``, ``pics``, ``nlinv`` and ``moba`` compute their forward
and adjoint transforms with FINUFFT's type 2 and type 1 rather than with
BART's Kaiser-Bessel gridder and oversampled FFT, and so does
:meth:`bartorch.ops.LinearOperator.nufft`.  A trajectory that varies across
frames is part of that -- the frames join the point set rather than splitting
it, so one plan serves them -- and a subspace basis contracts its coefficients
away on the k-space side of the pair.

None of that has to be asked for.  The substitution puts itself in place the
first time anything needs it, and what it cannot serve is an error naming the
reason rather than a quieter answer from BART: an order further from the
transform and several times slower, with nothing to say so.
:func:`decline_reason` is that reason, and ``operators_built()`` and
``normals_built()`` count what was built.

A transform is served by whichever library the data is on: FINUFFT on the
host, cuFINUFFT on a card, which :func:`used_on_device` reports.
:func:`configure` sets what they are planned with, and
:func:`bartorch.set_num_threads` how many threads a transform on the host
takes.
"""

from bartorch._finufft import (
    available,
    cuda_available,
    decline_reason,
    normals_built,
    operators_built,
    reset_counters,
    set_threads,
    stream_psf,
    streaming_psf,
    threads,
    tolerance,
    upsampling,
    used_on_device,
)
from bartorch._finufft import use_in_tools as configure
from bartorch._finufft import used_in_tools as enabled

__all__ = [
    "available",
    "configure",
    "cuda_available",
    "decline_reason",
    "enabled",
    "normals_built",
    "operators_built",
    "reset_counters",
    "set_threads",
    "stream_psf",
    "streaming_psf",
    "threads",
    "tolerance",
    "upsampling",
    "used_on_device",
]

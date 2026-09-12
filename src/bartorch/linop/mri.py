"""MRI encoding operators.

Each is a composition of operators BART already has, chained by
``linop_chain`` -- which means what a solver drives is one BART operator, not
a Python object walked per iteration.  That is how BART itself builds them:
``src/wave.c`` chains the coil multiply, the resize, two Fourier transforms,
the point-spread diagonal and the sampling mask into one ``linop_s``, and
:class:`Wave` below is the same six in the same order.

:class:`~bartorch.linop.Sense` is not one of these.  It is BART's own
operator, with the coil batching and the Toeplitz normal that make it what it
is, and what is here composes with it rather than replacing it.
"""

from __future__ import annotations

import torch

from bartorch._operator import Shape, as_operand
from bartorch.linop.base import LinearOperator
from bartorch.linop.basic import FFT, Diagonal, MultiplySum, Sampling
from bartorch.linop.sense import Sense
from bartorch.linop.shape import Resize

__all__ = ["CartesianSense", "Wave"]


def _spatial(image_shape: Shape) -> tuple[int, tuple[int, ...]]:
    """``(coils, spatial)`` with BART's third spatial axis written out."""
    image_shape = tuple(image_shape)
    if len(image_shape) < 3:
        raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")
    coils, spatial = image_shape[0], image_shape[1:]
    return coils, ((1, *spatial) if 2 == len(spatial) else spatial)


def CartesianSense(  # noqa: N802  (it is a constructor)
    sensitivities: torch.Tensor,
    image_shape: Shape,
    pattern: torch.Tensor | None = None,
    **kwargs,
) -> LinearOperator:
    """Coils, a Fourier transform, and the samples that were taken.

    What ``pics`` encodes on a grid: sensitivities, the centred unitary
    transform, and a pattern that keeps the samples the sequence acquired.

    The transform is :class:`~bartorch.linop.Sense` without a trajectory --
    BART's own operator, coil batching and all -- with
    :class:`~bartorch.linop.Sampling` chained onto it.  Without a pattern it
    *is* that operator, returned unchanged, because there is nothing to add.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities, ``(coils, *spatial)``.
    image_shape : tuple of int
        Coil-image shape, ``(coils, *spatial)``.  A two-dimensional problem
        may leave BART's third spatial axis out.
    pattern : tensor, optional
        Ones where a sample was taken and zeros where it was not, broadcast
        over the axes it has one of -- so ``(1, 1, y, 1)`` undersamples a
        phase encode across every coil and slice.
    **kwargs
        Passed to :class:`~bartorch.linop.Sense`: ``coil_batch``, ``kernels``,
        ``device`` and the rest.

    Notes
    -----
    The normal operator of the composition is the two applications rather than
    a point-spread convolution: a mask does not commute with the transform, so
    there is no Toeplitz shortcut to take.  On a grid there is nothing to gain
    from one anyway.

    Examples
    --------
    >>> A = CartesianSense(maps, (coils, y, x), pattern=mask)
    >>> x = bartorch.optim.CG(maxiter=30)(kspace, A)
    """
    if kwargs.get("traj") is not None:
        raise ValueError("a trajectory makes this non-Cartesian; use Sense for that")

    encoding = Sense(sensitivities, image_shape, **kwargs)
    if pattern is None:
        return encoding

    mask = as_operand(pattern, tuple(pattern.shape), "pattern")
    return Sampling(mask, encoding.oshape) @ encoding


def Wave(  # noqa: N802  (it is a constructor)
    sensitivities: torch.Tensor,
    psf: torch.Tensor,
    image_shape: Shape,
    readout: int,
    pattern: torch.Tensor | None = None,
    centred: bool = False,
) -> LinearOperator:
    """Wave-CAIPI encoding, as BART's ``wave`` builds it.

    The gradients that run during the readout spread each voxel along it, and
    the spreading is a multiplication by a point-spread function between the
    readout transform and the phase-encode ones.  So the encoding is six
    operators in a row, which is what ``src/wave.c`` chains:

    ``Sampling . FFT(phase) . Diagonal(psf) . FFT(readout) . Resize .
    MultiplySum(maps)``

    and every one of them is BART's, so the result is a single BART operator
    with an adjoint and a normal of its own.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities, ``(coils, *spatial)``.
    psf : tensor
        The wave point-spread function on the oversampled grid, broadcast over
        the axes it has one of.  :func:`bartorch.tools.wavepsf` makes one from
        the gradient waveform.
    image_shape : tuple of int
        Coil-image shape, ``(coils, *spatial)``, before the readout is
        oversampled.
    readout : int
        Length of the oversampled readout, ``wx`` in BART's sources.  At least
        the readout the image has.
    pattern : tensor, optional
        Ones where a sample was taken, on the oversampled grid.
    centred : bool
        Centre the two transforms.  BART's ``wave`` leaves them uncentred and
        this follows it; ``wshfl`` centres them for its calibration path.

    Examples
    --------
    >>> psf = bartorch.tools.wavepsf(...)
    >>> A = Wave(maps, psf, (coils, y, x), readout=2 * x, pattern=mask)
    """
    coils, spatial = _spatial(image_shape)
    if readout < spatial[-1]:
        raise ValueError(
            f"an oversampled readout of {readout} is shorter than the image's {spatial[-1]}"
        )

    maps = as_operand(sensitivities, tuple(sensitivities.shape), "sensitivities")
    if maps.shape[0] != coils:
        raise ValueError(f"{maps.shape[0]} sensitivities for {coils} coils")
    maps = maps.reshape(coils, *spatial)

    coil_shape = (coils, *spatial)
    over_shape = (coils, *spatial[:-1], readout)

    # The order is wave.c's: E, R, Fx, W, Fyz, M, applied left to right.
    out: LinearOperator = MultiplySum(maps, spatial, coil_shape)
    out = Resize(over_shape, coil_shape) @ out
    out = FFT(over_shape, axes=-1, centred=centred) @ out
    out = Diagonal(as_operand(psf, tuple(psf.shape), "psf"), over_shape) @ out
    out = FFT(over_shape, axes=(-2, -3), centred=centred) @ out

    if pattern is not None:
        out = Sampling(as_operand(pattern, tuple(pattern.shape), "pattern"), over_shape) @ out
    return out

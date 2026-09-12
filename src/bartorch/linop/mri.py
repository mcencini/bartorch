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

__all__ = ["CartesianSense", "FieldCorrected", "Wave"]


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


def _broadcastable(values: torch.Tensor, shape: tuple[int, ...], what: str) -> torch.Tensor:
    """``values`` given the rank of ``shape``, with ones where it is to broadcast."""
    got = tuple(values.shape)
    if len(got) > len(shape):
        raise ValueError(f"{what} has {len(got)} axes, more than the operator's {len(shape)}")
    widened = (1,) * (len(shape) - len(got)) + got
    for axis, (n, full) in enumerate(zip(widened, shape)):
        if n not in (1, full):
            raise ValueError(f"{what} is {n} along axis {axis}, where the operator is {full}")
    return values.reshape(widened)


def FieldCorrected(  # noqa: N802  (it is a constructor)
    encoding: LinearOperator,
    field_map: torch.Tensor | None = None,
    readout_time: torch.Tensor | None = None,
    *,
    mask: torch.Tensor | None = None,
    segments: int = -1,
    method: str = "svd",
    coefficients: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> LinearOperator:
    """An encoding with off-resonance during the readout, by time segmentation.

    A voxel off resonance by ``f`` accrues a phase ``exp(-i 2 pi f t)`` by the
    time ``t`` of each sample, so the exact operator is a different transform
    per sample and no transform at all in the usual sense.  Time segmentation
    approximates it as a short sum of ordinary encodings, each with a spatial
    weight before it and a sample weight after:

    ``A = sum_l diag(b_l) . E . diag(c_l)``

    which is ``linop_plus`` over ``linop_chain``: one BART operator, whatever
    ``E`` is.  So this wraps any encoding -- :func:`CartesianSense`,
    :func:`Wave`, or the non-Cartesian :class:`~bartorch.linop.Sense` -- and
    the last of those is what mirtorch calls ``Gmri``.

    The coefficients are ``mri-nufft``'s: the fit is a least-squares problem
    over a histogram of the field map, not something BART has a primitive for
    and not something to write twice.

    Parameters
    ----------
    encoding : LinearOperator
        The encoding without off-resonance.
    field_map : tensor
        Off-resonance in Hz, broadcastable to the encoding's domain.
    readout_time : tensor
        When each sample is taken, in seconds, broadcastable to the encoding's
        codomain.  Reciprocal units to ``field_map``.
    mask : tensor, optional
        Where the field map is meaningful; everywhere by default.  The fit
        weights the histogram by it, so a mask that excludes air spends the
        segments on tissue.  A map with a single value under the mask is
        refused: there is nothing to segment, and the correction is one phase.
    segments : int
        How many terms the sum has.  ``-1`` lets ``mri-nufft`` choose from the
        spread of the field map and the readout length.
    method : str
        ``"svd"``, ``"mti"`` or ``"mfi"``, ``mri-nufft``'s three factorizations.
    coefficients : tuple of tensor, optional
        ``(b, c)`` already computed, of shapes ``(L, *codomain)`` and
        ``(L, *domain)`` up to broadcasting.  Given these, nothing is fitted
        and ``field_map`` is not needed.

    Notes
    -----
    The normal operator is the sum applied twice rather than anything cheaper:
    a segmented encoding has no Toeplitz form, because the spatial weights do
    not commute with the transform.  More segments is a better approximation
    and proportionally more work.

    Examples
    --------
    >>> E = CartesianSense(maps, (coils, y, x), pattern=mask)
    >>> A = FieldCorrected(E, b0_hz, readout_time=times, segments=6)
    """
    if coefficients is None:
        if field_map is None or readout_time is None:
            raise ValueError("give a field map and readout times, or coefficients")
        b, c = _fit_coefficients(encoding, field_map, readout_time, mask, segments, method)
    else:
        b, c = (as_operand(t, tuple(t.shape), "coefficients") for t in coefficients)
        if b.shape[0] != c.shape[0]:
            raise ValueError(f"{b.shape[0]} sample weights against {c.shape[0]} spatial ones")

    terms = [
        Diagonal(_broadcastable(b[ell], encoding.oshape, "sample weights"), encoding.oshape)
        @ encoding
        @ Diagonal(_broadcastable(c[ell], encoding.ishape, "spatial weights"), encoding.ishape)
        for ell in range(b.shape[0])
    ]

    out = terms[0]
    for term in terms[1:]:
        out = out + term
    return out


def _fit_coefficients(encoding, field_map, readout_time, mask, segments, method):
    """``mri-nufft``'s factorization, in the shapes this operator needs.

    Everything stays at the rank the operator works at.  Squeezing the axes
    the field map or the readout times have only one of would line them up
    against the wrong axes of the operator on the way back, which is a wrong
    answer rather than an error.
    """
    import numpy as np
    from mrinufft.extras import get_orc_factorization

    voxels = _broadcastable(torch.as_tensor(field_map), encoding.ishape, "field_map")
    samples = _broadcastable(torch.as_tensor(readout_time), encoding.oshape, "readout_time")

    flat_map = voxels.reshape(-1).cpu().numpy().astype(np.float32)
    flat_times = samples.reshape(-1).cpu().numpy().astype(np.float32)
    support = (
        np.ones(flat_map.shape, dtype=bool)
        if mask is None
        else torch.as_tensor(mask).reshape(-1).cpu().numpy().astype(bool)
    )

    # The fit bins the field map and solves over the bin centres, so a map
    # with one value in it has one bin and nothing to interpolate between:
    # mri-nufft fails inside its own reshape.  Say what happened instead.
    # np.ptp rather than the array method, which numpy 2 removed.
    if flat_map[support].size and 0.0 == float(np.ptp(flat_map[support])):
        raise ValueError(
            "a field map with one value has nothing to segment: the off-resonance is a "
            "single phase, so demodulate the samples or chain one Diagonal instead"
        )

    b, c, _ = get_orc_factorization(method)(flat_map, flat_times, support, L=segments)

    # b arrives as (samples, L) and c as (L, voxels); both take the leading
    # segment axis and the shape they were fitted over.
    b = torch.as_tensor(np.ascontiguousarray(np.asarray(b).T)).reshape(-1, *samples.shape)
    c = torch.as_tensor(np.asarray(c)).reshape(-1, *voxels.shape)
    return b.to(torch.complex64), c.to(torch.complex64)

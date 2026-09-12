"""MRI encoding operators.

Each is a composition of operators BART already has, chained by
``linop_chain`` -- which means what a solver drives is one BART operator, not
a Python object walked per iteration.  That is how BART itself builds them:
``src/wave.c`` chains the coil multiply, the resize, two Fourier transforms,
the point-spread diagonal and the sampling mask into one ``linop_s``, and
:class:`WaveSense` below is the same six in the same order.

:class:`~bartorch.linop.NoncartesianSense` is not one of these.  It is BART's
own operator, with the coil batching and the Toeplitz normal that make it what
it is, and what is here composes with it rather than replacing it.
"""

from __future__ import annotations

import torch

from bartorch._operator import Shape, as_operand
from bartorch.linop.base import LinearOperator, _WithNormal
from bartorch.linop.basic import FFT, Diagonal, MultiplySum, Sampling
from bartorch.linop.sense import Coils, NoncartesianSense
from bartorch.linop.shape import Reshape, Resize

__all__ = ["CartesianSense", "FieldCorrected", "WaveSense"]


def _spatial(image_shape: Shape) -> tuple[int, tuple[int, ...]]:
    """``(coils, spatial)`` with BART's third spatial axis written out."""
    image_shape = tuple(image_shape)
    if len(image_shape) < 3:
        raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")
    coils, spatial = image_shape[0], image_shape[1:]
    return coils, ((1, *spatial) if 2 == len(spatial) else spatial)


class _GridSense(NoncartesianSense):
    """:class:`~bartorch.linop.NoncartesianSense` over BART's own FFT.

    The same operator, the same coil loop, the same sensitivities held either
    way -- with the transform each slab carries being the centred unitary FFT
    rather than a NUFFT.  Private because what a caller wants is the name for
    the encoding, which is :func:`CartesianSense`.
    """

    _needs_traj = False

    def __init__(self, *args, coeffs: int = 1, **kwargs):
        # Read by NoncartesianSense while it works the shapes out.  On a grid
        # the subspace contraction is chained on afterwards rather than folded
        # into the transform, as ``grecon/model.c`` builds it, so the image
        # carries coefficients that BART is not told about.
        self._coeff_count = int(coeffs)
        super().__init__(*args, **kwargs)


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


def _basis_matrix(basis: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(coeffs, frames)`` read off a basis whose remaining axes are one."""
    b = as_operand(basis, tuple(basis.shape), "basis")
    if b.ndim < 2:
        raise ValueError(f"a basis is (coeffs, frames, 1, ...), not {tuple(b.shape)}")
    if any(n != 1 for n in b.shape[2:]):
        raise ValueError(f"a basis is (coeffs, frames, 1, ...), not {tuple(b.shape)}")
    coeffs, frames = int(b.shape[0]), int(b.shape[1])
    return b.reshape(coeffs, frames), coeffs, frames


def _subspace_kernel(
    b: torch.Tensor, pattern: torch.Tensor | None, sample_shape: tuple[int, ...]
) -> torch.Tensor:
    """The one kernel a sampled subspace normal collapses to.

    With a pattern ``P`` and a basis ``B``, the normal contracts the frames
    away::

        (A^H A x)[k'] = sum_k ( sum_t P[t] conj(B[k',t]) B[k,t] ) x[k]

    so the sum over ``t`` can be done once, at build time, and the frames need
    never be made at all.  On sixty-four echoes over four coefficients that is
    sixteen times less k-space in the middle of every iteration, which is the
    whole reason a subspace reconstruction is affordable.

    It is a Toeplitz normal in the sense the non-Cartesian encoding means it --
    the product in closed form rather than the two applications -- but nothing
    is convolved and no grid is doubled: the pattern already lies on the grid
    the transform is circular over, so the kernel multiplies where the samples
    are.  In the language of the decomposed point spread function, one coset.

    Returned with the coefficients of the domain on BART's COEFF axis and
    those of the codomain on its TE axis, which is the one arrangement a
    single ``fmac`` can contract; the caller reads the answer back onto COEFF
    with a reshape, which moves nothing.
    """
    frames = int(b.shape[1])
    rank = len(sample_shape)

    if pattern is None:
        weights = torch.ones((1, frames, *(1,) * (rank - 2)), dtype=b.dtype, device=b.device)
    else:
        weights = _broadcastable(pattern, sample_shape, "pattern").to(b.dtype).to(b.device)

    outer = b[:, None, :] * b.conj()[None, :, :]
    return torch.tensordot(outer, weights.reshape(frames, *weights.shape[2:]), dims=([2], [0]))


def _subspace(
    encoding: LinearOperator,
    basis: torch.Tensor,
    pattern: torch.Tensor | None,
    *,
    toeplitz: bool,
) -> LinearOperator:
    """``encoding`` read through a temporal subspace, and sampled.

    The forward is what ``grecon/model.c`` chains: the encoding over the
    coefficient images, the basis contracting them into frames, the pattern
    keeping the samples that were taken.  The normal is what ``t2sh`` applies:
    the encoding, one kernel, the encoding back, with the frames never made.
    """
    b, coeffs, frames = _basis_matrix(basis)

    coil_shape = tuple(encoding.oshape)
    if coil_shape[0] != coeffs:
        raise ValueError(
            f"the encoding carries {coil_shape[0]} coefficients and the basis has {coeffs}"
        )

    sample_shape = (1, frames, *coil_shape[2:])
    contraction = b.reshape(coeffs, frames, *(1,) * (len(coil_shape) - 2))

    out = MultiplySum(contraction, coil_shape, sample_shape) @ encoding
    if pattern is not None:
        mask = _broadcastable(pattern, sample_shape, "pattern")
        out = Sampling(mask, sample_shape) @ out

    if not toeplitz:
        return out

    # The kernel's codomain coefficients sit on the TE axis, which is what an
    # ``fmac`` can contract onto; the reshape puts them back on COEFF and is
    # the identity on the buffer.
    mixed_shape = (1, coeffs, *coil_shape[2:])
    kernel = _subspace_kernel(b, pattern, sample_shape)
    normal = (
        encoding.H
        @ Reshape(coil_shape, mixed_shape)
        @ MultiplySum(kernel, coil_shape, mixed_shape)
        @ encoding
    )
    return _WithNormal(out, normal)


def CartesianSense(  # noqa: N802  (it is a constructor)
    sensitivities: torch.Tensor,
    image_shape: Shape,
    pattern: torch.Tensor | None = None,
    *,
    basis: torch.Tensor | None = None,
    toeplitz: bool = True,
    **kwargs,
) -> LinearOperator:
    """Coils, a Fourier transform, and the samples that were taken.

    What ``pics`` encodes on a grid: sensitivities, the centred unitary
    transform, and a pattern that keeps the samples the sequence acquired.

    The transform is :class:`~bartorch.linop.NoncartesianSense` over BART's
    own FFT instead of a NUFFT -- the same operator, coil batching and all --
    with :class:`~bartorch.linop.Sampling` chained onto it.  Without a pattern
    and without a basis it *is* that operator, returned unchanged, because
    there is nothing to add.

    With a basis it is what ``grecon/model.c`` chains: the encoding over the
    coefficient images, the basis contracting them into frames, the pattern
    keeping the samples.  The image is then
    ``(coeffs, 1, 1, 1, *spatial)`` and the samples ``(1, frames, 1, coils,
    *spatial)``.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities, ``(coils, *spatial)``, or their k-space kernels
        with ``kernels=True``.
    image_shape : tuple of int
        Coil-image shape, ``(coils, *spatial)``.  A two-dimensional problem
        may leave BART's third spatial axis out.
    pattern : tensor, optional
        Ones where a sample was taken and zeros where it was not, broadcast
        over the axes it has one of -- so ``(1, 1, y, 1)`` undersamples a
        phase encode across every coil and slice.  With a basis it is read
        against the sample shape, so its second axis is the frames.
    basis : tensor, optional
        Temporal subspace basis ``(coeffs, frames, 1, ...)``, contracting the
        image's coefficients into the frames that were acquired.  This is
        T2 shuffling and what ``pics -B`` takes.
    toeplitz : bool
        With a basis, apply the normal as one coefficient-by-coefficient
        kernel rather than as the two applications.  See the notes.

    Notes
    -----
    Without a basis the normal is the two applications: a mask does not
    commute with the transform, so there is no shortcut, and on a grid there
    is nothing to gain from one anyway.

    With a basis there is a great deal to gain, and it is the same shortcut
    the non-Cartesian encoding calls Toeplitz -- the product in closed form
    rather than the two applications::

        (A^H A x)[k'] = sum_k ( sum_t P[t] conj(B[k',t]) B[k,t] ) x[k]

    The sum over the frames is done once, when the operator is built, so an
    iteration never makes the frames at all: sixty-four echoes over four
    coefficients is sixteen times less k-space in the middle of every step.
    Nothing is convolved and no grid is doubled -- the pattern already lies on
    the grid the transform is circular over -- which is the one way this
    differs from the non-Cartesian normal.  It costs a kernel of
    ``coeffs x coeffs`` over the axes the pattern varies on, so a pattern that
    is flat along the readout keeps it flat too.

    Examples
    --------
    >>> A = CartesianSense(maps, (coils, y, x), pattern=mask)
    >>> x = bartorch.optim.CG(maxiter=30)(kspace, A)

    >>> A = CartesianSense(maps, (coils, y, x), pattern=mask, basis=phi)
    >>> A.ishape, A.oshape
    ((4, 1, 1, 1, 1, y, x), (1, 64, 1, coils, 1, y, x))
    """
    if kwargs.get("traj") is not None:
        raise ValueError("a trajectory makes this non-Cartesian; use NoncartesianSense for that")

    if basis is None:
        encoding = _GridSense(sensitivities, image_shape, **kwargs)
        if pattern is None:
            return encoding

        mask = as_operand(pattern, tuple(pattern.shape), "pattern")
        return Sampling(mask, encoding.oshape) @ encoding

    _, coeffs, _ = _basis_matrix(basis)
    encoding = _GridSense(sensitivities, image_shape, coeffs=coeffs, **kwargs)
    return _subspace(encoding, basis, pattern, toeplitz=toeplitz)


def WaveSense(  # noqa: N802  (it is a constructor)
    sensitivities: torch.Tensor,
    psf: torch.Tensor,
    image_shape: Shape,
    readout: int,
    pattern: torch.Tensor | None = None,
    centred: bool = False,
    *,
    basis: torch.Tensor | None = None,
    toeplitz: bool = True,
    kernels: bool = False,
    coil_batch: int = 1,
    device: torch.device | str | None = None,
) -> LinearOperator:
    """Wave-CAIPI encoding, as BART's ``wave`` builds it.

    The gradients that run during the readout spread each voxel along it, and
    the spreading is a multiplication by a point-spread function between the
    readout transform and the phase-encode ones.  So the encoding is six
    operators in a row, which is what ``src/wave.c`` chains:

    ``Sampling . FFT(phase) . Diagonal(psf) . FFT(readout) . Resize .
    Coils(maps)``

    and every one of them is BART's, so the result is a single BART operator
    with an adjoint and a normal of its own.

    The coils go on through :class:`~bartorch.linop.Coils` rather than a plain
    ``fmac``, which is what lets the sensitivities be held as the k-space
    kernels ``nlinv`` produces and inflated a slab at a time, exactly as
    :class:`~bartorch.linop.NoncartesianSense` holds them.

    With a basis this is Wave-Shuffling: the same encoding over coefficient
    images, the basis contracting them into frames, the pattern keeping the
    samples.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities, ``(coils, *spatial)``, or their k-space kernels
        with ``kernels=True``.
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
    basis : tensor, optional
        Temporal subspace basis ``(coeffs, frames, 1, ...)``.
    toeplitz : bool
        With a basis, apply the normal as one coefficient-by-coefficient
        kernel rather than as the two applications.  The kernel goes where the
        sampling goes -- after the phase-encode transforms, on the oversampled
        grid -- and is the same one :func:`CartesianSense` builds.
    kernels : bool
        Read ``sensitivities`` as k-space kernels.
    coil_batch : int
        Coils applied at once; 0 uses BART's own ``fmac`` over all of them,
        which is what this operator did before it had a choice.
    device : device, optional
        Where the coil multiply is built.

    Examples
    --------
    >>> psf = bartorch.tools.wavepsf(...)
    >>> A = WaveSense(maps, psf, (coils, y, x), readout=2 * x, pattern=mask)
    """
    coils, spatial = _spatial(image_shape)
    if readout < spatial[-1]:
        raise ValueError(
            f"an oversampled readout of {readout} is shorter than the image's {spatial[-1]}"
        )

    coeffs = 1 if basis is None else _basis_matrix(basis)[1]
    coil = Coils(
        sensitivities,
        (coils, *spatial),
        kernels=kernels,
        device=device,
        coil_batch=coil_batch,
        coeffs=coeffs,
    )

    coil_shape = coil.oshape
    over_shape = (*coil_shape[:-1], readout)

    # The order is wave.c's: E, R, Fx, W, Fyz, M, applied left to right.
    out: LinearOperator = Resize(over_shape, coil_shape) @ coil
    out = FFT(over_shape, axes=-1, centred=centred) @ out
    out = Diagonal(_broadcastable(psf, over_shape, "psf"), over_shape) @ out
    out = FFT(over_shape, axes=(-2, -3), centred=centred) @ out

    if basis is not None:
        return _subspace(out, basis, pattern, toeplitz=toeplitz)

    if pattern is not None:
        out = Sampling(_broadcastable(pattern, over_shape, "pattern"), over_shape) @ out
    return out


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
    :func:`WaveSense`, or :class:`~bartorch.linop.NoncartesianSense` -- and
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

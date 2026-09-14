"""MRI encoding operators, each a chain of BART's own.

Built with ``linop_chain``, so what a solver drives is one BART operator
rather than a Python object walked per iteration.
:class:`~bartorch.linop.NoncartesianSense` is not one of these: it is BART's
own operator, and what is here composes with it.
"""

from __future__ import annotations

import torch

from bartorch._dispatch import _lock
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator, _WithNormal
from bartorch.linop.basic import FFT, Diagonal, MultiplySum, Sampling
from bartorch.linop.sense import Coils, NoncartesianSense
from bartorch.linop.shape import Resize

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


class _CartesianNative(_GridSense):
    """:class:`_GridSense` with the pattern and the basis inside the coil loop.

    Each slab's transform carries the pattern and the basis, so they run where
    the transform does and the k-space never crosses back for them.  Its
    normal transforms only the axes the pattern varies along and applies the
    collapsed kernel between them; without ``toeplitz`` it is the two
    applications.  With a basis the image carries its coefficients in front of
    the spatial axes and the samples its frames in front of them.
    """

    def __init__(self, sensitivities, image_shape, pattern, basis, toeplitz=True, **kwargs):
        self._grid_pattern = (
            None if pattern is None else as_operand(pattern, tuple(pattern.shape), "pattern")
        )
        self._grid_basis = None
        if basis is not None:
            matrix, coeffs, frames = _basis_matrix(basis)
            self._grid_basis = matrix.contiguous()
        self._grid_toeplitz = bool(toeplitz)
        super().__init__(sensitivities, image_shape, **kwargs)
        if self._grid_pattern is not None:
            tail = self._kspace_tail()
            got = (1,) * (len(tail) - self._grid_pattern.ndim) + tuple(self._grid_pattern.shape)
            if len(got) != len(tail) or any(g not in (1, f) for g, f in zip(got, tail)):
                raise ValueError(
                    f"a pattern of {tuple(self._grid_pattern.shape)} does not broadcast over one "
                    f"coil's samples {tail}"
                )
            self._grid_pattern = self._grid_pattern.reshape(got)

    def _grid_encoding(self):
        return () if self._grid_basis is None else (int(self._grid_basis.shape[1]),)

    def _image_encoding(self):
        return () if self._grid_basis is None else (int(self._grid_basis.shape[0]),)

    def _has_basis(self):
        return self._grid_basis is not None

    def _create(self) -> Built:
        from bartorch._layout import vector
        from bartorch.linop.sense import _vector, sets_order, wrap_item

        lib = library()
        p, b = self._grid_pattern, self._grid_basis
        pvec = bvec = None
        if p is not None:
            shape = tuple(p.shape)
            spatial = shape[len(shape) - self.ndim :]
            z, y, x = spatial if self.ndim == 3 else (1, *spatial)
            placed = {0: x, 1: y, 2: z}
            if b is not None:
                placed[5] = shape[0]
            pvec = vector(placed)
        if b is not None:
            bvec = vector({5: b.shape[1], 6: b.shape[0]})
        with _lock:
            was = lib.bartorch_sense_coil_batch(), lib.bartorch_sense_fold_maps()
            lib.bartorch_sense_set_coil_batch(self.coil_batch)
            lib.bartorch_sense_set_fold_maps(int(self.fold_maps))
            try:
                item = self._under_lock(
                    lib.bartorch_linop_cartesian,
                    _vector(self._max_vector()),
                    _vector(self._sens_vector()),
                    self.sensitivities.data_ptr(),
                    int(self.kernels),
                    None if p is None else _vector(pvec),
                    None if p is None else p.data_ptr(),
                    None if b is None else _vector(bvec),
                    None if b is None else b.data_ptr(),
                    int(self._grid_toeplitz),
                    device=self.device,
                )
            finally:
                lib.bartorch_sense_set_coil_batch(was[0])
                lib.bartorch_sense_set_fold_maps(was[1])
            ptr = wrap_item(
                self,
                lib,
                item,
                self.kspace_shape,
                self.image_shape,
                self.batches,
                self.device,
                sets_order(self.batches, self.sets, self.image_encoding, self.ndim),
            )
        keep = tuple(t for t in (self.sensitivities, p, b) if t is not None)
        return Built(ptr, self.image_shape, self.kspace_shape, keep=keep, device=self.device)


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


class _Relabel(LinearOperator):
    """``op`` over other shapes holding the same elements in the same order.

    BART's ``operator_reshape``: the dimensions are relabelled, nothing is
    copied, and the operator keeps its own normal.  A contraction BART does
    across two axes runs this way on samples that hold the two on one.
    """

    def __init__(self, op: LinearOperator, oshape: Shape, ishape: Shape):
        self.op = op._bart()
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        super().__init__()

    def _create(self) -> Built:
        device = self.op.device
        ptr = self._under_lock(
            library().bartorch_linop_reshaped,
            self.op._h.ptr,
            DIMS,
            dims(self.oshape),
            dims(self.ishape),
            device=device,
        )
        return Built(ptr, self.ishape, self.oshape, keep=(self.op,), device=device)


def _subspace_kernel(
    b: torch.Tensor, pattern: torch.Tensor | None, sample_shape: tuple[int, ...], axis: int
) -> torch.Tensor:
    """The kernel a sampled subspace normal collapses to::

        K[k, k'] = sum_t |P[t]|^2 conj(B[k', t]) B[k, t]

    over the axes the pattern varies on, with ``k`` on ``axis`` of
    ``sample_shape`` (where its frames are) and ``k'`` on the axis after it.
    A ``MultiplySum`` from samples with the coefficients on ``axis`` and a one
    after them, to samples with the two swapped, is ``B^H P^H P B`` without
    the frames being made.
    """
    coeffs, frames = (int(n) for n in b.shape)
    outer = b.T[:, :, None] * b.conj().T[:, None, :]
    if pattern is None:
        return outer.sum(0).reshape(coeffs, coeffs, *(1,) * (len(sample_shape) - axis - 1))
    w = _broadcastable(pattern, sample_shape, "pattern").abs().square()
    w = w.to(b.dtype).to(b.device).movedim(axis, -1)
    if w.shape[-1] == 1:
        kernel = w[..., None] * outer.sum(0)
    else:
        kernel = torch.tensordot(w, outer, dims=([-1], [0]))
    return kernel.movedim((-2, -1), (axis, axis + 1))


def _wave_subspace(
    encoding: LinearOperator,
    basis: torch.Tensor,
    pattern: torch.Tensor | None,
    toeplitz: bool,
    ndim: int,
) -> LinearOperator:
    """``encoding`` over coefficient images, contracted into frames and sampled.

    ``encoding`` returns ``(..., coeffs, *spatial)``, and the basis turns the
    coefficients into frames on the same axis.  BART contracts across two
    axes, so the contraction runs on the samples relabelled with a one after
    the coefficients and is relabelled back.  With ``toeplitz`` the normal is
    the encoding, the kernel of :func:`_subspace_kernel` between the same two
    relabellings, and the encoding back.
    """
    b, coeffs, frames = _basis_matrix(basis)
    over = tuple(encoding.oshape)
    axis = len(over) - ndim - 1
    split = (*over[: axis + 1], 1, *over[axis + 1 :])
    swapped = (*over[:axis], 1, coeffs, *over[axis + 1 :])
    framed = (*over[:axis], 1, frames, *over[axis + 1 :])
    samples = (*over[:axis], frames, *over[axis + 1 :])

    contraction = b.reshape(coeffs, frames, *(1,) * ndim)
    spread = MultiplySum(contraction, split, framed) @ _Relabel(encoding, split, encoding.ishape)
    out: LinearOperator = _Relabel(spread, samples, encoding.ishape)
    if pattern is not None:
        out = Sampling(_broadcastable(pattern, samples, "pattern"), samples) @ out
    if not toeplitz:
        return out

    kernel = _subspace_kernel(b, pattern, samples, axis)
    middle = _Relabel(MultiplySum(kernel, split, swapped), over, over)
    return _WithNormal(out, encoding.H @ middle @ encoding)


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

    Without a pattern and without a basis this is
    :class:`~bartorch.linop.NoncartesianSense` over BART's own FFT -- the same
    operator, coil batching and all.  With either, the pattern and the basis
    run inside the coil loop, where the transform does.

    With a basis it is T2 shuffling, what ``pics -B`` takes: the image holds
    coefficients and the basis contracts them into the frames that were
    acquired.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities ``([sets,] coils, [z,] y, x)``, or their k-space
        kernels with ``kernels=True``.
    image_shape : tuple of int
        The image, ``(*batches, [sets,] [coeffs,] [z,] y, x)``.  The samples
        are ``(*batches, coils, [frames,] [z,] y, x)``.
    pattern : tensor, optional
        Ones where a sample was taken and zeros where it was not, broadcast
        over one coil's samples ``([frames,] [z,] y, x)`` -- so ``(y, 1)``
        undersamples a phase encode for every coil and batch item.
    basis : tensor, optional
        Temporal subspace basis ``(coeffs, frames)``.
    toeplitz : bool
        Apply the normal in closed form; see the notes.  Without it the
        normal is the two applications.
    **kwargs
        Passed to :class:`~bartorch.linop.NoncartesianSense`: ``coil_batch``,
        ``kernels``, ``modulated``, ``device``, ``ndim`` and the rest.
        ``modulated`` asks for BART's own sample convention -- the one
        ``pics`` works in -- instead of the centred one.

    Notes
    -----
    With a pattern or a basis, ``toeplitz`` gives the normal as::

        (A^H A x)[k'] = sum_k ( sum_t |P[t]|^2 conj(B[k',t]) B[k,t] ) x[k]

    between transforms along only the axes the pattern varies on.  The sum
    over the frames is done once, when the operator is built, so an iteration
    never makes the frames, and a pattern flat along the readout never
    transforms it.  In BART's modulated convention the samples differ from the
    centred ones by a phase of modulus one, which leaves this normal as it is.

    Examples
    --------
    >>> A = CartesianSense(maps, (y, x), pattern=mask)          # mask (y, 1)
    >>> x = bartorch.optim.CG(maxiter=30)(kspace, A)

    >>> A = CartesianSense(maps, (4, y, x), pattern=mask, basis=phi)
    >>> A.ishape, A.oshape                                       # phi (4, 64)
    ((4, y, x), (coils, 64, y, x))
    """
    if kwargs.get("traj") is not None:
        raise ValueError("a trajectory makes this non-Cartesian; use NoncartesianSense for that")

    if basis is None and pattern is None:
        return _GridSense(sensitivities, image_shape, **kwargs)

    if not kwargs.pop("modulated", False):
        return _CartesianNative(
            sensitivities, image_shape, pattern, basis, toeplitz=toeplitz, **kwargs
        )

    if basis is None:
        encoding = _GridSense(sensitivities, image_shape, modulated=True, **kwargs)
        mask = as_operand(pattern, tuple(pattern.shape), "pattern")
        return Sampling(mask, encoding.oshape) @ encoding

    from bartorch.fourier import fftmod

    native = _CartesianNative(
        sensitivities, image_shape, pattern, basis, toeplitz=toeplitz, **kwargs
    )
    axes = tuple(range(-native.ndim, 0))
    phase = fftmod(torch.ones(native.spatial, dtype=torch.complex64), axes, inverse=True)
    phase = _broadcastable(phase, native.oshape, "phase")
    return _WithNormal(Diagonal(phase, native.oshape) @ native, native.gram())


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
    ndim: int | None = None,
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
    ``fmac``, so the sensitivities may be held as the k-space kernels
    ``nlinv`` produces and inflated a slab at a time.

    With a basis this is Wave-Shuffling: the same encoding over coefficient
    images, the basis contracting the coefficients into frames, the pattern
    keeping the samples.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities ``([sets,] coils, [z,] y, x)``, or their k-space
        kernels with ``kernels=True``.
    psf : tensor
        The wave point-spread function on the oversampled grid, broadcast over
        one coil's samples ``([z,] y, readout)``; the same for every frame.
        :func:`bartorch.tools.wavepsf` makes one from the gradient waveform.
    image_shape : tuple of int
        The image, ``(*batches, [sets,] [coeffs,] [z,] y, x)``, before the
        readout is oversampled.  The samples are ``(*batches, coils,
        [frames,] [z,] y, readout)``.
    readout : int
        Length of the oversampled readout, ``wx`` in BART's sources.  At least
        the readout the image has.
    pattern : tensor, optional
        Ones where a sample was taken, broadcast over one coil's samples
        ``([frames,] [z,] y, readout)``.
    centred : bool
        Centre the two transforms.  BART's ``wave`` leaves them uncentred and
        this follows it; ``wshfl`` centres them for its calibration path.
    basis : tensor, optional
        Temporal subspace basis ``(coeffs, frames)``, of two coefficients or
        more.
    toeplitz : bool
        With a basis, apply the normal as one coefficient-by-coefficient
        kernel rather than as the two applications.  The kernel goes where the
        sampling goes -- after the phase-encode transforms, on the oversampled
        grid -- and is the one :func:`CartesianSense` collapses to.
    kernels : bool
        Read ``sensitivities`` as k-space kernels.
    coil_batch : int
        Coils applied at once; 0 uses BART's own ``fmac`` over all of them.
    device : device, optional
        Where the coil multiply is built.
    ndim : int, optional
        Spatial axes, where the sensitivities and the image do not say: a bank
        of four kernel axes is either three behind the coils or two behind
        sets and coils.

    Examples
    --------
    >>> psf = bartorch.tools.wavepsf(x=2 * x, y=y)
    >>> A = WaveSense(maps, psf, (y, x), readout=2 * x, pattern=mask)
    >>> A.oshape
    (coils, y, 2 * x)
    """
    coeffs = 1
    if basis is not None:
        _, coeffs, _ = _basis_matrix(basis)
        if coeffs < 2:
            raise ValueError("a basis for Wave-Shuffling has two coefficients or more")

    coil = Coils(
        sensitivities,
        image_shape,
        kernels=kernels,
        device=device,
        coil_batch=coil_batch,
        coeffs=coeffs,
        ndim=ndim,
    )
    if readout < coil.spatial[-1]:
        raise ValueError(
            f"an oversampled readout of {readout} is shorter than the image's {coil.spatial[-1]}"
        )

    coil_shape = coil.oshape
    over_shape = (*coil_shape[:-1], readout)
    # The phase encodes are the spatial axes in front of the readout.
    phase = (-2, -3) if coil.ndim == 3 else (-2,)

    # The order is wave.c's: E, R, Fx, W, Fyz, M, applied left to right.
    out: LinearOperator = Resize(over_shape, coil_shape) @ coil
    out = FFT(over_shape, axes=-1, centred=centred) @ out
    out = Diagonal(_broadcastable(psf, over_shape, "psf"), over_shape) @ out
    out = FFT(over_shape, axes=phase, centred=centred) @ out

    if basis is not None:
        return _wave_subspace(out, basis, pattern, toeplitz, coil.ndim)
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
    over a histogram of the field map, which BART has no primitive for.

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
    >>> E = CartesianSense(maps, (y, x), pattern=mask)
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

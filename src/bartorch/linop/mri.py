"""MRI encoding operators, each a chain of BART's own.

Built with ``linop_chain``, so what a solver drives is one BART operator
rather than a Python object walked per iteration.
:class:`~bartorch.linop.NoncartesianSense` is BART's own operator with the
coil loop in it; :func:`CartesianSense` and :func:`WaveSense` are that loop
over a Cartesian or a wave transform.
"""

from __future__ import annotations

import math

import torch

from bartorch import _layout
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator, _WithNormal
from bartorch.linop.basic import Diagonal, MultiplySum, Sampling
from bartorch.linop.form import Array, Contraction, Form
from bartorch.linop.sense import NoncartesianSense

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


def _flat_along(t: torch.Tensor, axes) -> torch.Tensor | None:
    """``t`` at the first index of each of ``axes`` it is the same all along, or ``None``."""
    for axis in axes:
        if t.shape[axis] > 1:
            first = t.narrow(axis, 0, 1)
            if not torch.equal(t, first.expand_as(t)):
                return None
            t = first
    return t


def _segment_layout(encoding, b, c, sample_dims, image_dims):
    """The segments as the form's contraction, or ``None``.

    The sample weights may vary along one coil's samples and the spatial ones
    along the image's spatial axes; weights the same along the batches, the
    coils, the sets or the coefficients are taken once.  Weights that vary
    along any of those are left to the sum of the segments.
    """
    count = int(b.shape[0])
    oshape, ishape = tuple(encoding.oshape), tuple(encoding.ishape)
    tail = tuple(encoding._kspace_tail())
    lead = len(oshape) - len(tail)
    image_lead = len(ishape) - encoding.ndim

    samples = _flat_along(_per_segment(b, oshape, "sample weights"), range(1, 1 + lead))
    image = _flat_along(_per_segment(c, ishape, "spatial weights"), range(1, 1 + image_lead))
    if samples is None or image is None:
        return None

    samples = samples.reshape(count, *samples.shape[1 + lead :])
    image = image.reshape(count, *image.shape[1 + image_lead :])
    samples = as_operand(samples, tuple(samples.shape), "sample weights")
    image = as_operand(image, tuple(image.shape), "spatial weights")
    sample_vector = _layout.vector({d: int(n) for d, n in zip(sample_dims, samples.shape[1:])})
    image_vector = _layout.vector({d: int(n) for d, n in zip(image_dims, image.shape[1:])})
    return Contraction(count, Array(samples, sample_vector), Array(image, image_vector))


class _Segmentable:
    """A grid encoding whose coil loop a contraction over terms can go inside.

    The axes each side's weights are laid out on: the image's spatial ones,
    and one coil's samples, which carry the frames as well where a basis
    contracts them.
    """

    def _image_dims(self):
        return (2, 1, 0) if self.ndim == 3 else (1, 0)

    def _sample_dims(self):
        return ((5,) if self._grid_basis is not None else ()) + self._image_dims()


class _CartesianNative(_Segmentable, _GridSense):
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

    def _form(self) -> Form:
        from bartorch._layout import vector

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
        return Form(
            transform="fft",
            max_vector=self._max_vector(),
            kspace_vector=self._kspace_vector(),
            sensitivities=Array(self.sensitivities, self._sens_vector()),
            kernels=self.kernels,
            pattern=None if p is None else Array(p, pvec),
            basis=None if b is None else Array(b, bvec),
            toeplitz=self._grid_toeplitz,
            coil_batch=self.coil_batch,
            fold_maps=self.fold_maps,
            coils=self.coils,
            sets=self.sets,
            coeffs=1 if b is None else int(b.shape[0]),
        )

    def _create(self) -> Built:
        from bartorch.linop.sense import build_form

        return build_form(self, self._form())


def _checked_positions(positions, frames, encodes) -> torch.Tensor:
    """Positions as int64 on the host, checked against the frames a basis has and the phase encodes.

    Checked before the operator is built, which happens as it is made.
    """
    pos = torch.as_tensor(positions)
    if pos.is_floating_point() or pos.is_complex():
        raise ValueError("positions are integer phase-encode indices")
    pos = pos.to(device="cpu", dtype=torch.int64).contiguous()
    if pos.ndim < 2:
        raise ValueError(f"positions are (*encoding, shots, d), not {tuple(pos.shape)}")
    lead = tuple(pos.shape[:-2])
    if frames is None and lead:
        raise ValueError(f"positions over frames {lead} need a basis to contract them")
    if frames is not None and lead != (frames,):
        raise ValueError(f"positions over frames {lead}, and a basis of {frames}")
    if pos.shape[-1] != len(encodes):
        raise ValueError(
            f"positions of {pos.shape[-1]} indices, "
            f"for {len(encodes)} phase-encode axes {tuple(encodes)}"
        )
    padding = (pos == -1).all(-1)
    inside = ((pos >= 0) & (pos < torch.tensor(encodes))).all(-1)
    if not bool((padding | inside).all()):
        raise ValueError(
            f"a position lies outside the phase encodes {tuple(encodes)}, and is not padding"
        )
    return pos


class _CartesianSampled(_GridSense):
    """:class:`_GridSense` over a table of the phase encodes that were sampled.

    Each frame's phase encodes are positions ``(shots, d)`` -- ``(y,)`` in 2D,
    ``(z, y)`` in 3D, ``-1`` for padding -- with the whole readout along each,
    so the samples are ``(*batches, coils, [frames,] shots, readout)`` and
    nothing the size of the phase-encode plane is held for them.  The
    transforms, the basis and the normal are :class:`_CartesianNative`'s, over
    the pattern the positions stand for.
    """

    def __init__(
        self,
        sensitivities,
        image_shape,
        positions,
        basis,
        readout="kspace",
        toeplitz=True,
        **kwargs,
    ):
        from bartorch.linop.sense import _grid_ndim

        if readout not in ("kspace", "image"):
            raise ValueError(f"the readout is 'kspace' or 'image', not {readout!r}")
        self._grid_basis = None
        frames = None
        if basis is not None:
            matrix, _, frames = _basis_matrix(basis)
            self._grid_basis = matrix.contiguous()
        ndim = _grid_ndim(
            sensitivities, image_shape, kwargs.get("kernels", False), kwargs.get("ndim")
        )
        encodes = tuple(image_shape)[len(image_shape) - ndim : -1]
        self._positions = _checked_positions(positions, frames, encodes)
        self._kspace_readout = readout == "kspace"
        self._grid_toeplitz = bool(toeplitz)
        super().__init__(sensitivities, image_shape, **kwargs)

    def _grid_encoding(self):
        return tuple(int(n) for n in self._positions.shape[:-2])

    def _image_encoding(self):
        return () if self._grid_basis is None else (int(self._grid_basis.shape[0]),)

    def _has_basis(self):
        return self._grid_basis is not None

    def _kspace_tail(self):
        return (*self.encoding, int(self._positions.shape[-2]), int(self.spatial[-1]))

    def _kspace_vector(self):
        base = {1: self.spatial[-1], 2: int(self._positions.shape[-2]), _layout.COIL: self.coils}
        return self._encoding_vector(base, self.encoding)

    def _form(self) -> Form:
        b, pos = self._grid_basis, self._positions
        bvec = None if b is None else _layout.vector({5: b.shape[1], 6: b.shape[0]})
        return Form(
            transform="fft",
            max_vector=self._max_vector(),
            kspace_vector=self._kspace_vector(),
            sensitivities=Array(self.sensitivities, self._sens_vector()),
            kernels=self.kernels,
            basis=None if b is None else Array(b, bvec),
            positions=pos,
            frames=1 if b is None else int(b.shape[1]),
            shots=int(pos.shape[-2]),
            components=int(pos.shape[-1]),
            kspace_readout=self._kspace_readout,
            toeplitz=self._grid_toeplitz,
            coil_batch=self.coil_batch,
            fold_maps=self.fold_maps,
            coils=self.coils,
            sets=self.sets,
            coeffs=1 if b is None else int(b.shape[0]),
        )

    def _create(self) -> Built:
        from bartorch.linop.sense import build_form

        return build_form(self, self._form())


class _WaveNative(_Segmentable, _GridSense):
    """:class:`_GridSense` over the wave encoding, with everything after the coils in the coil loop.

    Each slab's transform zero-fills the readout to ``readout`` about its
    centre, transforms along it, multiplies by the point spread function and
    transforms along the phase encodes -- ``src/wave.c``'s chain -- with the
    pattern or the positions, and a basis, after it as :func:`CartesianSense`
    has them.
    """

    def __init__(
        self,
        sensitivities,
        psf,
        image_shape,
        readout,
        pattern,
        positions,
        basis,
        centred,
        toeplitz,
        **kwargs,
    ):
        from bartorch.linop.sense import _grid_ndim

        image_shape = tuple(image_shape)
        self._wave_readout = int(readout)
        if self._wave_readout < image_shape[-1]:
            raise ValueError(
                f"an oversampled readout of {readout} is shorter than the image's {image_shape[-1]}"
            )
        if pattern is not None and positions is not None:
            raise ValueError("give a pattern or positions, not both")
        self._wave_centred = bool(centred)
        self._grid_toeplitz = bool(toeplitz)

        self._grid_basis = None
        frames = None
        if basis is not None:
            matrix, _, frames = _basis_matrix(basis)
            self._grid_basis = matrix.contiguous()

        ndim = _grid_ndim(
            sensitivities, image_shape, kwargs.get("kernels", False), kwargs.get("ndim")
        )
        spatial = image_shape[len(image_shape) - ndim :]
        over = (*spatial[:-1], self._wave_readout)

        # The point spread function is over one coil's samples before any frame.
        w = as_operand(psf, tuple(psf.shape), "psf")
        self._psf = (
            _broadcastable(_one_coil(w, len(over), "psf"), over, "psf").expand(over).contiguous()
        )

        self._positions = None
        if positions is not None:
            self._positions = _checked_positions(positions, frames, spatial[:-1])

        self._grid_pattern = None
        if pattern is not None:
            tail = (*(() if frames is None else (frames,)), *over)
            m = as_operand(pattern, tuple(pattern.shape), "pattern")
            self._grid_pattern = _broadcastable(_one_coil(m, len(tail), "pattern"), tail, "pattern")

        super().__init__(sensitivities, image_shape, **kwargs)

    def _sample_dims(self):
        if self._positions is None:
            return super()._sample_dims()
        return ((5,) if self._grid_basis is not None else ()) + (2, 1)

    def _grid_encoding(self):
        return () if self._grid_basis is None else (int(self._grid_basis.shape[1]),)

    def _image_encoding(self):
        return () if self._grid_basis is None else (int(self._grid_basis.shape[0]),)

    def _has_basis(self):
        return self._grid_basis is not None

    def _kspace_tail(self):
        if self._positions is not None:
            return (*self.encoding, int(self._positions.shape[-2]), self._wave_readout)
        return (*self.encoding, *self.spatial[:-1], self._wave_readout)

    def _kspace_vector(self):
        if self._positions is not None:
            base = {
                1: self._wave_readout,
                2: int(self._positions.shape[-2]),
                _layout.COIL: self.coils,
            }
        else:
            z, y, _ = self._spatial3()
            base = {0: self._wave_readout, 1: y, 2: z, _layout.COIL: self.coils}
        return self._encoding_vector(base, self.encoding)

    def _form(self) -> Form:
        w, p, b, pos = self._psf, self._grid_pattern, self._grid_basis, self._positions
        pvec = bvec = None
        if p is not None:
            shape = tuple(p.shape)
            spatial = shape[len(shape) - self.ndim :]
            z, y, x = spatial if self.ndim == 3 else (1, *spatial)
            placed = {0: x, 1: y, 2: z}
            if b is not None:
                placed[5] = shape[0]
            pvec = _layout.vector(placed)
        if b is not None:
            bvec = _layout.vector({5: b.shape[1], 6: b.shape[0]})
        return Form(
            transform="wave",
            max_vector=self._max_vector(),
            kspace_vector=self._kspace_vector(),
            sensitivities=Array(self.sensitivities, self._sens_vector()),
            kernels=self.kernels,
            pattern=None if p is None else Array(p, pvec),
            basis=None if b is None else Array(b, bvec),
            positions=pos,
            frames=1 if b is None else int(b.shape[1]),
            shots=0 if pos is None else int(pos.shape[-2]),
            components=0 if pos is None else int(pos.shape[-1]),
            readout=self._wave_readout,
            psf=w,
            centred=self._wave_centred,
            toeplitz=self._grid_toeplitz,
            coil_batch=self.coil_batch,
            fold_maps=self.fold_maps,
            coils=self.coils,
            sets=self.sets,
            coeffs=1 if b is None else int(b.shape[0]),
        )

    def _create(self) -> Built:
        from bartorch.linop.sense import build_form

        return build_form(self, self._form())


class _Relabel(LinearOperator):
    """``op`` over other shapes holding the same elements in the same order.

    BART's ``operator_reshape``: the dimensions are relabelled, nothing is
    copied, and the operator keeps its own normal.
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


class _SegmentedSense(NoncartesianSense):
    """:class:`~bartorch.linop.NoncartesianSense` read through a basis over the samples.

    The image carries ``L`` segment images in front of its spatial axes, and a
    sample is ``sum_l b_l(t) E(x_l)`` -- time segmentation, with each
    segment's sample weights as a basis along the shots and the readout.  The
    normal is the Toeplitz one over that basis's packed Gram, so the segments
    cost a kernel per pair of them rather than a transform each.
    """

    def __init__(self, encoding: NoncartesianSense, basis: torch.Tensor):
        self._sample_basis = basis
        segments = int(basis.shape[0])
        super().__init__(
            encoding.sensitivities,
            (*encoding.batches, segments, *encoding.spatial),
            traj=encoding.traj,
            kernels=encoding.kernels,
            toeplitz=encoding.toeplitz,
            weights=encoding.weights,
            device=encoding.device,
            coil_batch=encoding.coil_batch,
            fold_maps=encoding.fold_maps,
        )

    def _image_encoding(self):
        return (int(self._sample_basis.shape[0]),)

    def _max_vector(self):
        z, y, x = self._spatial3()
        return _layout.vector(
            {
                _layout.READ: x,
                _layout.PHS1: y,
                _layout.PHS2: z,
                _layout.COIL: self.coils,
                _layout.MAPS: self.sets,
                _layout.COEFF: int(self._sample_basis.shape[0]),
            }
        )

    def _basis_layout(self):
        b = self._sample_basis
        return b, _layout.vector({1: b.shape[2], 2: b.shape[1], _layout.COEFF: b.shape[0]})


def _per_segment(values: torch.Tensor, shape: tuple[int, ...], what: str) -> torch.Tensor:
    """``values`` of ``(segments, ...)``, what follows the segments broadcast over ``shape``."""
    one = _broadcastable(values[0], shape, what)
    return values.reshape(values.shape[0], *one.shape)


def contracted(encoding, b: torch.Tensor, c: torch.Tensor) -> LinearOperator | None:
    """``sum_l diag(b_l) E diag(c_l)`` as one encoding, or ``None``.

    What the planner reaches for a sum of chains sharing an encoding.  On a
    grid the contraction goes inside the coil loop, around the transform each
    slab carries, and the normal is the two applications: the closed form
    would save no transform there.  Over a NUFFT it becomes a subspace basis
    along the samples, whose Toeplitz normal is a point spread function per
    pair of terms rather than two transforms per term per coil.

    ``None`` says the factors do not fit the form -- weights that vary along
    the batches, the coils, the sets or the coefficients, or an encoding that
    already carries a contraction -- and the sum of chains stands instead.
    """
    if isinstance(encoding, _Segmentable):
        return _grid_contraction(encoding, b, c)
    return _nufft_contraction(encoding, b, c)


def _grid_contraction(encoding, b: torch.Tensor, c: torch.Tensor) -> LinearOperator | None:
    """The contraction in the coil loop of a grid encoding, or ``None``."""
    from bartorch.linop.sense import Encoded

    form = encoding._form()
    if form.contraction is not None:
        return None
    layout = _segment_layout(encoding, b, c, encoding._sample_dims(), encoding._image_dims())
    if layout is None:
        return None
    return Encoded(encoding, form.with_contraction(layout))


def _nufft_contraction(encoding, b: torch.Tensor, c: torch.Tensor) -> LinearOperator | None:
    """The contraction over a NUFFT as one subspace operator, or ``None``.

    The spatial weights fan the image out into one image per term, and the
    terms' sample weights are a basis over the samples.  Taken where the
    encoding is a plain non-Cartesian SENSE operator with no basis, sets or
    encoding axes of its own, and where the sample weights vary along the
    shots and the readout alone.
    """
    if type(encoding) is not NoncartesianSense:
        return None
    if encoding.basis is not None or encoding.sets > 1 or encoding.encoding:
        return None

    segments = int(b.shape[0])
    oshape, ishape = tuple(encoding.oshape), tuple(encoding.ishape)
    weights = _per_segment(b, oshape, "sample weights")
    if any(n != 1 for n in weights.shape[1:-2]):
        return None
    basis = weights.reshape(segments, weights.shape[-2], weights.shape[-1]).contiguous()

    nb = len(encoding.batches)
    spatial = _per_segment(c, ishape, "spatial weights")
    fan = spatial.movedim(0, nb).contiguous()
    lifted = (*ishape[:nb], 1, *ishape[nb:])
    fanned = (*ishape[:nb], segments, *ishape[nb:])
    C = _Relabel(MultiplySum(fan, lifted, fanned), fanned, ishape)

    E = _SegmentedSense(encoding, basis)
    return _WithNormal(E @ C, C.H @ E.gram() @ C)


#: The gyromagnetic ratio of hydrogen, in Hz per Gauss, as BART's ``wavepsf`` has it.
_LARMOR = 4257.56


def _fftc1(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """The centred unitary transform of a vector of even length."""
    fn = torch.fft.ifft if inverse else torch.fft.fft
    return torch.fft.fftshift(fn(torch.fft.ifftshift(x), norm="ortho"))


def _wave_phase_per_cm(
    readout, cycles, max_grad, max_slew, adc, cosine, delay=0.0, scale=1.0
) -> torch.Tensor:
    """Phase per cm along an oversampled readout of a sine or cosine gradient wave.

    BART's ``wavepsf``: the wave on a 10 µs raster over ``adc`` milliseconds,
    as strong as the amplitude and the slew allow, its phase integrated with
    the trapezoid rule from the pre-phase that centres the sine, and resampled
    to ``readout`` points by zero-filling its centred spectrum.

    ``scale`` multiplies the phase and ``delay``, in milliseconds, moves it
    later in time: a linear phase on its spectrum, so a delay need not fall on
    the raster.  The wave is a whole number of cycles over the readout, so its
    phase is periodic there.  At their defaults the function is BART's.
    """
    dt = 1e-5
    points = int(round(adc * 1e-3 / dt))
    w = 2 * math.pi * cycles / (points * dt)
    amp = max_grad if max_slew >= w * max_grad else max_slew / w

    t = torch.arange(points, dtype=torch.float64) * dt
    g = amp * (torch.cos(w * t) if cosine else torch.sin(w * t))
    before = torch.cumsum(g, 0) - g
    phase = 2 * math.pi * _LARMOR * ((before + g / 2) * dt - amp / w)

    spectrum = _fftc1(phase.to(torch.complex128))
    if delay:
        frequency = (torch.arange(points, dtype=torch.float64) - points // 2) / (points * dt)
        spectrum = spectrum * torch.exp(-2j * math.pi * frequency * (delay * 1e-3))
    start = abs(readout // 2 - points // 2)
    if readout >= points:
        resized = torch.zeros(readout, dtype=torch.complex128)
        resized[start : start + points] = spectrum
    else:
        resized = spectrum[start : start + readout]
    return scale * _fftc1(resized, inverse=True).real * math.sqrt(readout / points)


def _per_axis(value, n: int, what: str) -> tuple[float, ...]:
    """One value for each of ``n`` phase-encode axes, ``(z, y)`` or ``(y,)``."""
    values = (
        tuple(float(v) for v in value) if isinstance(value, (tuple, list)) else (float(value),) * n
    )
    if len(values) != n:
        raise ValueError(f"{what} has {len(values)} values for {n} phase-encode axes")
    return values


def _wave_psf(
    readout, encodes, max_grad, max_slew, cycles, adc, resolution, offset, delay=0.0, scale=1.0
) -> torch.Tensor:
    """The wave point-spread function over ``(*encodes, readout)``.

    A sine wave drives y and, in 3D, a cosine wave drives z; each axis
    contributes ``exp(-i phase_per_cm * location)``, with a location in cm
    measured from the middle voxel and shifted by ``offset``.  ``delay`` and
    ``scale`` are each axis's gradient delay and amplitude error, as
    :func:`_wave_phase_per_cm` applies them.
    """
    encodes = tuple(int(n) for n in encodes)
    resolutions = _per_axis(resolution, len(encodes), "resolution")
    offsets = _per_axis(offset, len(encodes), "offset")
    delays = _per_axis(delay, len(encodes), "delay")
    scales = _per_axis(scale, len(encodes), "scale")

    out = torch.ones((*encodes, readout), dtype=torch.complex128)
    for axis, (n, step, shift) in enumerate(zip(encodes, resolutions, offsets)):
        cosine = len(encodes) == 2 and axis == 0
        ppcm = _wave_phase_per_cm(
            readout, cycles, max_grad, max_slew, adc, cosine, delays[axis], scales[axis]
        )
        location = step * (torch.arange(n, dtype=torch.float64) - n // 2) - shift
        shape = [1] * len(encodes) + [readout]
        shape[axis] = n
        out = out * torch.exp(-1j * location[:, None] * ppcm[None, :]).reshape(shape)
    return out


def _one_coil(values: torch.Tensor, rank: int, what: str) -> torch.Tensor:
    """``values`` without the leading axes of one that one coil's samples do not have."""
    while values.ndim > rank and values.shape[0] == 1:
        values = values.reshape(values.shape[1:])
    if values.ndim > rank:
        raise ValueError(
            f"{what} of {tuple(values.shape)} varies along an axis in front of one coil's "
            f"{rank} sample axes"
        )
    return values


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


def CartesianSense(  # noqa: N802  (it is a constructor)
    sensitivities: torch.Tensor,
    image_shape: Shape,
    pattern: torch.Tensor | None = None,
    *,
    positions: torch.Tensor | None = None,
    readout: str = "kspace",
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
    positions : tensor, optional
        Instead of a pattern, the phase encodes that were sampled, as integer
        indices ``([frames,] shots, d)``: ``(y,)`` in 2D, ``(z, y)`` in 3D,
        ``-1`` for padding where frames sample different numbers.  The samples
        are then ``(*batches, coils, [frames,] shots, readout)``, the whole
        readout along each phase encode, and nothing the size of the
        phase-encode plane is held for them.
    readout : {"kspace", "image"}
        With positions, whether the samples are in k-space along the readout
        or already transformed back along it (hybrid space).  Either way the
        volume is not transformed along the readout: the samples are.
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

    if positions is not None:
        if pattern is not None:
            raise ValueError("give a pattern or positions, not both")
        if kwargs.get("modulated", False):
            raise NotImplementedError("positions are laid out in the centred convention only")
        return _CartesianSampled(
            sensitivities,
            image_shape,
            positions,
            basis,
            readout=readout,
            toeplitz=toeplitz,
            **kwargs,
        )
    if readout != "kspace":
        raise ValueError("the readout convention is for positions; dense samples are k-space")

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
    image_shape: Shape,
    readout: int,
    pattern: torch.Tensor | None = None,
    centred: bool = False,
    *,
    psf: torch.Tensor | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    cycles: int | None = None,
    adc: float | None = None,
    resolution: float | tuple[float, ...] | None = None,
    offset: float | tuple[float, ...] = 0.0,
    delay: float | tuple[float, ...] = 0.0,
    scale: float | tuple[float, ...] = 1.0,
    positions: torch.Tensor | None = None,
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
    readout transform and the phase-encode ones.  So the encoding is what
    ``src/wave.c`` chains:

    ``Sampling . FFT(phase) . Diagonal(psf) . FFT(readout) . Resize .
    Coils(maps)``

    all of it after the coils running in the coil loop a slab at a time,
    where the arithmetic is -- on a card for host arrays -- so what crosses is
    the image and the samples.

    The point-spread function is given, or made from the gradient wave: a sine
    wave along y and, in 3D, a cosine wave along z, as BART's ``wavepsf`` makes
    them and ``fmac`` combines them.

    With a basis this is Wave-Shuffling: the same encoding over coefficient
    images, the basis contracting the coefficients into frames, the pattern or
    the positions keeping the samples.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities ``([sets,] coils, [z,] y, x)``, or their k-space
        kernels with ``kernels=True``.
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
        Centre the two transforms, making them unitary.  BART's ``wave`` leaves
        them uncentred and unnormalized and this follows it; ``wshfl`` centres
        them for its calibration path.
    psf : tensor, optional
        The point-spread function on the oversampled grid, broadcast over one
        coil's samples ``([z,] y, readout)``; the same for every frame.  Give
        it or the gradient wave below.
    max_grad, max_slew : float
        The largest gradient amplitude in G/cm and slew rate in G/cm/s the
        wave may use.  The wave takes the larger amplitude either allows.
    cycles : int
        Sine cycles over the readout.
    adc : float
        Readout duration in milliseconds, on a 10 µs gradient raster.
    resolution : float or tuple of float
        Voxel size in cm along the phase encodes, one value or ``(z, y)``.
    offset : float or tuple of float
        How far the field of view's centre is from the isocentre along the
        phase encodes, in cm, one value or ``(z, y)``.
    delay : float or tuple of float
        How much later than nominal each wave runs, in milliseconds, one value
        or ``(z, y)``.
    scale : float or tuple of float
        How much stronger than nominal each wave is, one value or ``(z, y)``.
    positions : tensor, optional
        Instead of a pattern, the phase encodes that were sampled, as for
        :func:`CartesianSense`: ``([frames,] shots, d)`` with ``(y,)`` in 2D,
        ``(z, y)`` in 3D and ``-1`` for padding.  The samples are then
        ``(*batches, coils, [frames,] shots, readout)``, the oversampled
        readout along each.
    basis : tensor, optional
        Temporal subspace basis ``(coeffs, frames)``.
    toeplitz : bool
        Apply the normal as one coefficient-by-coefficient kernel between the
        phase-encode transforms, with the point-spread function on either
        side, rather than as the two applications.  A pattern that varies
        along the readout has no such form, and keeps the two applications.
    kernels : bool
        Read ``sensitivities`` as k-space kernels.
    coil_batch : int
        Coils applied at once; 0 is every coil at once, as BART chains it.
    device : device, optional
        Where the operator is built and does its arithmetic.
    ndim : int, optional
        Spatial axes, where the sensitivities and the image do not say: a bank
        of four kernel axes is either three behind the coils or two behind
        sets and coils.

    Examples
    --------
    >>> A = WaveSense(maps, (z, y, x), readout=3 * x, pattern=mask,
    ...               max_grad=2.7, max_slew=18700.0, cycles=5, adc=5.0688,
    ...               resolution=(0.1, 0.1))
    >>> A.oshape
    (coils, z, y, 3 * x)
    """
    from bartorch.linop.sense import _grid_ndim

    wave = {
        "max_grad": max_grad,
        "max_slew": max_slew,
        "cycles": cycles,
        "adc": adc,
        "resolution": resolution,
    }
    if psf is None:
        missing = [name for name, value in wave.items() if value is None]
        if missing:
            raise ValueError(f"give psf=, or the gradient wave: {', '.join(missing)} missing")
        spatial_ndim = _grid_ndim(sensitivities, image_shape, kernels, ndim)
        encodes = tuple(image_shape)[len(image_shape) - spatial_ndim : -1]
        psf = _wave_psf(readout, encodes, offset=offset, delay=delay, scale=scale, **wave)
    elif (
        any(value is not None for value in wave.values())
        or offset != 0.0
        or delay != 0.0
        or scale != 1.0
    ):
        raise ValueError(
            "psf= is the point-spread function the gradient wave would make; give one or the other"
        )

    return _WaveNative(
        sensitivities,
        psf,
        image_shape,
        readout,
        pattern,
        positions,
        basis,
        centred,
        toeplitz,
        kernels=kernels,
        coil_batch=coil_batch,
        device=device,
        ndim=ndim,
    )


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
    Over a :class:`~bartorch.linop.NoncartesianSense` with no basis, sets or
    encoding axes, and sample weights that vary along the shots and the
    readout alone, the sum is one operator: the spatial weights fan the image
    out into segment images, and the segments' sample weights are a basis over
    the samples.  Its normal is then the Toeplitz one over the basis's packed
    Gram, with the spatial weights on either side -- a point-spread function
    per pair of segments rather than two transforms per segment per coil.

    Over :func:`CartesianSense` with a pattern or a basis, and over
    :func:`WaveSense`, the segments go into the coil loop instead, around the
    transform each slab carries: the same sum, run where the slab is, with the
    two applications as its normal.  On a grid the closed form would not save
    a transform.  Weights that vary along the batches, the coils, the sets or
    the coefficients keep the sum over the whole encoding, as any other
    encoding does.

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

    from bartorch.linop import plan

    samples = _per_segment(b, encoding.oshape, "sample weights")
    voxels = _per_segment(c, encoding.ishape, "spatial weights")
    return plan.build(plan.Contract(encoding, samples, voxels))


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

"""The non-Cartesian SENSE encoding, applied a slab of coils at a time."""

from __future__ import annotations

import torch

from bartorch._dispatch import _lock
from bartorch._lib import library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator

__all__ = ["Coils", "NoncartesianSense"]


def _without_maps(shape: tuple[int, ...]) -> tuple[int, ...]:
    """``shape`` with BART's MAPS axis set to one.

    The operator sums the sets of maps on the way out, so the samples never
    carry them.  In the arrangement here -- ``(coeffs, te, maps, coils,
    *spatial)`` -- that axis is the third; a shape short enough not to have
    one has no sets to drop.
    """
    return (*shape[:2], 1, *shape[3:]) if len(shape) > 4 else shape


def _bank(sensitivities: torch.Tensor, coils: int, spatial: tuple[int, ...]):
    """``(bank, sets, sens_spatial)`` from sensitivities held either way.

    A bank is ``(coils, *spatial)`` for the one set a single-map calibration
    gives, and ``(sets, coils, *spatial)`` for the several that ESPIRiT's
    second map and ENLIVE's relaxed model give -- which is the shape
    :func:`bartorch.tools.ecalib` and :func:`bartorch.tools.nlinv` return for
    ``maps > 1``, with the spatial axes written out.

    Writing them out is what tells the two apart: a bank of one set is at most
    ``(coils, z, y, x)``, so anything with an axis beyond that and the coils in
    second place is carrying sets.  A two-dimensional bank may still leave
    BART's third spatial axis out, and then it is one set by construction.
    """
    s = as_operand(sensitivities, tuple(sensitivities.shape), "sensitivities")
    shape = tuple(s.shape)

    if len(shape) == len(spatial) + 2 and shape[1] == coils:
        sets, sens_spatial = shape[0], shape[2:]
    elif shape[0] == coils:
        sets, sens_spatial = 1, shape[1:]
        if len(sens_spatial) == 2:
            sens_spatial = (1, *sens_spatial)
    else:
        raise ValueError(
            f"sensitivities of {shape} are neither (coils, *spatial) nor "
            f"(sets, coils, *spatial) for {coils} coils"
        )

    return s.reshape(sets, coils, *sens_spatial), sets, tuple(sens_spatial)


class NoncartesianSense(LinearOperator):
    """Sensitivities followed by a NUFFT, applied ``coil_batch`` coils at a time.

    Memory held -- and the doubled grid the Toeplitz normal convolves on --
    scales with ``coil_batch`` rather than with the number of coils.

    On a grid the operator to use is
    :func:`~bartorch.linop.CartesianSense`, which is this one's Cartesian path
    under the name that says so.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities of shape ``(coils, *image_shape[1:])``, or their
        k-space kernels when ``kernels`` is set.  A bank on the host while the
        transform is on a card is transferred a slab at a time.

        Several sets of maps -- ESPIRiT's second, ENLIVE's relaxed model --
        are ``(sets, coils, *spatial)``, which is what
        :func:`bartorch.tools.ecalib` and :func:`bartorch.tools.nlinv` return
        for ``maps > 1``.  The image then carries the sets and the samples do
        not: the encoding is ``y[c] = sum_m S[m, c] x[m]``, summed in BART's
        own contraction rather than by anything here.
    image_shape : tuple of int
        Coil-image shape, C order, for instance ``(coils, y, x)``.
    traj : tensor
        Trajectory in grid units, as :func:`bartorch.tools.traj` produces.
    kspace_shape : tuple of int, optional
        Sample shape; by default the trajectory's, or the image's on a grid.
    kernels : bool
        Read ``sensitivities`` as k-space kernels, zero-padded to the image grid
        and transformed a slab at a time.  The operator then applies the maps
        band-limited to the kernel; :func:`bartorch.maps_to_kernels` makes such
        kernels and :func:`bartorch.kernels_to_maps` gives the maps they stand for.
    toeplitz : bool
        Apply the normal as a convolution with a point spread function.
    modulated : bool
        On a grid, answer in BART's own sample convention rather than the
        centred one -- a scale and a modulation folded into the sensitivities
        and the plain transform after them, which is what ``pics`` works in
        and what its k-space is written in.  The two differ by an ``fftmod``
        on the sample axes; the default is the centred convention, which is
        what :func:`bartorch.fft` produces and so what an operator chained
        against one expects.

        It does not depend on ``coil_batch``: every slab answers in the
        convention that was asked for.  At ``coil_batch=0`` this is BART's own
        operator, arithmetic and all, which is what reproduces ``pics`` to the
        last bit.  Refused off a grid, where there is only one convention, and
        with ``kernels``, because the modulation is the whole grid's and a
        kernel cannot carry it.
    weights : tensor, optional
        Diagonal in k-space, as :class:`~bartorch.linop.NUFFT` takes it.
    basis : tensor, optional
        Subspace basis ``(coeffs, frames, 1, 1, 1, 1, 1)``, as
        :class:`~bartorch.linop.NUFFT` takes it.  The image then has shape
        ``(coeffs, 1, 1, 1, *spatial)`` and the samples one set per frame.
    device : device, optional
        Where the operator is built and does its arithmetic; by default where the
        trajectory is, or the sensitivities for a Cartesian operator.  With a card
        here, operands may stay on the host: the image crosses once each way per
        application, the samples a slab at a time, and between applications the
        card holds the operator only.
    coil_batch : int
        Coils applied at once; 0 uses BART's own operator over all coils.  A
        larger batch is faster and holds proportionally more.  A single-coil
        operator is always BART's own.  A batch that does not divide the coils
        is cut down to one that does, because the loop steps by the slab and
        the transform is built for a slab.

        What it changes is residency, not arithmetic.  The sample convention
        is ``modulated``'s to say and not this one's.
    fold_maps : bool
        Apply the sensitivities inside the transform of the normal, which saves
        two coil images per batch.  Takes effect only where the transform works
        one coefficient at a time, the gathered arrangement of a compressed
        Toeplitz function.
    """

    #: Whether a trajectory is required.  The Cartesian encoding is the same
    #: operator over BART's own FFT, and reaches it by clearing this.
    _needs_traj = True

    #: How many subspace coefficients the image carries when no basis is handed
    #: to BART.  A basis says this for itself; the grid encoding sets it,
    #: because on a grid the contraction is chained on afterwards rather than
    #: folded into the transform -- which is how ``grecon/model.c`` builds it.
    _coeff_count = 1

    def __init__(
        self,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        traj: torch.Tensor | None = None,
        kspace_shape: Shape | None = None,
        kernels: bool = False,
        toeplitz: bool = True,
        modulated: bool = False,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        coil_batch: int = 1,
        fold_maps: bool = True,
    ):
        image_shape = tuple(image_shape)
        if len(image_shape) < 3:
            raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")
        if traj is None and self._needs_traj:
            raise ValueError("this is the encoding off a grid; CartesianSense is the one on it")
        if traj is None and (weights is not None or basis is not None):
            raise ValueError("weights and a basis belong to a non-Cartesian transform")
        if traj is not None and modulated:
            raise ValueError("the modulated convention is the grid's; off it there is only one")

        # BART reads the coils off a dimension of their own, which sits after
        # the three spatial ones, so a two-dimensional problem carries the
        # third as a singleton.  The caller need not write it out.
        coils, spatial = image_shape[0], image_shape[1:]
        if len(spatial) == 2:
            spatial = (1, *spatial)

        s, sets, sens_spatial = _bank(sensitivities, coils, spatial)

        self.sensitivities = s
        self.sets = sets
        self.image_shape = image_shape
        self.kernels = bool(kernels)
        self.toeplitz = bool(toeplitz)
        self.modulated = bool(modulated)
        self.traj = None if traj is None else as_operand(traj, tuple(traj.shape), "traj")
        self.weights = (
            None if weights is None else as_operand(weights, tuple(weights.shape), "weights")
        )
        self.basis = None if basis is None else as_operand(basis, tuple(basis.shape), "basis")
        self._sens_shape = (sets, coils, *sens_spatial)
        if coil_batch < 0:
            raise ValueError(f"coil_batch is a number of coils, not {coil_batch}")
        self.coil_batch = int(coil_batch)
        self.fold_maps = bool(fold_maps)

        # A basis puts the coefficients on BART's COEFF axis, three past the
        # coils, and sets of maps on its MAPS axis, which is the one between.
        # The image carries both where the coils do not; the samples carry
        # neither the maps, which the operator sums over, nor -- behind a
        # basis -- the coefficients, which it contracts.
        b = self.basis
        coeffs = self._coeff_count if b is None else int(b.shape[0])
        if b is None and 1 == coeffs and 1 == sets:
            self._max_shape = (coils, *spatial)
            self.ishape = spatial
        else:
            self._max_shape = (coeffs, 1, sets, coils, *spatial)
            self.ishape = (coeffs, 1, sets, 1, *spatial)

        self.kspace_shape = tuple(
            self._default_kspace(coils) if kspace_shape is None else kspace_shape
        )

        # Where the operator is built follows the transform's own data, not
        # the sensitivities: a bank left on the host is the point.  Named, it
        # may be a card none of the inputs are on.
        if device is not None:
            built_on = torch.device(device)
            if built_on.type == "cuda" and built_on.index is None:
                built_on = torch.device("cuda", torch.cuda.current_device())
        else:
            built_on = s.device if self.traj is None else self.traj.device
        self.device = built_on

        super().__init__()

    def _default_kspace(self, coils: int) -> Shape:
        if self.traj is None:
            # The samples do not carry the sets of maps: the operator sums
            # over them on the way out and hands them back on the way in.
            return _without_maps(self._max_shape)
        # BART lays non-Cartesian samples out as the trajectory is, with the
        # coordinate axis a singleton and the coils on the axis after the
        # spokes -- which is where the sensitivities have them, rather than
        # the one that carries sets of maps.
        bart = [1, *list(self.traj.shape)[::-1][1:]]
        bart += [1] * max(0, 4 - len(bart))
        if bart[3] != 1:
            raise ValueError("the trajectory already has an axis where the coils go")
        bart[3] = coils
        return tuple(bart[::-1])

    def _create(self) -> Built:
        t, w, b = self.traj, self.weights, self.basis
        lib = library()
        # The library reads both settings from process-wide state when it builds
        # an operator, so they are set for this build and restored after it.
        with _lock:
            was = lib.bartorch_sense_coil_batch(), lib.bartorch_sense_fold_maps()
            lib.bartorch_sense_set_coil_batch(self.coil_batch)
            lib.bartorch_sense_set_fold_maps(int(self.fold_maps))
            try:
                ptr = self._build_sense(lib, t, w, b)
            finally:
                lib.bartorch_sense_set_coil_batch(was[0])
                lib.bartorch_sense_set_fold_maps(was[1])
        keep = tuple(x for x in (self.sensitivities, t, w, b) if x is not None)
        return Built(ptr, self.ishape, self.kspace_shape, keep=keep, device=self.device)

    def _build_sense(self, lib, t, w, b) -> int:
        return self._under_lock(
            lib.bartorch_linop_sense,
            dims(self._max_shape),
            dims(self.kspace_shape),
            dims(self._sens_shape),
            self.sensitivities.data_ptr(),
            int(self.kernels),
            None if t is None else dims(tuple(t.shape)),
            None if t is None else t.data_ptr(),
            None if w is None else dims(tuple(w.shape)),
            None if w is None else w.data_ptr(),
            None if b is None else dims(tuple(b.shape)),
            None if b is None else b.data_ptr(),
            int(self.toeplitz),
            int(self.modulated),
            device=self.device,
        )


class Coils(LinearOperator):
    """Coil sensitivities, without the transform that usually follows them.

    The multiply on its own: an image to coil images, and the conjugate
    sensitivities summed over the coils on the way back.  For an encoding
    whose transform is not a Fourier transform, and which therefore cannot be
    a SENSE operator.

    Held as maps it is what :class:`~bartorch.linop.MultiplySum` builds, and
    with ``coil_batch=0`` it *is* that operator.  Held as kernels, or walked a
    slab at a time, the bank is never resident whole -- the arrangement
    :class:`NoncartesianSense` uses.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities of shape ``(coils, *image_shape[1:])``, or their
        k-space kernels when ``kernels`` is set.  Several sets of maps are
        ``(sets, coils, *spatial)``, as :class:`NoncartesianSense` takes them.
    image_shape : tuple of int
        Coil-image shape, C order, for instance ``(coils, y, x)``.
    kernels : bool
        Read ``sensitivities`` as k-space kernels, zero-padded to the image
        grid and transformed a slab at a time, as :class:`NoncartesianSense`
        reads them.
    device : device, optional
        Where the operator is built and does its arithmetic; by default where
        the sensitivities are.
    coil_batch : int
        Coils applied at once; 0 uses BART's own ``fmac`` over all of them.
    coeffs : int
        Subspace coefficients the image carries.  With more than one the
        domain is ``(coeffs, 1, 1, 1, *spatial)`` and the codomain
        ``(coeffs, 1, 1, coils, *spatial)``: the sensitivities are the same for
        every coefficient, so nothing about the multiply changes.

    Examples
    --------
    >>> S = Coils(kernels, (coils, y, x), kernels=True)
    >>> A = FFT(S.oshape, axes=(-2, -1)) @ S
    """

    def __init__(
        self,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        kernels: bool = False,
        device: torch.device | str | None = None,
        coil_batch: int = 1,
        coeffs: int = 1,
    ):
        image_shape = tuple(image_shape)
        if len(image_shape) < 3:
            raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")
        if coeffs < 1:
            raise ValueError(f"coeffs is a number of coefficients, not {coeffs}")

        coils, spatial = image_shape[0], image_shape[1:]
        if len(spatial) == 2:
            spatial = (1, *spatial)

        s, sets, sens_spatial = _bank(sensitivities, coils, spatial)

        if coil_batch < 0:
            raise ValueError(f"coil_batch is a number of coils, not {coil_batch}")

        self.sensitivities = s
        self.image_shape = image_shape
        self.kernels = bool(kernels)
        self.coil_batch = int(coil_batch)
        self.coeffs = int(coeffs)
        self.sets = sets
        if 1 == self.coeffs and 1 == sets:
            self._max_shape = (coils, *spatial)
            self.ishape = spatial
        else:
            # The coefficients ride on BART's COEFF axis, three past the
            # coils, and sets of maps on its MAPS axis, the one between.
            self._max_shape = (self.coeffs, 1, sets, coils, *spatial)
            self.ishape = (self.coeffs, 1, sets, 1, *spatial)
        # The coil images are summed over the sets, so they do not carry them.
        self.oshape = _without_maps(self._max_shape)
        self._sens_shape = (sets, coils, *sens_spatial)
        self.device = torch.device(device) if device is not None else s.device

        super().__init__()

    def _create(self) -> Built:
        lib = library()
        # As in NoncartesianSense: the slab size is read from process-wide
        # state when the operator is built, so it is set for this build alone.
        with _lock:
            was = lib.bartorch_sense_coil_batch()
            lib.bartorch_sense_set_coil_batch(self.coil_batch)
            try:
                ptr = self._under_lock(
                    lib.bartorch_linop_coils,
                    dims(self._max_shape),
                    dims(self._sens_shape),
                    self.sensitivities.data_ptr(),
                    int(self.kernels),
                    device=self.device,
                )
            finally:
                lib.bartorch_sense_set_coil_batch(was)
        return Built(ptr, self.ishape, self.oshape, keep=(self.sensitivities,), device=self.device)

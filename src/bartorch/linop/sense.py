"""The non-Cartesian SENSE encoding, applied a slab of coils at a time."""

from __future__ import annotations

import torch

from bartorch._dispatch import _lock
from bartorch._lib import library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator

__all__ = ["NoncartesianSense"]


class NoncartesianSense(LinearOperator):
    """Sensitivities followed by a NUFFT, applied ``coil_batch`` coils at a time.

    Memory held -- and the doubled grid the Toeplitz normal convolves on --
    scales with ``coil_batch`` rather than with the number of coils.

    A trajectory is what this operator is for.  On a grid the operator is
    :func:`~bartorch.linop.CartesianSense`, which is this one's own Cartesian
    path under the name that says so.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities of shape ``(coils, *image_shape[1:])``, or their
        k-space kernels when ``kernels`` is set.  A bank on the host while the
        transform is on a card is transferred a slab at a time.
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
        operator is always BART's own.
    fold_maps : bool
        Apply the sensitivities inside the transform of the normal, which saves
        two coil images per batch.  Takes effect only where the transform works
        one coefficient at a time, the gathered arrangement of a compressed
        Toeplitz function.
    """

    #: Whether a trajectory is required.  The Cartesian encoding is the same
    #: operator over BART's own FFT, and reaches it by clearing this.
    _needs_traj = True

    def __init__(
        self,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        traj: torch.Tensor | None = None,
        kspace_shape: Shape | None = None,
        kernels: bool = False,
        toeplitz: bool = True,
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

        # BART reads the coils off a dimension of their own, which sits after
        # the three spatial ones, so a two-dimensional problem carries the
        # third as a singleton.  The caller need not write it out.
        coils, spatial = image_shape[0], image_shape[1:]
        if len(spatial) == 2:
            spatial = (1, *spatial)

        s = as_operand(sensitivities, tuple(sensitivities.shape), "sensitivities")
        if s.shape[0] != coils:
            raise ValueError(f"{s.shape[0]} sensitivities for {coils} coils")
        sens_spatial = tuple(s.shape[1:])
        if len(sens_spatial) == 2:
            sens_spatial = (1, *sens_spatial)
            s = s.reshape(coils, *sens_spatial)

        self.sensitivities = s
        self.image_shape = image_shape
        self.kernels = bool(kernels)
        self.toeplitz = bool(toeplitz)
        self.traj = None if traj is None else as_operand(traj, tuple(traj.shape), "traj")
        self.weights = (
            None if weights is None else as_operand(weights, tuple(weights.shape), "weights")
        )
        self.basis = None if basis is None else as_operand(basis, tuple(basis.shape), "basis")
        self._sens_shape = (coils, *sens_spatial)
        if coil_batch < 0:
            raise ValueError(f"coil_batch is a number of coils, not {coil_batch}")
        self.coil_batch = int(coil_batch)
        self.fold_maps = bool(fold_maps)

        # A basis puts the coefficients on BART's COEFF axis, three past the
        # coils, and the image carries that axis where the coils do not.
        b = self.basis
        coeffs = 1 if b is None else int(b.shape[0])
        if b is None:
            self._max_shape = (coils, *spatial)
            self.ishape = spatial
        else:
            self._max_shape = (coeffs, 1, 1, coils, *spatial)
            self.ishape = (coeffs, 1, 1, 1, *spatial)

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
            return self._max_shape
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
            device=self.device,
        )

"""The SENSE encoding, over one slab of coils at a time."""

from __future__ import annotations

import torch

from bartorch._lib import library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import BartLinearOperator

__all__ = ["Sense"]


class Sense(BartLinearOperator):
    """Sensitivities and a transform, walking the coils a slab at a time.

    The coils are independent until the sum that ends the adjoint, so this
    applies a slab of sensitivities, transforms that slab and sums it in.
    What is resident is a slab rather than the whole bank, and the transform
    is built for a slab, so the grid its Toeplitz normal convolves on shrinks
    with it.  :func:`bartorch.set_coil_batch` is how many coils a slab holds.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities of shape ``(coils, *image_shape[1:])``, or the
        k-space kernels they band-limit to when ``kernels`` is set.  A bank
        left on the host while the transform is on a card is brought over a
        slab at a time, so the bank itself never has to fit.
    image_shape : tuple of int
        Coil-image shape, C order, for instance ``(coils, y, x)``.
    traj : tensor, optional
        Trajectory in grid units; without one this is the Cartesian operator
        and the transform is the centred unitary FFT, which is
        ``bartorch.tools.fft(..., unitary=True)``.
    kspace_shape : tuple of int, optional
        Sample shape; by default the trajectory's, or the image's on a grid.
    kernels : bool
        Read ``sensitivities`` as kernels: the centre of each map's spectrum,
        which is all a smooth map carries.  A slab is padded back on to the
        image grid and transformed when it is needed, so a bank that would not
        fit is never held.  What the operator applies is then the maps
        band-limited to the kernel, which :func:`bartorch.maps_to_kernels`
        reports the error of.
    toeplitz : bool
        Apply the normal through the Toeplitz embedding.
    weights : tensor, optional
        A diagonal in k-space, as :class:`~bartorch.linop.NUFFT` takes it.
    basis : tensor, optional
        A subspace basis over frames and coefficients, as
        :class:`~bartorch.linop.NUFFT` takes it: ``(coeffs, frames, 1, 1, 1, 1, 1)``.
        The image then carries one volume per coefficient,
        ``(coeffs, 1, 1, 1, *spatial)``, and the samples one set per frame.
    device : device, optional
        Where the operator is built and does its arithmetic.  Without one,
        that is where the trajectory is, or the sensitivities on a grid.
        Given a card, every operand may be on the host: an image crosses
        whole, once each way, the samples cross a slab at a time, and between
        two applications the card holds the operator and nothing of the
        caller's -- so a solver that keeps its vectors on the host uses the
        card for the operator alone.
    """

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
    ):
        image_shape = tuple(image_shape)
        if len(image_shape) < 3:
            raise ValueError("image_shape is (coils, *spatial), for instance (coils, y, x)")
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
        ptr = self._under_lock(
            library().bartorch_linop_sense,
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
        keep = tuple(x for x in (self.sensitivities, t, w, b) if x is not None)
        return Built(ptr, self.ishape, self.kspace_shape, keep=keep, device=self.device)

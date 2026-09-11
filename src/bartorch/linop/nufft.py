"""The non-uniform FFT operator, computed by FINUFFT through BART's ``nufft_create``."""

from __future__ import annotations

import torch

from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator

__all__ = ["NUFFT"]


def default_kspace_shape(traj_shape: Shape, image_shape: Shape, ndim: int) -> Shape:
    """Where BART puts the samples of a trajectory, given the image beside it."""
    # BART: traj dims [3, samples, spokes, ...], kspace dims [1, samples, spokes, coils, ...].
    # In C order the trajectory is (..., spokes, samples, 3) and the coil axes of
    # the image sit in front of its spatial ones -- two or three of them, which
    # only the trajectory can say, since (coils, y, x) and (z, y, x) are the same
    # shape.
    coils = tuple(image_shape[:-ndim])
    return coils + tuple(traj_shape[:-1]) + (1,)


class NUFFT(LinearOperator):
    """Non-uniform FFT from coil images to samples along a trajectory.

    Parameters
    ----------
    traj : tensor
        Trajectory of shape ``(..., samples, 3)`` in grid units, as
        :func:`bartorch.tools.traj` produces.  Its third component being zero
        makes the transform two-dimensional.
    image_shape : tuple of int
        Coil-image shape, C order, for instance ``(coils, y, x)``.
    kspace_shape : tuple of int, optional
        Sample shape; by default the trajectory's, with the coordinate axis
        replaced by the image's coil axes.
    weights : tensor, optional
        Diagonal in k-space, applied on the way out and conjugated on the way
        back.
    basis : tensor, optional
        Subspace basis over frames and coefficients, contracting the image's
        coefficients into k-space frames.  The weights and the basis are part
        of the operator because its Toeplitz normal is built over both.
    toeplitz : bool
        Apply the normal as a convolution with a point spread function.
    oversampling, width : float
        Grid oversampling and kernel width; zero keeps the defaults.

    Examples
    --------
    >>> A = NUFFT(traj, image_shape=(8, 128, 128))
    >>> A.adjoint(kspace).shape
    torch.Size([8, 128, 128])
    """

    def __init__(
        self,
        traj: torch.Tensor,
        image_shape: Shape,
        kspace_shape: Shape | None = None,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        toeplitz: bool = True,
        oversampling: float = 0.0,
        width: float = 0.0,
    ):
        self.traj = as_operand(traj, tuple(traj.shape), "traj")
        self.image_shape = tuple(image_shape)
        self.weights = (
            None if weights is None else as_operand(weights, tuple(weights.shape), "weights")
        )
        self.basis = None if basis is None else as_operand(basis, tuple(basis.shape), "basis")
        self.toeplitz = bool(toeplitz)
        self.oversampling = float(oversampling)
        self.width = float(width)

        if kspace_shape is None:
            from bartorch import _finufft

            kspace_shape = default_kspace_shape(
                tuple(self.traj.shape), self.image_shape, _finufft.spatial_ndim(self.traj)
            )
        self.kspace_shape = tuple(kspace_shape)
        super().__init__()

    def _create(self) -> Built:
        w, b = self.weights, self.basis
        ptr = self._under_lock(
            library().bartorch_linop_nufft,
            DIMS,
            dims(self.kspace_shape),
            dims(self.image_shape),
            dims(tuple(self.traj.shape)),
            self.traj.data_ptr(),
            None if w is None else dims(tuple(w.shape)),
            None if w is None else w.data_ptr(),
            None if b is None else dims(tuple(b.shape)),
            None if b is None else b.data_ptr(),
            int(self.toeplitz),
            self.oversampling,
            self.width,
            device=self.traj.device,
        )
        keep = tuple(x for x in (self.traj, w, b) if x is not None)
        return Built(ptr, self.image_shape, self.kspace_shape, keep=keep)

"""The non-uniform FFT operator, computed by FINUFFT through BART's ``nufft_create``."""

from __future__ import annotations

import torch

from bartorch import _layout
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator

__all__ = ["NUFFT"]


def default_kspace_shape(traj_shape: Shape, image_shape: Shape, ndim: int) -> Shape:
    """The samples of a trajectory, given the image beside it.

    The image is ``(*batches, *encoding, [z,] y, x)`` and the trajectory
    ``(*encoding, shots, samples, d)``, so the samples are the image's batches
    followed by the trajectory without its coordinate axis.
    """
    encoding = len(traj_shape) - 3
    batches, _ = _layout.split(tuple(image_shape), encoding + ndim, "the image")
    return (*batches, *traj_shape[:-1])


class NUFFT(LinearOperator):
    """Non-uniform FFT from images to samples along a trajectory.

    Parameters
    ----------
    traj : tensor
        Trajectory ``(*encoding, shots, samples, 3)`` in grid units, as
        :func:`bartorch.tools.traj` produces.  Its third component being zero
        everywhere makes the transform two-dimensional.  The encoding axes are
        whatever the samples vary along besides the shots -- frames, echoes,
        cardiac phases -- in any number.
    image_shape : tuple of int
        Image shape ``(*batches, *encoding, [z,] y, x)``: two spatial axes for
        a two-dimensional trajectory and three for a three-dimensional one.
        The batches -- coils, slices, anything transformed alike -- are
        whatever leads the encoding axes, and each is transformed on its own.
    kspace_shape : tuple of int, optional
        Sample shape; by default ``(*batches, *encoding, shots, samples)``.
    weights : tensor, optional
        Diagonal in k-space, broadcast over ``(*encoding, shots, samples)``,
        applied on the way out and conjugated on the way back.
    basis : tensor, optional
        Subspace basis ``(coeffs, frames)``, contracting the image's
        coefficients into the frames of the last encoding axis: that axis is
        ``frames`` long in k-space and ``coeffs`` long in the image.  The
        weights and the basis are part of the operator because its Toeplitz
        normal is built over both.
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
        from bartorch import _finufft

        self.traj = as_operand(traj, tuple(traj.shape), "traj")
        if self.traj.ndim < 3:
            raise ValueError(
                f"a trajectory is (*encoding, shots, samples, 3), not {tuple(self.traj.shape)}"
            )
        self.image_shape = tuple(image_shape)
        self.ndim = _finufft.spatial_ndim(self.traj)
        self.encoding = tuple(self.traj.shape[:-3])

        self.basis = None
        coeffs = None
        if basis is not None:
            b = as_operand(basis, tuple(basis.shape), "basis")
            if b.ndim < 2 or any(n != 1 for n in b.shape[2:]):
                raise ValueError(f"a basis is (coeffs, frames), not {tuple(b.shape)}")
            if not self.encoding:
                raise ValueError("a basis contracts an encoding axis, and the trajectory has none")
            coeffs, frames = int(b.shape[0]), int(b.shape[1])
            if frames != self.encoding[-1]:
                raise ValueError(
                    f"the basis has {frames} frames and the trajectory's last encoding axis "
                    f"{self.encoding[-1]}"
                )
            self.basis = b.reshape(coeffs, frames)

        image_encoding = self.encoding if coeffs is None else (*self.encoding[:-1], coeffs)
        batches, rest = _layout.split(self.image_shape, len(self.encoding) + self.ndim, "the image")
        if tuple(rest[: len(self.encoding)]) != tuple(image_encoding):
            raise ValueError(
                f"the image {self.image_shape} does not carry the encoding axes "
                f"{tuple(image_encoding)} before its {self.ndim} spatial ones"
            )
        self.batches = tuple(batches)
        self.spatial = tuple(rest[len(self.encoding) :])

        self.kspace_shape = tuple(
            default_kspace_shape(tuple(self.traj.shape), self.image_shape, self.ndim)
            if kspace_shape is None
            else kspace_shape
        )
        if self.kspace_shape != (*self.batches, *self.traj.shape[:-1]):
            raise ValueError(
                f"the samples of this trajectory and image are "
                f"{(*self.batches, *self.traj.shape[:-1])}, not {self.kspace_shape}"
            )

        self.weights = None
        if weights is not None:
            per_block = tuple(self.traj.shape[:-1])
            w = as_operand(weights, tuple(weights.shape), "weights")
            got = (1,) * (len(per_block) - w.ndim) + tuple(w.shape)
            if len(got) != len(per_block) or any(g not in (1, f) for g, f in zip(got, per_block)):
                raise ValueError(f"weights of {tuple(w.shape)} do not broadcast over {per_block}")
            self.weights = w.reshape(got)

        self.toeplitz = bool(toeplitz)
        self.oversampling = float(oversampling)
        self.width = float(width)
        super().__init__()

    def _whole(self) -> bool:
        """Whether one BART operator describes every batch item.

        Without encoding axes the batches are the slowest axes in both layouts
        and lie on BART's coil axis side by side, as BART batches coils: one
        plan transforms them all.  With encoding axes they sit behind them in
        memory, which BART's order of roles cannot express, and a batch item
        at a time is what is built.
        """
        return not self.encoding

    def _block_vectors(self):
        """BART dimension vectors one operator is built for.

        The operator holds every batch item on the coil axis where
        :meth:`_whole` says so, and one batch item otherwise.  Returned as
        k-space, image, trajectory, weights and basis.
        """
        shots, samples, d = (int(n) for n in self.traj.shape[-3:])
        kdims, idims = _layout.encoding_dims(len(self.encoding), self.basis is not None)
        batch = _layout.count(self.batches) if self._whole() else 1

        k = {1: samples, 2: shots, _layout.COIL: batch}
        t = {0: d, 1: samples, 2: shots}
        for dim, n in zip(kdims, self.encoding):
            k[dim] = n
            t[dim] = n

        spatial = self.spatial if self.ndim == 3 else (1, *self.spatial)
        i = {
            _layout.READ: spatial[2],
            _layout.PHS1: spatial[1],
            _layout.PHS2: spatial[0],
            _layout.COIL: batch,
        }
        image_encoding = (
            self.encoding
            if self.basis is None
            else (
                *self.encoding[:-1],
                int(self.basis.shape[0]),
            )
        )
        for dim, n in zip(idims, image_encoding):
            i[dim] = n

        w = None
        if self.weights is not None:
            wshape = tuple(self.weights.shape)
            w = {1: wshape[-1], 2: wshape[-2]}
            for dim, n in zip(kdims, wshape[:-2]):
                w[dim] = n
            w = _layout.vector(w)

        b = None
        if self.basis is not None:
            b = _layout.vector(
                {_layout.TE: self.basis.shape[1], _layout.COEFF: self.basis.shape[0]}
            )

        return _layout.vector(k), _layout.vector(i), _layout.vector(t), w, b

    def _create(self) -> Built:
        lib = library()
        kvec, ivec, tvec, wvec, bvec = self._block_vectors()
        w, b = self.weights, self.basis
        block = self._under_lock(
            lib.bartorch_linop_nufft,
            DIMS,
            _vector(kvec),
            _vector(ivec),
            _vector(tvec),
            self.traj.data_ptr(),
            None if w is None else _vector(wvec),
            None if w is None else w.data_ptr(),
            None if b is None else _vector(bvec),
            None if b is None else b.data_ptr(),
            int(self.toeplitz),
            self.oversampling,
            self.width,
            device=self.traj.device,
        )
        if not block:
            ptr = 0
        else:
            try:
                if self._whole():
                    ptr = self._under_lock(
                        lib.bartorch_linop_reshaped,
                        block,
                        DIMS,
                        dims(self.kspace_shape),
                        dims(self.image_shape),
                        device=self.traj.device,
                    )
                else:
                    ptr = self._under_lock(
                        lib.bartorch_linop_blocks,
                        block,
                        DIMS,
                        dims(self.kspace_shape),
                        dims(self.image_shape),
                        _layout.count(self.batches),
                        device=self.traj.device,
                    )
            finally:
                with_lock_free(lib, block)
        keep = tuple(x for x in (self.traj, w, b) if x is not None)
        return Built(ptr, self.image_shape, self.kspace_shape, keep=keep)


def _vector(v):
    """A BART dimension vector as the C entry points take it.

    ``dims`` reverses a C-order shape into BART's order, so a vector already in
    BART's order goes through reversed.
    """
    return dims(tuple(v)[::-1])


def with_lock_free(lib, ptr) -> None:
    """Free a block operator the loop over blocks has taken its own reference to."""
    from bartorch._dispatch import _lock

    with _lock:
        lib.bartorch_linop_free(ptr)

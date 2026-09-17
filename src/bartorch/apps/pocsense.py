"""``pocsense`` as a pipeline: the three projections, and the sweep over them."""

from __future__ import annotations

import math

import torch

import bartorch
from bartorch import linop, optim, priors
from bartorch._operator import Shape
from bartorch.linop import basic
from bartorch.optim.blocks import _prox
from bartorch.tools import sampling

__all__ = ["pocsense"]

#: BART's own ``-i`` default (pocsense.c:66).
_MAXITER = 50


class _Consistency:
    """The measured samples put back, ``mri2.c``'s ``data_consistency``.

    ``(I - P) x + P y``: where a sample was taken the measurement stands, and
    elsewhere the iterate is left alone.  With ``robust`` the residual at a
    measured position is soft-thresholded instead of discarded, which is
    ``pocs.c``'s ``robust_consistency``.
    """

    def __init__(self, P, measured: torch.Tensor, robust: float | None = None):
        self.P = P
        self.measured = measured
        self.sampled = P(measured)
        self.robust = robust

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.robust is not None:
            kept = self.measured + bartorch.soft_thresh(self.robust, x - self.measured)
            return (x - self.P(x)) + self.P(kept)
        return (x - self.P(x)) + self.sampled


class _Range:
    """The range of the coil sensitivities, ``pocs.c``'s ``sense_proj_apply``.

    ``E E^H``, which is a projection because the encoding carries the
    sensitivities scaled the way ``maps_create`` scales them.
    """

    def __init__(self, A):
        self.A = A

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(self.A.H(x))


class _Sparsity:
    """A threshold on the coil images, ``pocs.c``'s ``sparsity_proj_apply``.

    The term's proximal operator conjugated by the transform between the
    samples and the coil images: the uncentred transform, and beside it the
    modulation and the scaling that together make it the unitary centred one
    in the convention the application modulates its k-space into.

    The two ride in one array, as ``pocs.c`` builds them --
    ``fftmod(fftscale(ones))`` -- and are applied by a diagonal operator, whose
    adjoint is the conjugate multiply on the way in.  Both are then BART's own
    ``md_zmul2`` and ``md_zmulc2``, which is what an odd axis needs: there the
    modulation is a phase rather than a sign, and a complex product computed
    with a fused multiply-add does not round where two multiplies and a sum
    round.
    """

    def __init__(self, term, axes: tuple[int, ...], shape: Shape):
        self.term = term
        self.axes = axes
        voxels = math.prod(shape[axis] for axis in axes)
        broadcast = (1, *shape[1:])
        ones = torch.full(broadcast, 1.0 / math.sqrt(voxels), dtype=torch.complex64)
        # `fftmod` is a command, and a command drops the leading axes that are
        # one; the diagonal is broadcast along the coils, so it needs them.
        modulation = bartorch.fftmod(ones, axes).reshape(broadcast)
        self.modulation = basic.Diagonal(modulation, shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        image = self.modulation.H(bartorch.ifft(x, self.axes, uncentred=True))
        image = _prox(self.term, image, 1.0, tuple(x.shape))
        return bartorch.fft(self.modulation(image), self.axes, uncentred=True)


def pocsense(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    *,
    maxiter: int | None = None,
    alpha: float = 0.0,
    wavelet: bool = False,
    robust: float | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    """POCSENSE reconstruction: the measured samples, the coils, and sparsity.

    The pipeline :func:`bartorch.tools.pocsense` runs, assembled here: the
    scaling the application estimates, the sampling pattern read off the
    k-space, the modulation into the convention BART iterates in, and then a
    sweep of three projections from :mod:`bartorch.linop` and
    :mod:`bartorch.priors` under :class:`bartorch.optim.POCS`.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order ``(coils, z, y, x)``.
    sensitivities : torch.Tensor
        Coil sensitivities, normalized, as :func:`~bartorch.tools.ecalib`
        produces them.
    maxiter : int, optional
        Sweeps of the projections; BART's default is fifty.
    alpha : float
        Regularization weight.  Zero leaves the sparsity projection out
        entirely, as the application does, and the sweep is then the two
        projections onto the data and onto the coils.
    wavelet : bool
        Threshold the wavelet coefficients of the coil images rather than
        shrinking the samples towards zero, which is ``pocsense -l 1``.
    robust : float, optional
        Soft-threshold the residual at a measured position by this much
        rather than discarding it, which is the application's ``-o``.
    scaling : float, optional
        The data scaling to divide by; estimated when it is not given, and
        the answer is put back into the data's units either way.

    Returns
    -------
    torch.Tensor
        Coil k-space, in the centred convention, of ``kspace``'s shape.  The
        image is the coil combination of its inverse transform.

    Examples
    --------
    >>> samples = pocsense(kspace, maps)
    >>> image = bartorch.rss(bartorch.ifft(samples, (-1, -2, -3), unitary=True), axes=(0,))
    """
    maps = sensitivities.squeeze(1) if sensitivities.ndim > 3 else sensitivities

    shape = tuple(kspace.shape)
    image_shape = shape[1:]
    if len(shape) > 3 and shape[1] == 1:
        image_shape = shape[2:]

    scale = optim.data_scaling(kspace) if scaling is None else scaling
    measured = kspace * (1.0 / scale)
    pattern = sampling.pattern(measured)

    A = linop.CartesianSense(maps, image_shape, coil_batch=0, modulated=True)
    P = basic.Sampling(pattern.squeeze(), A.oshape)
    # The axes the transform runs along, counted from the last: every spatial
    # axis of the image that has more than one sample.
    spatial = tuple(-1 - axis for axis, size in enumerate(reversed(image_shape)) if size > 1)
    data = bartorch.fftmod(measured.reshape(A.oshape), spatial)

    sweep: list = [_Consistency(P, data, robust), _Range(A)]
    if alpha != 0.0:
        if wavelet:
            term = priors.Wavelet(
                spatial,
                alpha,
                joint_axes=(-len(A.oshape),),
                family="dau2",
                randshift=False,
            )
            sweep.append(_Sparsity(term, spatial, A.oshape))
        else:
            sweep.append(priors.L2(alpha))

    iterate = optim.POCS(sweep, maxiter=_MAXITER if maxiter is None else maxiter)
    answer = bartorch.fftmod(iterate(data), spatial, inverse=True)
    return (answer * scale).reshape(shape)

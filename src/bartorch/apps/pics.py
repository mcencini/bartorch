"""``pics`` as a pipeline: the encoding, the terms and the iteration."""

from __future__ import annotations

from collections.abc import Iterable

import torch

import bartorch
from bartorch import linop, optim
from bartorch.linop import basic
from bartorch.priors import Regularizer
from bartorch.tools import sampling

__all__ = ["pics"]

#: BART's own ``-i`` default (pics.c:82).
_MAXITER = 30

#: Which iteration ``italgo_choose`` (grecon/italgo.c) picks from the terms,
#: by the letter each term carries as its ``kind``.  A term not named here makes the
#: iteration FISTA when it is the first and ADMM when it is not.
_ALWAYS_ADMM = frozenset({"T", "G", "C", "V", "R1", "R2"})
_ALWAYS_NIHT = frozenset({"H", "N"})
#: An l2 penalty on the image leaves the choice where it was.
_KEEPS = frozenset({"Q"})

_SOLVERS = {
    "cg": optim.CG,
    "ist": optim.IST,
    "fista": optim.FISTA,
    "admm": optim.ADMM,
    "pridu": optim.PRIDU,
    "niht": optim.NIHT,
}


def _terms(regularizers: Regularizer | Iterable[Regularizer] | None) -> list[Regularizer]:
    if regularizers is None:
        return []
    if isinstance(regularizers, Regularizer):
        return [regularizers]
    return list(regularizers)


def _chosen(terms: list[Regularizer]) -> str:
    """The iteration ``pics`` selects for these terms, when none was named."""
    algorithm = "cg"
    for position, term in enumerate(terms):
        letter = term.kind
        if letter in _KEEPS:
            continue
        if letter in _ALWAYS_NIHT:
            algorithm = "niht"
        elif letter in _ALWAYS_ADMM:
            algorithm = "admm"
        else:
            algorithm = "fista" if position == 0 else "admm"
    return algorithm


def pics(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    *,
    regularizers: Regularizer | Iterable[Regularizer] | None = None,
    l2: float | None = None,
    solver: str | None = None,
    maxiter: int | None = None,
    step: float | None = None,
    admm_rho: float | None = None,
    cg_maxiter: int | None = None,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    toeplitz: bool | None = None,
    eigen_step: bool = False,
    scaling: float | None = None,
) -> torch.Tensor:
    """Parallel-imaging compressed-sensing reconstruction.

    The pipeline :func:`bartorch.tools.pics` runs, assembled here: the
    sampling pattern applied to the k-space, the modulation into the
    convention BART iterates in, the scaling estimated from what is left, and
    then an encoding from :mod:`bartorch.linop` under an iteration from
    :mod:`bartorch.optim`.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order: ``(coils, z, y, x)`` on a grid, and
        ``(coils, *encoding, shots, samples)`` off it.
    sensitivities : torch.Tensor
        Coil sensitivities, as :func:`~bartorch.tools.ecalib` produces them.
    regularizers : Regularizer or iterable of Regularizer, optional
        :mod:`bartorch.priors` terms.  Their axes index the image's shape.
    l2 : float, optional
        Plain Tikhonov weight.
    solver : {'cg', 'ist', 'fista', 'admm', 'pridu', 'niht'}, optional
        ``None`` chooses from the terms, as the application does.
    maxiter : int, optional
        Iterations; BART's default is thirty.
    step : float, optional
        Step size for the gradient iterations.
    admm_rho : float, optional
        ADMM penalty; setting it selects ADMM unless ``solver`` says otherwise.
    cg_maxiter : int, optional
        Inner conjugate-gradient steps for ADMM.
    traj : torch.Tensor, optional
        Non-Cartesian trajectory, in grid units.
    pattern : torch.Tensor, optional
        Sampling pattern or weights; on a grid it is read off ``kspace`` when
        it is not given.
    basis : torch.Tensor, optional
        Subspace basis over frames and coefficients.
    initial : torch.Tensor, optional
        An image to start the iteration from, in the units the solve works in
        -- that is, already divided by ``scaling``.  ``pics -W`` reads it the
        same way: it rescales the warm start only under ``-S``, where the
        answer is put back into the data's units at the end.
    eigen_step : bool
        Take the step size from the largest eigenvalue of the normal operator
        rather than from ``step``, estimated by thirty power iterations as
        ``pics -e`` estimates it.  The starting vector comes from BART's
        process-global generator, so this is the one setting under which two
        runs in a process do not agree to the bit.
    toeplitz : bool, optional
        ``False`` applies the encoding and its adjoint rather than the normal
        operator's convolution.
    scaling : float, optional
        The data scaling to divide by; estimated when it is not given, which
        is what makes a regularization weight transferable.

    Returns
    -------
    torch.Tensor
        The reconstructed image.

    Examples
    --------
    >>> image = pics(kspace, maps, l2=0.01, maxiter=50)
    >>> image = pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
    """
    terms = _terms(regularizers)
    if solver is None:
        solver = "admm" if admm_rho is not None else _chosen(terms)
    if solver not in _SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_SOLVERS)}, got {solver!r}")

    maps = sensitivities.squeeze(1) if sensitivities.ndim > 3 else sensitivities

    if traj is None:
        # On a grid the image is sampled where the k-space is, so its shape is
        # the k-space's without the coils.
        image_shape = tuple(kspace.shape[1:])
        if kspace.ndim > 3 and kspace.shape[1] == 1:
            image_shape = tuple(kspace.shape[2:])
        if pattern is None:
            pattern = sampling.pattern(kspace)
        measured = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
        encoding = linop.CartesianSense(maps, image_shape, coil_batch=0, modulated=True)
        A = basic.Sampling(pattern.squeeze(), encoding.oshape) @ encoding
        scale = optim.data_scaling(measured) if scaling is None else scaling
        data = (measured * (1.0 / scale)).squeeze(1)
    else:
        # Off it the k-space is shots and samples, which say nothing about the
        # grid, so the image is the shape the sensitivities are given on.
        image_shape = tuple(maps.shape[1:])
        # `toeplitz` defaults to the normal operator's convolution, which is
        # what the application uses; None means nobody asked.
        off_grid: dict[str, object] = {"traj": traj, "basis": basis}
        if toeplitz is not None:
            off_grid["toeplitz"] = toeplitz
        A = linop.NoncartesianSense(maps, image_shape, **off_grid)
        measured = kspace if pattern is None else kspace * pattern
        # Off the grid the scaling comes from the spread of the adjoint
        # reconstruction, so it is estimated over the samples in the layout
        # the application hands them in: a trailing readout axis of one.
        if scaling is None:
            scaling = optim.data_scaling(measured[..., None], A=A)
        scale = scaling
        data = measured * (1.0 / scale)

    extra: dict[str, object] = {}
    if step is not None:
        extra["step"] = step
    if admm_rho is not None:
        extra["rho"] = admm_rho
    if cg_maxiter is not None:
        extra["cg_maxiter"] = cg_maxiter
    if solver == "pridu":
        extra["sigma_tau_ratio"] = scale
    if eigen_step:
        if solver == "cg":
            raise ValueError("eigen_step scales a gradient step, which cg does not take")
        extra["eigen"] = True

    iteration = _SOLVERS[solver]
    arguments = [] if solver == "cg" else [terms]
    if solver == "cg" and l2 is not None:
        arguments = [l2]
    iterate = iteration(*arguments, maxiter=_MAXITER if maxiter is None else maxiter, **extra)
    return iterate(data, A, initial)

"""``mobafit`` as a pipeline: a signal model, and Gauss-Newton per voxel."""

from __future__ import annotations

from typing import Any

import torch

from bartorch import nlop, optim

__all__ = ["mobafit"]

#: Gauss-Newton steps.  The command's own default is five (mobafit.c:155),
#: which it can afford because its coefficients are the model's own variables
#: scaled to order one by ``--scale``.  TorchSim's parameterisation is bounded
#: instead, and a bounded variable moves slowly while the Tikhonov weight is
#: large: on a mono-exponential decay of a known T2, five steps answer 430 ms
#: for 60 and for 110 alike, ten come within three per cent, and twenty are
#: exact to the millisecond.  So the default is what converges rather than
#: what the command writes.
_ITERATIONS = 20

#: BART's own ``--liniter`` default (mobafit.c:246), which is both the inner
#: solver's iteration count and the Gauss-Newton step's ``cgiter``.
_CG_MAXITER = 50


def mobafit(
    images: torch.Tensor,
    model: nlop.SignalModel,
    *,
    iterations: int = _ITERATIONS,
    cg_maxiter: int = _CG_MAXITER,
    inner: Any = None,
    alpha: float = 1.0,
    alpha_min: float = 0.0,
    redu: float = 2.0,
    magnitude: bool = False,
    start: torch.Tensor | None = None,
    **values: Any,
) -> dict[str, torch.Tensor]:
    """Fit a signal model to reconstructed contrast images, voxel by voxel.

    The pipeline :func:`bartorch.tools.mobafit` runs, assembled here: a
    forward model from :mod:`bartorch.nlop`, the Gauss-Newton loop of
    :class:`bartorch.nlop.IRGNM` over it, and the fitted variables read back
    into their own units.

    The model is TorchSim's rather than BART's -- what a fit needs is a
    forward it can differentiate, with bounds and a starting state, which is
    what :class:`~bartorch.nlop.SignalModel` is over a TorchSim simulator --
    so this does not reproduce the command's coefficients.  It solves the same
    problem with the same method and answers in named maps.

    Parameters
    ----------
    images : torch.Tensor
        Contrast images, ``(contrasts, *voxels)``, C order: one image per
        echo, inversion time or repetition, in the order the model's
        acquisition lists them.
    model : bartorch.nlop.SignalModel
        The signal model, built on the acquisition that produced ``images``
        -- :func:`~bartorch.nlop.MultiEcho`, :func:`~bartorch.nlop.InversionRecovery`
        or :func:`~bartorch.nlop.Bloch` -- on the voxel shape of ``images``.
    iterations : int
        Gauss-Newton steps.  The command takes five over its own scaled
        coefficients; twenty are what a bounded parameterisation needs.
    cg_maxiter : int
        Conjugate-gradient steps per linearized problem.
    inner : solver, optional
        A configured solver from :mod:`bartorch.optim` for the linearized
        problem, whose regularizers then penalize the maps.  The default is
        plain conjugate gradients, which is what the command solves with.
    alpha : float
        Initial Tikhonov weight on the Gauss-Newton step.
    alpha_min : float
        What that weight decays towards.
    redu : float
        Factor the weight is divided by after each step.
    magnitude : bool
        Fit the magnitude of the model to the data rather than the signal
        itself, which is BART's ``-a``.
    start : torch.Tensor, optional
        Maps to start from, of the model's input shape.  Built from
        ``**values`` when it is not given.
    **values
        Starting values per unknown, in that property's own units, as
        :meth:`~bartorch.nlop.SignalModel.initial` takes them.

    Returns
    -------
    dict of str to torch.Tensor
        The fitted maps in their own units, by the name each unknown carries.
        A voxel whose data is zero across every contrast is left at the value
        it started from, as the command leaves such a patch at zero.

    Examples
    --------
    >>> M = nlop.MultiEcho([12.5 * (echo + 1) for echo in range(8)], (128, 128))
    >>> maps = mobafit(images, M, T2=80.0)
    >>> maps["T2"]
    """
    forward: Any = model
    if magnitude:
        forward = nlop.Abs(model.oshape) @ model

    x0 = model.initial(**values) if start is None else start

    solver = nlop.IRGNM(
        iterations=iterations,
        alpha=alpha,
        alpha_min=alpha_min,
        redu=redu,
        cg_maxiter=cg_maxiter,
        inner=optim.CG(maxiter=cg_maxiter) if inner is None else inner,
    )
    fitted = solver(images.reshape(forward.oshape), forward, x0=x0)

    # A voxel with no signal at all constrains nothing, and a Gauss-Newton
    # step on it walks wherever the bounds allow; the command skips such a
    # patch and leaves it at zero, so the fit is put back to where it started.
    empty = images.reshape(forward.oshape).abs().amax(dim=0) == 0
    if bool(empty.any()):
        fitted = torch.where(empty, x0.expand_as(fitted), fitted)

    return model.split(fitted)

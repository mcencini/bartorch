"""Reconstructions.

Where BART writes a choice as a run of flags, this takes the choice; where it
takes an array through a flag it calls a file, this takes a tensor.
"""

from __future__ import annotations

import torch

from bartorch.core.graph import dispatch
from bartorch.tools._call import curated

__all__ = ["nlinv", "pics"]

#: The solvers ``pics`` chooses between, which BART writes as five separate
#: flags into one variable.  Reading them one at a time is how the choice
#: stopped being reachable from Python at all.
SOLVERS = {
    "ist": "ist",
    "fista": "fista",
    "admm": "admm",
    "pridu": "pridu",
    "eulermaruyama": "eulermaruyama",
}


@curated("pics")
def pics(
    kspace: torch.Tensor,
    sensitivities: torch.Tensor,
    *,
    regularizers: str | list[str] | None = None,
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
    psf: torch.Tensor | None = None,
    toeplitz: bool | None = None,
    lowmem: bool = False,
    real: bool = False,
    eigen_step: bool = False,
    **extra,
) -> torch.Tensor:
    """Parallel-imaging compressed-sensing reconstruction.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order.
    sensitivities : torch.Tensor
        Coil sensitivities, as :func:`ecalib` or :func:`caldir` produce them.
    regularizers : str or list of str, optional
        BART's generalized regularization, ``<T>:A:B:C`` (``-R``), one or
        several.  ``"W:7:0:0.005"`` is wavelet regularization on the first
        three axes with weight 0.005.
    l2 : float, optional
        Plain Tikhonov weight (``-r``).
    solver : {'ist', 'fista', 'admm', 'pridu', 'eulermaruyama'}, optional
        Which solver to use.  BART spells each as its own flag; this is the
        one choice they make between them.  ``None`` leaves BART its default.
    maxiter : int, optional
        Iterations (``-i``).
    step : float, optional
        Step size (``-s``).
    admm_rho : float, optional
        ADMM penalty (``-u``); setting it selects ADMM unless ``solver`` says
        otherwise.
    cg_maxiter : int, optional
        Inner conjugate-gradient steps for ADMM (``-C``).
    traj : tensor, optional
        Non-Cartesian trajectory (``-t``), in grid units.
    pattern : tensor, optional
        Sampling pattern or weights (``-p``).
    basis : tensor, optional
        Subspace basis over frames and coefficients (``-B``).
    initial : tensor, optional
        Warm start (``-W``).
    psf : tensor, optional
        A point spread function computed elsewhere (``--psf_import``), which
        is how a normal operator built outside BART is brought in.
    toeplitz : bool, optional
        ``False`` passes ``--no-toeplitz``; ``None`` leaves BART's default.
    lowmem : bool
        Hold one set of frequencies of the point spread function at a time
        (``--lowmem``).
    real : bool
        Constrain the image to be real (``-c``).
    eigen_step : bool
        Scale the step size by the largest eigenvalue (``-e``).
    **extra
        Further BART ``pics`` flags, by name.

    Returns
    -------
    torch.Tensor
        The reconstructed image.

    Examples
    --------
    >>> image = pics(kspace, maps, l2=0.01, maxiter=50)
    >>> image = pics(kspace, maps, regularizers="W:7:0:0.005", solver="fista")
    >>> image = pics(kspace, maps, traj=trajectory, basis=subspace)
    """
    flags: dict = dict(extra)
    if regularizers is not None:
        flags["R"] = [regularizers] if isinstance(regularizers, str) else list(regularizers)
    if l2 is not None:
        flags["r"] = l2
    if solver is not None:
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {sorted(SOLVERS)}, not {solver!r}")
        flags[SOLVERS[solver]] = True
    if maxiter is not None:
        flags["i"] = maxiter
    if step is not None:
        flags["s"] = step
    if admm_rho is not None:
        flags["u"] = admm_rho
    if cg_maxiter is not None:
        flags["C"] = cg_maxiter
    for keyword, value in (
        ("t", traj),
        ("p", pattern),
        ("B", basis),
        ("W", initial),
        ("psf_import", psf),
    ):
        if value is not None:
            flags[keyword] = value
    if toeplitz is False:
        flags["no_toeplitz"] = True
    if lowmem:
        flags["lowmem"] = True
    if real:
        flags["c"] = True
    if eigen_step:
        flags["e"] = True
    return dispatch("pics", [kspace, sensitivities], None, **flags)


@curated("nlinv")
def nlinv(
    kspace: torch.Tensor,
    *,
    maxiter: int | None = None,
    maps: int | None = None,
    traj: torch.Tensor | None = None,
    pattern: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
    alpha: float | None = None,
    real: bool = False,
    normalize: bool = False,
    return_sensitivities: bool = False,
    **extra,
):
    """Nonlinear inversion: the image and the sensitivities together.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space, C order.
    maxiter : int, optional
        Gauss-Newton steps (``-i``).
    maps : int, optional
        How many sets of sensitivities to estimate (``-m``).
    traj : tensor, optional
        Non-Cartesian trajectory (``-t``).
    pattern : tensor, optional
        Sampling pattern (``-p``).
    basis : tensor, optional
        Subspace basis (``-B``).
    initial : tensor, optional
        Warm start (``-I``).
    alpha : float, optional
        Initial regularization weight (``-a``).
    real : bool
        Constrain the image to be real (``-g`` is the GPU flag; this is
        ``--real-constraint`` where BART has one, otherwise passed through).
    normalize : bool
        Normalize the sensitivities (``-N``).
    return_sensitivities : bool
        Also return the sensitivities, which BART writes as a second array.
    **extra
        Further BART ``nlinv`` flags, by name.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The image, and the sensitivities when asked for.
    """
    flags: dict = dict(extra)
    if maxiter is not None:
        flags["i"] = maxiter
    if maps is not None:
        flags["m"] = maps
    for keyword, value in (("t", traj), ("p", pattern), ("B", basis), ("I", initial)):
        if value is not None:
            flags[keyword] = value
    if alpha is not None:
        flags["a"] = alpha
    if real:
        flags["real_constraint"] = True
    if normalize:
        flags["N"] = True
    return dispatch(
        "nlinv", [kspace], None, _n_out=2 if return_sensitivities else 1, **flags
    )

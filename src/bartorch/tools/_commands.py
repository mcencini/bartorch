"""bartorch.tools._commands — Special-case CLI wrappers that override generated ones.

The auto-generated layer in :mod:`bartorch.tools._generated` provides thin
wrappers for every BART command.  This module imports that full suite and then
re-defines a small set of commands that benefit from a richer Python API.
The overrides are purely Pythonic argument-translation layers — each one calls
its auto-generated counterpart in :mod:`bartorch.tools._generated` after
mapping human-readable keyword names to the raw BART flag letters.

The following commands are overridden:

* :func:`phantom` — accepts positional ``dims`` and ergonomic ``kspace`` /
  ``ncoils`` kwargs; delegates to :func:`_generated.phantom`.
* :func:`fft` / :func:`ifft` — accept ``axes=`` (C-order indices) instead of
  a raw BART bitmask; ``inverse`` and ``unitary`` flags use full words.
* :func:`avg`, :func:`cdf97`, :func:`conv`, :func:`fftmod`, :func:`fftshift`,
  :func:`flip`, :func:`hist`, :func:`mip`, :func:`rss`, :func:`std`,
  :func:`var`, :func:`wavelet` — replace the positional ``bitmask`` argument
  with ``axes`` (C-order axis indices, negative values supported); the
  bitmask conversion is done internally.
* :func:`ecalib` — maps ``calib_size``, ``maps``, ``threshold``, … to flags,
  then calls :func:`_generated.ecalib`.
* :func:`scale` — corrects the generated signature so that ``factor`` (the
  scale factor) is treated as a positional scalar argument, not as a CFL
  tensor input.  Delegates to ``dispatch("scale", [input_], _pos=[factor])``.
* :func:`caldir` — maps ``calib_size`` → positional ``cal_size`` argument,
  then calls :func:`_generated.caldir`.
* :func:`pics` — ``R`` accepts ``list[str]`` for stacked regularisers; all
  solver parameters use full Python names; delegates to :func:`_generated.pics`.
* :func:`nlinv` — maps ``iter_`` → ``-i`` and ``nmaps`` → ``-m`` to avoid
  cryptic single-letter kwargs; delegates to :func:`_generated.nlinv`.
* :func:`moba` — maps ``model``, ``iter_``, ``inner_iter``, ``gpu`` to BART
  flags; delegates to :func:`_generated.moba`.
* :func:`nufft` — maps ``adjoint``, ``inverse``, ``image_dims``, ``l2_reg``,
  ``max_iter``, ``gpu``, ``toeplitz`` to BART flags; delegates to
  :func:`_generated.nufft`.

All other commands are re-exported unchanged.
The ``__init__.py`` imports from this module to build the public API.
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch.core.graph import dispatch
from bartorch.core.tensor import bart_op

# Module-level reference used by the override implementations below.
from bartorch.tools import _generated

# Re-export the full generated suite so that ``from _commands import *`` in
# __init__.py exposes every BART command.  The manually-defined functions that
# follow shadow the generated versions of the same name.
from bartorch.tools._generated import *  # noqa: F401,F403
from bartorch.tools._generated import __all__ as _generated_all
from bartorch.utils.flags import _axes_to_flags

__all__ = [*_generated_all, "ifft", "scale"]


# ---------------------------------------------------------------------------
# Phantom generation — positional dims + ergonomic kspace/ncoils kwargs
# ---------------------------------------------------------------------------


def phantom(
    dims: list[int],
    *,
    kspace: bool = False,
    ncoils: int | None = None,
    **extra_flags: Any,
) -> torch.Tensor:
    """Generate a numerical Shepp-Logan phantom or coil-sensitivity phantom.

    Parameters
    ----------
    dims : list of int
        Output image dimensions (C-order), e.g. ``[256, 256]`` for a 2-D
        phantom or ``[64, 64, 64]`` for a 3-D phantom.  Passed to
        ``output_dims`` of :func:`_generated.phantom`.
    kspace : bool, optional
        Return k-space instead of image-domain data (``-k``).
        Default ``False``.
    ncoils : int, optional
        Generate *ncoils* coil-sensitivity-weighted copies (``-s``).
        ``None`` → single coil (BART default).
    **extra_flags :
        Any additional BART ``phantom`` flags forwarded directly (e.g.
        ``flag_3=True`` for 3-D simulation, ``g=2`` to select a geometry).

    Returns
    -------
    torch.Tensor
        Complex64 phantom array (C-order).

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> ph = bt.phantom([256, 256])                          # 2-D Shepp-Logan
    >>> ksp = bt.phantom([256, 256], kspace=True, ncoils=8)  # 8-coil k-space
    """
    # BART phantom's -x n flag sets all spatial dimensions to n.
    # Without it BART defaults to 128.  Derive from dims unless the caller
    # already passed x= explicitly via extra_flags.
    x_val = extra_flags.pop("x", dims[-1] if dims else None)
    return _generated.phantom(
        output_dims=dims,
        k=kspace or None,
        s=ncoils,
        x=x_val,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# FFT — axes= ergonomics (C-order axis indices → BART bitmask)
# ---------------------------------------------------------------------------


def fft(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    unitary: bool = False,
    inverse: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Multidimensional FFT over C-order axis indices.

    Converts Python C-order axis indices (including negative indices) to the
    BART Fortran-order bitmask required by ``bart fft``, then delegates to
    :func:`_generated.fft`.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array (any dtype; cast to ``complex64`` automatically).
    axes : int or sequence of int
        C-order axis index or indices to transform.  Negative values are
        supported.  Examples:

        * ``axes=-1``        — transform the last (read) axis only
        * ``axes=(-1, -2)``  — transform the last two axes (typical 2-D FFT)
        * ``axes=(0, 1, 2)`` — transform the first three axes
    unitary : bool, optional
        Apply unitary (1/√N) normalisation (``-u``).  Default ``False``.
    inverse : bool, optional
        Compute inverse FFT (``-i``).  Default ``False`` (forward FFT).
    **extra_flags :
        Additional BART ``fft`` flags forwarded directly.

    Returns
    -------
    torch.Tensor
        Complex64 tensor with the same shape as *input_*.

    See Also
    --------
    ifft : Convenience alias for inverse FFT.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> ph = bt.phantom([256, 256])
    >>> kspace = bt.fft(ph, axes=(-1, -2))   # 2-D FFT over last two axes
    """
    bitmask = _axes_to_flags(axes, ndim=input_.ndim)
    return _generated.fft(
        input_,
        bitmask,
        u=unitary or None,
        i=inverse or None,
        **extra_flags,
    )


def ifft(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    unitary: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Inverse multidimensional FFT.

    Convenience alias for :func:`fft` with ``inverse=True``.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array (any dtype; cast to ``complex64`` automatically).
    axes : int or sequence of int
        C-order axis index or indices to transform.  Negative values are
        supported.
    unitary : bool, optional
        Apply unitary (1/√N) normalisation (``-u``).  Default ``False``.
    **extra_flags :
        Additional BART ``fft`` flags forwarded directly.

    Returns
    -------
    torch.Tensor
        Complex64 tensor with the same shape as *input_*.

    See Also
    --------
    fft : Forward FFT.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.fft(bt.phantom([256, 256]), axes=(-1, -2))
    >>> image  = bt.ifft(kspace, axes=(-1, -2))
    """
    return fft(input_, axes, unitary=unitary, inverse=True, **extra_flags)


# ---------------------------------------------------------------------------
# Scale — factor is a positional scalar (not a CFL tensor)
# ---------------------------------------------------------------------------


@bart_op
def scale(
    factor: float | complex,
    input_: torch.Tensor,
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor:
    """Scale array by *factor*.

    The auto-generated :func:`_generated.scale` incorrectly treats the scale
    factor as a CFL tensor input.  In BART's CLI the factor is a positional
    scalar argument::

        bart scale <factor> <input> <output>

    This override passes *factor* through ``_pos`` so it is inserted in the
    argv between the flags and the input CFL name.

    Parameters
    ----------
    factor : float or complex
        Multiplicative scale factor.  May be complex (e.g. ``1j`` for a 90°
        phase rotation).
    input_ : torch.Tensor
        Input array (any dtype; cast to ``complex64`` automatically).
    output_dims : list of int, optional
        Expected output shape; ``None`` to infer at runtime.
    **extra_flags :
        Additional BART ``scale`` flags forwarded directly.

    Returns
    -------
    torch.Tensor
        Scaled array with the same shape as *input_*, dtype ``complex64``.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> x = bt.phantom([64, 64])
    >>> y = bt.scale(2.0, x)          # double the magnitude
    >>> z = bt.scale(1j,  x)          # 90° phase rotation
    """
    return dispatch("scale", [input_], output_dims, _pos=[factor], **extra_flags)


# ---------------------------------------------------------------------------
# ESPIRiT calibration — Pythonic keyword → BART flag mapping
# ---------------------------------------------------------------------------


def ecalib(
    kspace: torch.Tensor,
    *,
    calib_size: int | tuple[int, int, int] | None = None,
    maps: int | None = None,
    threshold: float | None = None,
    crop: float | None = None,
    smooth: bool = False,
    intensity: bool = False,
    weighting: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using ESPIRiT.

    ESPIRiT computes sensitivity maps directly from the auto-calibration signal
    (ACS) region of k-space by eigen-decomposition of a calibration matrix.

    Parameters
    ----------
    kspace : torch.Tensor
        Fully-sampled calibration k-space or k-space containing an ACS region.
        Expected shape (C-order): ``(ncoils[, nz], ny, nx)``.
    calib_size : int or tuple of int, optional
        Size of the calibration region (``-r``).  A single int is applied
        uniformly to all dimensions; a 3-tuple sets ``(read, phase1, phase2)``
        (i.e. ``(x, y, z)`` in BART convention) independently.
        ``None`` → BART auto-detects.
    maps : int, optional
        Number of ESPIRiT maps to compute (``-m``).  ``None`` → BART default
        (``1``).  Use ``2`` to handle phase singularities.
    threshold : float, optional
        Singular-value threshold for the calibration matrix (``-t``).
        ``None`` → BART default.
    crop : float, optional
        Crop sensitivity maps below this image-domain threshold (``-c``).
        ``None`` → BART default.
    smooth : bool, optional
        Create maps with smooth transitions using Soft-SENSE (``-S``).
        Default ``False``.
    intensity : bool, optional
        Intensity-correction (``-I``).  Default ``False``.
    weighting : bool, optional
        Apply soft-weighting of the singular vectors (``-W``).
        Default ``False``.
    **extra_flags :
        Any additional BART ``ecalib`` flags passed directly (e.g.
        ``v=0.001`` for noise variance).

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps (C-order).

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], s=8)
    >>> sens = bt.ecalib(kspace, calib_size=24)
    """
    # ecalib output is always 5-dimensional in BART Fortran order:
    # (nx, ny, nz=1, nc, maps) → C-order: (maps, nc, nz, ny, nx).
    # Passing the expected C-order shape as output_dims ensures that the
    # maps dimension is preserved even when maps=1 (which would otherwise
    # be trimmed as a trailing 1 by run()'s default logic).
    # Spatial dims: last (ndim_in-1) dims of kspace in C-order
    spatial = list(kspace.shape[1:])  # (nz?, ny, nx)
    # Pad spatial to 3 dims (nz, ny, nx) with leading 1s
    while len(spatial) < 3:
        spatial.insert(0, 1)
    nz, ny, nx = spatial[-3], spatial[-2], spatial[-1]
    nc = kspace.shape[0]
    nmaps = maps if maps is not None else 1
    # C-order output shape: (maps, nc, nz, ny, nx)
    _output_dims = [nmaps, nc, nz, ny, nx]

    return _generated.ecalib(
        kspace,
        output_dims=_output_dims,
        r=calib_size,
        m=maps,
        t=threshold,
        c=crop,
        S=smooth or None,
        I=intensity or None,
        W=weighting or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# Direct calibration
# ---------------------------------------------------------------------------


def caldir(
    kspace: torch.Tensor,
    *,
    calib_size: int,
    **extra_flags: Any,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using direct calibration (CALDIR).

    A simpler, faster alternative to :func:`ecalib` that computes sensitivity
    maps by direct Fourier-space operations on the ACS region.

    Parameters
    ----------
    kspace : torch.Tensor
        Calibration k-space (C-order).
    calib_size : int
        Size of the calibration region (positional BART argument ``cal_size``).
    **extra_flags :
        Additional BART ``caldir`` flags.

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], s=8)
    >>> sens = bt.caldir(kspace, calib_size=24)
    """
    return _generated.caldir(kspace, calib_size, **extra_flags)


# ---------------------------------------------------------------------------
# PICS regularisation guide
# ---------------------------------------------------------------------------

_R_GUIDE = """\
The ``-R`` flag selects a regulariser and is the primary way to configure the
compressed-sensing penalty in PICS.  It uses the syntax::

    "<type>:<transform_flags>:<joint_dims>:<lambda>"

**type** selects the penalty:

* ``W`` — Wavelet (ℓ₁-Wavelet)
* ``T`` — Total Variation (TV)
* ``L`` — Locally Low-Rank (LLR)
* ``B`` — Block-wise Low-Rank
* ``N`` — Nuclear-norm Low-Rank

**transform_flags** is a BART bitmask of the axes to which the transform
is applied.  The common values below use Fortran-order BART bitmasks directly;
for a C-order Python axis index, the equivalent bitmask is
``1 << (ndim - 1 - axis)``.

Common values for a ``(coils, ny, nx)`` k-space (3-D, C-order):

* ``7``  — all three spatial axes (read + phase1 + phase2 in BART)
* ``3``  — last two axes only (phase1 + read)
* ``4``  — first axis only (coils / z)

**joint_dims** selects which additional dimensions are processed jointly
(e.g. for temporal or multi-contrast data); ``0`` = no joint processing.

**lambda** is the regularisation strength (float > 0; larger = more
regularisation).

Multiple regularisers can be stacked by passing a list to ``R``.

Examples
--------
Single wavelet::

    pics(kspace, sens, R="W:7:0:0.005")

TV + wavelet::

    pics(kspace, sens, R=["T:7:0:0.002", "W:7:0:0.005"])
"""


# ---------------------------------------------------------------------------
# Torch-prior helper — BART img_dims from kspace tensor
# ---------------------------------------------------------------------------

_BART_DIMS = 16  # BART's DIMS constant
_COIL_DIM = 3  # BART's COIL_DIM (Fortran index of the coil dimension)


# ---------------------------------------------------------------------------
# PICS reconstruction
# ---------------------------------------------------------------------------


def pics(
    kspace: torch.Tensor,
    sens: torch.Tensor,
    *,
    # Regularisation
    lambda_: float | None = None,
    R: list[str] | str | None = None,
    # Solver
    iter_: int | None = None,
    step: float | None = None,
    admm_rho: float | None = None,
    cg_iter: int | None = None,
    # Output
    real: bool = False,
    gpu: bool = False,
    # Advanced
    subspace_basis: str | None = None,
    init: str | None = None,
    fast_est: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    (
        """Parallel Imaging Compressed Sensing (PICS) reconstruction.

    Iteratively reconstructs an image from under-sampled k-space data using
    sensitivity encoding and compressed-sensing regularisation.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space data (C-order).  Zero-padded entries indicate
        missing samples.
    sens : torch.Tensor
        Coil sensitivity maps, typically from :func:`ecalib` or
        :func:`caldir`.

    Regularisation
    --------------
    lambda_ : float, optional
        Regularisation strength λ (``-r``).
    R : str or list of str, optional
        BART regularisation specification(s) (``-R``).

"""
        + _R_GUIDE
        + """
    Solver
    ------
    iter_ : int, optional
        Maximum number of solver iterations (``-i``).  ``None`` → BART
        default.
    step : float, optional
        Iteration step size (``-s``).
    admm_rho : float, optional
        ADMM penalty parameter ρ (``-u``).  Setting this enables the ADMM
        solver; ``None`` uses the default IST/FISTA solver.
    cg_iter : int, optional
        Maximum inner CG iterations (ADMM only) (``-C``).

    Output
    ------
    real : bool, optional
        Real-value constraint: cast result to real-valued image (``-c``).
        Default ``False``.
    gpu : bool, optional
        Use GPU via BART's internal CUDA support (``-g``).  Requires BART
        compiled with ``USE_CUDA=ON``.  Default ``False``.

    Advanced
    --------
    subspace_basis : str, optional
        Path to a CFL subspace basis (``-B``).
    init : str, optional
        Warm-start initialisation CFL (``-W``).
    fast_est : bool, optional
        Scale step size based on max. eigenvalue (``-e``).  Default ``False``.
    **extra_flags :
        Any additional BART ``pics`` flags not listed above, passed directly.
        For example ``R="W:7:0:0.005"`` (via ``R=`` parameter above),
        ``p="mask.cfl"`` → ``-p mask.cfl``, ``N=True`` → ``-N``.

    Returns
    -------
    torch.Tensor
        Reconstructed complex image (C-order).

    Examples
    --------
    Basic PICS with wavelet regularisation:

    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], s=8)
    >>> sens   = bt.ecalib(kspace, calib_size=24)
    >>> reco   = bt.pics(kspace, sens, R="W:7:0:0.005")

    Multiple regularisers (TV + wavelet):

    >>> reco = bt.pics(kspace, sens, R=["T:7:0:0.002", "W:7:0:0.005"])

    ADMM solver with TV:

    >>> reco = bt.pics(kspace, sens, R="T:7:0:0.01", admm_rho=0.01)

    """
    )
    return _generated.pics(
        kspace,
        sens,
        r=lambda_,
        R=R,
        i=iter_,
        s=step,
        u=admm_rho,
        C=cg_iter,
        c=real or None,
        g=gpu or None,
        B=subspace_basis,
        W=init,
        e=fast_est or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# Nonlinear inversion (nlinv)
# ---------------------------------------------------------------------------


def nlinv(
    kspace: torch.Tensor,
    *,
    output_dims: list[int] | None = None,
    iter_: int | None = None,
    nmaps: int | None = None,
    gpu: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Jointly estimate image and coil sensitivities via nonlinear inversion.

    Iteratively solves the nonlinear inverse problem of joint image and
    sensitivity estimation (Newton-CG / IRGNM).

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space data (C-order).
    output_dims : list[int], optional
        Expected output shape; ``None`` to infer at runtime.
    iter_ : int, optional
        Number of Newton steps (``-i``).  ``None`` → BART default.
    nmaps : int, optional
        Number of ENLIVE maps to reconstruct (``-m``).  ``None`` → BART
        default (``1``).
    gpu : bool, optional
        Use GPU via BART's internal CUDA support (``-g``).  Default ``False``.
    **extra_flags :
        Any additional BART ``nlinv`` flags forwarded directly (e.g.
        ``c=True`` for real-value constraint, ``N=True`` to skip
        normalisation, ``alpha=1e-3`` for regularisation).

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Reconstructed image (and optionally sensitivity maps).

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], s=8)
    >>> image  = bt.nlinv(kspace, iter_=8)
    """
    return _generated.nlinv(
        kspace,
        output_dims=output_dims,
        i=iter_,
        m=nmaps,
        g=gpu or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# Model-based reconstruction (moba)
# ---------------------------------------------------------------------------


def moba(
    kspace: torch.Tensor,
    TI_per_TE: torch.Tensor,
    *,
    output_dims: list[int] | None = None,
    model: int | None = None,
    iter_: int | None = None,
    inner_iter: int | None = None,
    min_reg: float | None = None,
    gpu: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Model-based quantitative MRI reconstruction (MOBA).

    Reconstructs quantitative parameter maps (e.g. T1, T2, fat/water) directly
    from multi-contrast k-space data by fitting a forward signal model via
    Gauss-Newton iteration.

    Parameters
    ----------
    kspace : torch.Tensor
        Multi-contrast k-space data (C-order).
    TI_per_TE : torch.Tensor
        Timing array: inversion times (TI) for T1 models, echo times (TE)
        for T2/T2* models.  The correct interpretation depends on ``model``.
    output_dims : list[int], optional
        Expected output shape; ``None`` to infer at runtime.
    model : int, optional
        Signal model to fit (``-m``).  Selected from BART's MGRE model enum:

        ===  ==========  =============================================
        ID   Name        Description
        ===  ==========  =============================================
        0    WF          Water-Fat (Dixon)
        1    WFR2S       Water-Fat + R2* (default)
        2    WF2R2S      Water-Fat with two R2* components
        3    R2S         R2* only
        4    PHASEDIFF   Phase-difference fat-water
        ===  ==========  =============================================

        Run ``bart moba -h`` for the complete list.  ``None`` → BART
        default (``WFR2S = 1``).
    iter_ : int, optional
        Number of Gauss-Newton (outer) iterations (``-i``).  ``None`` →
        BART default.
    inner_iter : int, optional
        Number of inner CG iterations per Gauss-Newton step (``-C``).
        ``None`` → BART default.
    min_reg : float, optional
        Minimum regularisation parameter (``-j``).  ``None`` → BART
        default.
    gpu : bool, optional
        Use GPU via BART's internal CUDA support (``-g``).  Default ``False``.
    **extra_flags :
        Any additional BART ``moba`` flags forwarded directly (e.g.
        ``l=1`` to toggle l1-wavelet regularisation, ``N=True`` to
        normalise, ``s=0.95`` to set the step size).

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Quantitative parameter maps (C-order).

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> # WFR2S fat-water separation
    >>> maps = bt.moba(kspace, echo_times, model=1, iter_=10)
    """
    return _generated.moba(
        kspace,
        TI_per_TE,
        output_dims=output_dims,
        m=model,
        i=iter_,
        C=inner_iter,
        j=min_reg,
        g=gpu or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# Non-uniform FFT (nufft)
# ---------------------------------------------------------------------------


def nufft(
    traj: torch.Tensor,
    kspace: torch.Tensor,
    *,
    output_dims: list[int] | None = None,
    adjoint: bool = False,
    inverse: bool = False,
    image_dims: tuple[int, int, int] | None = None,
    l2_reg: float | None = None,
    max_iter: int | None = None,
    toeplitz: bool = False,
    gpu: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Non-uniform Fast Fourier Transform (NUFFT).

    Performs forward, adjoint, or iterative-inverse NUFFT using BART's
    ``nufft`` command.

    Parameters
    ----------
    traj : torch.Tensor
        Non-Cartesian k-space trajectory (C-order).
    kspace : torch.Tensor
        K-space data (forward) or image data (adjoint/inverse) (C-order).
    output_dims : list[int], optional
        Expected output shape; ``None`` to infer at runtime.
    adjoint : bool, optional
        Compute the adjoint NUFFT (k-space → image) (``-a``).
        Default ``False`` (forward: image → k-space).
    inverse : bool, optional
        Compute the iterative inverse NUFFT via CG (``-i``).
        Default ``False``.
    image_dims : tuple of (int, int, int), optional
        Image dimensions ``x:y:z`` for adjoint/inverse NUFFT (``-x``).
        ``None`` → infer from trajectory.
    l2_reg : float, optional
        L2 regularisation strength for the iterative inverse (``-l``).
        ``None`` → no regularisation.
    max_iter : int, optional
        Maximum number of CG iterations for the iterative inverse (``-m``).
        ``None`` → BART default.
    toeplitz : bool, optional
        Use Toeplitz embedding for the normal operator (``-t``).
        Default ``False``.
    gpu : bool, optional
        Use GPU via BART's internal CUDA support (``-g``).  Requires BART
        compiled with ``USE_CUDA=ON``.  Default ``False``.
    **extra_flags :
        Any additional BART ``nufft`` flags forwarded directly.

    Returns
    -------
    torch.Tensor
        NUFFT result (C-order).

    Examples
    --------
    Adjoint NUFFT (gridding):

    >>> import bartorch.tools as bt
    >>> image = bt.nufft(traj, kspace, adjoint=True, image_dims=(256, 256, 1))

    Iterative inverse NUFFT:

    >>> image = bt.nufft(traj, kspace, inverse=True, l2_reg=1e-3, max_iter=50)
    """
    return _generated.nufft(
        traj,
        kspace,
        output_dims=output_dims,
        a=adjoint or None,
        i=inverse or None,
        x=image_dims,
        l=l2_reg,
        m=max_iter,
        t=toeplitz or None,
        g=gpu or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# reshape — pass per-dimension sizes as positional BART args
#
# BART CLI: ``bart reshape flags d1 d2 ... dN input output``
# The generated wrapper only passes ``flags``; the new sizes per selected
# dimension must follow as positional args.  Sizes are derived from
# ``output_dims`` (C-order) and reversed to BART Fortran order.
# ---------------------------------------------------------------------------


def reshape(
    input_: torch.Tensor,
    flags: int,
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor:
    """Reshape selected dimensions.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    flags : int
        Bitmask selecting the dimensions to reshape.
    output_dims : list[int]
        New shape in C-order.  The sizes are automatically reversed to Fortran
        order when passed to BART so that the number of positional size
        arguments matches the number of bits set in ``flags``.
    **extra_flags :
        Additional BART ``reshape`` flags forwarded directly.
    """
    if output_dims is not None:
        pos: list[Any] = [flags, *reversed(output_dims)]
    else:
        pos = [flags]
    return dispatch("reshape", [input_], output_dims, _pos=pos, **extra_flags)


# ---------------------------------------------------------------------------
# ones / zeros — pass dimension sizes as positional BART args
#
# BART CLI: ``bart ones D d1 d2 ... dD output``
#                        ``bart zeros D d1 d2 ... dD output``
# The generated wrappers only pass ``D``; the per-dimension sizes are
# derived from ``output_dims`` (C-order) and forwarded in reversed (Fortran)
# order so BART produces the expected shape.
# ---------------------------------------------------------------------------


def ones(
    dims: int,
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor:
    """Create an array filled with ones.

    Parameters
    ----------
    dims : int
        Number of dimensions (``D`` in BART ``ones D d1…dD``).
    output_dims : list[int]
        Shape of the output array in C-order.  The sizes are automatically
        converted to Fortran order when passed to BART.
    **extra_flags :
        Additional BART ``ones`` flags forwarded directly.
    """
    if output_dims is not None:
        pos: list[Any] = [dims, *reversed(output_dims)]
    else:
        pos = [dims]
    return dispatch("ones", [], output_dims, _pos=pos, **extra_flags)


def zeros(
    dims: int,
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor:
    """Create a zero-filled array.

    Parameters
    ----------
    dims : int
        Number of dimensions (``D`` in BART ``zeros D d1…dD``).
    output_dims : list[int]
        Shape of the output array in C-order.  The sizes are automatically
        converted to Fortran order when passed to BART.
    **extra_flags :
        Additional BART ``zeros`` flags forwarded directly.
    """
    if output_dims is not None:
        pos: list[Any] = [dims, *reversed(output_dims)]
    else:
        pos = [dims]
    return dispatch("zeros", [], output_dims, _pos=pos, **extra_flags)


# ---------------------------------------------------------------------------
# Bitmask → axes overrides
#
# Every BART command whose first positional scalar argument is a bitmask
# (selecting dimensions by set-bits) is wrapped here so that callers pass
# C-order *axis indices* instead.  The conversion to a BART bitmask is done
# internally via ``_axes_to_flags(axes, ndim=input_.ndim)``.
#
# Commands covered: avg, cdf97, conv, fftmod, fftshift, flip, hist, mip,
#                   rss, std, var, wavelet
# ---------------------------------------------------------------------------


def avg(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    w: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Calculate (weighted) average along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices to average over.  Negative values
        are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    w : bool, optional
        Weighted averaging (``-w``).  Default ``False``.
    **extra_flags :
        Additional BART ``avg`` flags forwarded directly.
    """
    return _generated.avg(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        w=w or None,
        **extra_flags,
    )


def cdf97(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    i: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Perform a CDF 9/7 wavelet transform along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    i : bool, optional
        Inverse transform (``-i``).  Default ``False``.
    **extra_flags :
        Additional BART ``cdf97`` flags forwarded directly.
    """
    return _generated.cdf97(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        i=i or None,
        **extra_flags,
    )


def conv(
    input_: torch.Tensor,
    kernel: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Convolve *input_* with *kernel* along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    kernel : torch.Tensor
        Convolution kernel.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
        The ndim of *input_* is used for the axis-to-bitmask conversion.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    **extra_flags :
        Additional BART ``conv`` flags forwarded directly.
    """
    return _generated.conv(
        input_,
        kernel,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        **extra_flags,
    )


def fftmod(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    b: bool = False,
    i: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Apply ±1 modulation along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    b : bool, optional
        Apply modulation to both halves (``-b``).  Default ``False``.
    i : bool, optional
        Inverse modulation (``-i``).  Default ``False``.
    **extra_flags :
        Additional BART ``fftmod`` flags forwarded directly.
    """
    return _generated.fftmod(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        b=b or None,
        i=i or None,
        **extra_flags,
    )


def fftshift(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    b: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Apply FFT shift along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    b : bool, optional
        Apply to both halves (``-b``).  Default ``False``.
    **extra_flags :
        Additional BART ``fftshift`` flags forwarded directly.
    """
    return _generated.fftshift(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        b=b or None,
        **extra_flags,
    )


def flip(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Flip (reverse) array along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices to flip.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    **extra_flags :
        Additional BART ``flip`` flags forwarded directly.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> img = bt.phantom([64, 64])
    >>> flipped = bt.flip(img, axes=-1)   # reverse last (read) axis
    """
    return _generated.flip(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        **extra_flags,
    )


def hist(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    c: bool = False,
    s: int | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute histogram along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    c : bool, optional
        Cumulative histogram (``-c``).  Default ``False``.
    s : int, optional
        Number of histogram bins (``-s``).  ``None`` → BART default.
    **extra_flags :
        Additional BART ``hist`` flags forwarded directly.
    """
    return _generated.hist(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        c=c or None,
        s=s,
        **extra_flags,
    )


def mip(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    m: bool = False,
    a: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Maximum (or minimum) intensity projection along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    m : bool, optional
        *Minimum* intensity projection instead of maximum (``-m``).
        Default ``False``.
    a : bool, optional
        Absolute value before projection (``-a``).  Default ``False``.
    **extra_flags :
        Additional BART ``mip`` flags forwarded directly.
    """
    return _generated.mip(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        m=m or None,
        a=a or None,
        **extra_flags,
    )


def rss(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Root-sum-of-squares combination along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array (typically coil images).
    axes : int or sequence of int
        C-order axis index or indices over which to compute RSS.
        Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    **extra_flags :
        Additional BART ``rss`` flags forwarded directly.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> coil_imgs = bt.phantom(s=4, x=64)       # 4-coil phantom
    >>> combined  = bt.rss(coil_imgs, axes=0)   # RSS over first (coil) axis
    """
    return _generated.rss(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        **extra_flags,
    )


def std(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute standard deviation along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    **extra_flags :
        Additional BART ``std`` flags forwarded directly.
    """
    return _generated.std(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        **extra_flags,
    )


def var(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute variance along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
    **extra_flags :
        Additional BART ``var`` flags forwarded directly.
    """
    return _generated.var(
        input_,
        _axes_to_flags(axes, ndim=input_.ndim),
        output_dims=output_dims,
        **extra_flags,
    )


def wavelet(
    input_: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    output_dims: list[int] | None = None,
    a: bool = False,
    **extra_flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Perform a wavelet transform along C-order axes.

    Parameters
    ----------
    input_ : torch.Tensor
        Input array.
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported.
    output_dims : list[int], optional
        Expected output shape (C-order).  ``None`` → inferred at runtime.
        **Required when** ``a=True``: BART adjoint wavelet needs the original
        image dimensions as positional CLI args
        (``wavelet -a flags d1 d2 … input output``).
    a : bool, optional
        Adjoint / inverse transform (``-a``).  Default ``False``.
    **extra_flags :
        Additional BART ``wavelet`` flags forwarded directly.
    """
    if a and output_dims is not None:
        # Adjoint wavelet: BART CLI = ``wavelet -a flags d1 d2 … input output``
        # BART's forward wavelet collapses all flagged spatial dims into a single
        # 1D coefficient array (wavelet_coeffs2 sets odims[first_bit] = total,
        # odims[other_bits] = 1).  Therefore input_.ndim == 1 here, and using it
        # to compute the bitmask would raise "axis -2 out of range for ndim=1".
        # Use len(output_dims) instead — the output (reconstructed image) ndim.
        bitmask = _axes_to_flags(axes, ndim=len(output_dims))
        # The image dims (C-order output_dims reversed to BART Fortran order)
        # must follow the bitmask as positional scalar args.
        # Call dispatch directly (not _generated.wavelet) so we can pass the
        # extra dimension positional args. The C++ layer handles normalization.
        return dispatch(
            "wavelet",
            [input_],
            output_dims,
            _pos=[bitmask, *reversed(output_dims)],
            a=True,
            **extra_flags,
        )
    # Forward path: bitmask from input shape.
    bitmask = _axes_to_flags(axes, ndim=input_.ndim)
    # Delegate to the @bart_op-decorated generated wrapper.
    return _generated.wavelet(
        input_,
        bitmask,
        output_dims=output_dims,
        a=a or None,
        **extra_flags,
    )

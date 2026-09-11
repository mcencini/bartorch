"""Coil calibration: sensitivities, compression, whitening and noise estimates."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["caldir", "ecalib"]


@curated("ecalib")
def ecalib(
    kspace: torch.Tensor,
    *,
    maps: int | None = None,
    calib_size: int | tuple[int, ...] | None = None,
    threshold: float | None = None,
    crop: float | None = None,
    kernel_size: int | None = None,
    softsense: bool = False,
    intensity_correction: bool = False,
    return_eigenvalues: bool = False,
    **extra,
):
    """Coil sensitivities by ESPIRiT.

    Parameters
    ----------
    kspace : torch.Tensor
        Fully sampled calibration data, or k-space with a sampled centre.
    maps : int, optional
        How many sets of sensitivities to produce (``-m``).
    calib_size : int or tuple of int, optional
        The calibration region's size (``-r``), the same on every axis or one
        per axis.
    threshold : float, optional
        The singular-value threshold for the calibration matrix (``-t``).
    crop : float, optional
        The eigenvalue below which a sensitivity is set to zero (``-c``).
    kernel_size : int, optional
        The calibration kernel's size (``-k``).
    softsense : bool
        Return the maps without the eigenvalue crop, for soft-SENSE (``-S``).
    intensity_correction : bool
        Correct for intensity rather than normalising (``-I``).
    return_eigenvalues : bool
        Also return the eigenvalue map, which BART writes as a second array
        only when asked.
    **extra
        Further BART ``ecalib`` options, by name.  ``e``, the axis the second
        step is split along, takes an axis of ``kspace``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        The sensitivities, and the eigenvalues when asked for.

    Examples
    --------
    >>> maps = ecalib(kspace, maps=1, crop=0.8)
    """
    flags: dict = _call.translate("ecalib", dict(extra), [kspace])
    if maps is not None:
        flags["m"] = maps
    if calib_size is not None:
        flags["r"] = (
            calib_size
            if isinstance(calib_size, int)
            else tuple(int(n) for n in reversed(tuple(calib_size)))
        )
    if threshold is not None:
        flags["t"] = threshold
    if crop is not None:
        flags["c"] = crop
    if kernel_size is not None:
        flags["k"] = kernel_size
    if softsense:
        flags["S"] = True
    if intensity_correction:
        flags["I"] = True
    return dispatch("ecalib", [kspace], None, _n_out=2 if return_eigenvalues else 1, **flags)


@curated("caldir")
def caldir(kspace: torch.Tensor, calib_size: int, **extra) -> torch.Tensor:
    """Coil sensitivities from the centre of k-space directly.

    Parameters
    ----------
    kspace : torch.Tensor
        k-space with a sampled centre.
    calib_size : int
        The size of the calibration region to use.
    **extra
        Further BART ``caldir`` flags, by name.

    Returns
    -------
    torch.Tensor
        The sensitivities.
    """
    return dispatch("caldir", [kspace], None, _pos=[int(calib_size)], **extra)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = (
    "calmat",
    "cc",
    "ccapply",
    "ecaltwo",
    "estscaling",
    "estvar",
    "ncalib",
    "phasepole",
    "rovir",
    "walsh",
    "whiten",
)

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]

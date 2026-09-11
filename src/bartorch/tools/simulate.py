"""Phantoms and signal models."""

from __future__ import annotations

import torch

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["phantom"]


@curated("phantom")
def phantom(
    shape: int | tuple[int, ...] | None = None,
    *,
    kspace: bool = False,
    coils: int | None = None,
    traj: torch.Tensor | None = None,
    geometry: str | None = None,
    rotation_angle: float | None = None,
    rotation_steps: int | None = None,
    **extra,
) -> torch.Tensor:
    """An analytical phantom, as an image or as its k-space.

    Parameters
    ----------
    shape : int or tuple of int, optional
        The image's spatial size (``-x``), square unless a tuple is given.
        Without one, BART's own default size.
    kspace : bool
        Return the analytical k-space rather than the image (``-k``).
    coils : int, optional
        Simulate this many coil sensitivities (``-s``).
    traj : tensor, optional
        Sample the k-space along this trajectory (``-t``) rather than on a
        grid, which implies ``kspace``.
    geometry : {'circle', 'brain', 'nist', 'sonar', 'tubes', 'bart'}, optional
        Which phantom to make; BART spells each as its own flag.
    rotation_angle, rotation_steps : float, int, optional
        Rotate the phantom (``--rotation-angle``, ``--rotation-steps``).
    **extra
        Further BART ``phantom`` flags, by name.

    Returns
    -------
    torch.Tensor
        The phantom.

    Examples
    --------
    >>> image = phantom(128)
    >>> kspace = phantom(128, coils=8, kspace=True)
    """
    geometries = {
        "circle": "c",
        "brain": "BRAIN",
        "nist": "NIST",
        "sonar": "SONAR",
        "tubes": "T",
        "bart": "B",
    }
    flags: dict = dict(extra)
    if shape is not None:
        if isinstance(shape, int):
            flags["x"] = shape
        else:
            spatial = tuple(shape)
            if len(set(spatial)) != 1:
                raise ValueError(
                    "BART's phantom is square or cubic; give one size, or resize after"
                )
            flags["x"] = spatial[0]
            if len(spatial) == 3:
                flags["flag_3"] = True
    if kspace:
        flags["k"] = True
    if coils is not None:
        flags["s"] = coils
    if traj is not None:
        flags["t"] = traj
    if geometry is not None:
        if geometry not in geometries:
            raise ValueError(f"geometry must be one of {sorted(geometries)}, not {geometry!r}")
        flags[geometries[geometry]] = True
    if rotation_angle is not None:
        flags["rotation_angle"] = rotation_angle
    if rotation_steps is not None:
        flags["rotation_steps"] = rotation_steps
    return dispatch("phantom", [], None, **flags)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = ("bloch", "coils", "epg", "fakeksp", "mobasig", "noise", "pulse", "seq", "signal", "sim")

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]

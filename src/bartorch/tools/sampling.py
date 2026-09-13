"""Trajectories and sampling patterns."""

from __future__ import annotations

from bartorch import _call
from bartorch._call import curated
from bartorch._dispatch import dispatch

__all__ = ["traj"]


@curated("traj")
def traj(
    *,
    readout: int | None = None,
    spokes: int | None = None,
    frames: int | None = None,
    radial: bool = False,
    golden: bool = False,
    turns: int | None = None,
    oversampling: bool = False,
    **extra,
):
    """A k-space trajectory, in grid units.

    Parameters
    ----------
    readout : int, optional
        Samples along a readout (``-x``).
    spokes : int, optional
        Readouts per frame (``-y``).
    frames : int, optional
        Turns of a multi-shot trajectory (``-t``, BART's ``conf.turns``).
        The name here does not describe the flag.
    radial : bool
        A radial trajectory rather than Cartesian (``-r``).
    golden : bool
        Golden-angle spacing between spokes (``-G``).
    turns : int, optional
        SMS multiband factor (``-m``).  The name here does not describe the
        flag either.
    oversampling : bool
        Correct the transverse gradient error of a radial trajectory
        (``-O``).  Despite the name this is not readout oversampling, which
        is ``traj -o`` and reachable through ``**extra`` as ``o=factor``.
    **extra
        Further BART ``traj`` flags, by name.

    Returns
    -------
    torch.Tensor
        Coordinates of shape ``(..., samples, 3)``, which every non-Cartesian
        transform in this package takes.

    Examples
    --------
    >>> trajectory = traj(readout=256, spokes=64, radial=True, golden=True)
    """
    flags: dict = dict(extra)
    for keyword, value in (("x", readout), ("y", spokes), ("t", frames), ("m", turns)):
        if value is not None:
            flags[keyword] = value
    if radial:
        flags["r"] = True
    if golden:
        flags["G"] = True
    if oversampling:
        flags["O"] = True
    return dispatch("traj", [], None, **flags)


#: Commands in this section without a hand-written wrapper, built from the catalogue.
_DERIVED = (
    "bin",
    "estdelay",
    "estdims",
    "grid",
    "nufftbase",
    "pattern",
    "poisson",
    "psf",
    "raga",
    "rmfreq",
    "ssa",
    "trajcor",
    "upat",
    "wavepsf",
)

for _name in _DERIVED:
    globals()[_name] = _call.build(_name, __name__)
del _name

__all__ = [*__all__, *_DERIVED]

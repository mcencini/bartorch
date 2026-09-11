"""Trajectories and sampling patterns."""

from __future__ import annotations

from bartorch.core.graph import dispatch
from bartorch.tools._call import curated

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
        Frames (``-t``).
    radial : bool
        A radial trajectory rather than Cartesian (``-r``).
    golden : bool
        Golden-angle spacing between spokes (``-G``).
    turns : int, optional
        Turns, for a multi-shot trajectory (``-m``).
    oversampling : bool
        Two-fold readout oversampling (``-o`` takes a factor; this is ``-O``).
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

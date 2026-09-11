"""Thresholding."""

from __future__ import annotations

import torch

from bartorch._call import curated
from bartorch._dispatch import dispatch
from bartorch._operator import axes_flags

__all__ = ["hard_thresh", "soft_thresh"]


@curated("threshold")
def soft_thresh(
    lamda: float, input: torch.Tensor, *, joint_axes: int | tuple[int, ...] = ()
) -> torch.Tensor:
    """Soft thresholding, ``x * max(1 - lamda / |x|, 0)``.

    Parameters
    ----------
    lamda : float
        Threshold, in the units of ``input``'s magnitude.
    input : torch.Tensor
    joint_axes : int or tuple of int
        Axes over which ``|x|`` is the Euclidean norm, so that each fibre
        along them is shrunk as one vector (``-j``); empty for element-wise.

    Returns
    -------
    torch.Tensor
        The shape of ``input``.
    """
    joint = joint_axes if isinstance(joint_axes, int) else tuple(joint_axes)
    flags = axes_flags(joint, input.ndim) if joint != () else None
    return dispatch("threshold", [input], None, _pos=[float(lamda)], j=flags).reshape(input.shape)


@curated("threshold")
def hard_thresh(lamda: float, input: torch.Tensor) -> torch.Tensor:
    """Hard thresholding: keep ``x`` where ``|x| > lamda``, zero elsewhere (``-H``).

    Returns
    -------
    torch.Tensor
        The shape of ``input``.
    """
    return dispatch("threshold", [input], None, _pos=[float(lamda)], H=True).reshape(input.shape)

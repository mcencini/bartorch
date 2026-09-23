"""Real-valued image networks applied to the complex images of a reconstruction."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Denoiser"]

#: Lower bound on a modulus, so that an all-zero image is not divided by zero.
_TINY = 1e-12

_PARTS = ("channels", "separate", "magnitude")


class Denoiser(nn.Module):
    """Adapter applying a real-valued image network to a complex image.

    Image-restoration networks operate on real tensors of shape
    ``(n, channels, *spatial)`` whose values are of order unity.  An image
    here is complex, may carry frames, contrasts or subspace coefficients in
    front of its spatial axes, may be volumetric, and is scaled by the
    normalization applied to the measured data.  This module performs the four
    conversions between the two representations:

    1. the last ``spatial`` axes form the image seen by the network, the
       leading axis is a batch, and the axes between them are folded into the
       network's batch axis;
    2. the complex values are laid out as real planes according to ``parts``;
    3. a single plane is replicated across three channels, and the three
       returned channels are averaged, where the network is an RGB one;
    4. each item of the leading axis is divided by its own peak modulus before
       the call and multiplied by it afterwards.

    Instances are called as ``denoiser(x)`` or ``denoiser(x, sigma)``, the
    interface :class:`bartorch.priors.ImplicitPrior` requires of a denoiser,
    and are therefore accepted wherever a regularizer is.  The wrapped network
    may come from any library: a ``deepinv`` denoiser, a ``monai`` network and
    a network defined locally are :class:`torch.nn.Module` objects with the
    same calling convention, differing from this package only in input layout.

    Parameters
    ----------
    net : callable
        Called as ``net(planes)``, or as ``net(planes, sigma)`` when a
        ``sigma`` is supplied to :meth:`forward`, on a real tensor of shape
        ``(n, channels, *spatial)``.
    spatial : int, default=2
        Number of trailing image axes the network operates on: 2 for a
        network trained on slices, 3 for one trained on volumes.
    channels : int, default=1
        Number of input channels the network expects.  A grayscale network
        takes 1 and an RGB network 3, both operating on one plane at a time;
        2 is a network taking the real and imaginary planes jointly, as
        MoDL's does.
    parts : {"channels", "separate", "magnitude"}, default=None
        Representation of the complex values as real planes.  ``"channels"``
        assigns the real and imaginary parts to the network's two input
        channels.  ``"separate"`` applies the network to each part
        independently, in a single call over a doubled batch axis.
        ``"magnitude"`` applies it to the modulus and restores the original
        phase, leaving the phase unaltered.  Defaults to ``"channels"`` for a
        two-channel network and ``"separate"`` otherwise.
    normalize : bool, default=True
        Whether each image is scaled to unit peak modulus around the call.
        The noise level of a network trained on images in the unit range is
        not interpretable without this scaling.

    Notes
    -----
    ``sigma`` is expressed in the units of the scaled image, following the
    convention of plug-and-play noise levels, and is passed to the network
    unchanged.

    A real-valued input is treated as a complex image with zero imaginary
    part, and the real part of the result is returned.

    The scale of step 4 is detached, so no gradient propagates to the input
    through the normalization.

    Examples
    --------
    >>> from deepinv.models import DRUNet
    >>> denoiser = learning.Denoiser(DRUNet(in_channels=3, out_channels=3), channels=3)
    >>> image = optim.fista(kspace, A, priors.ImplicitPrior(denoiser, sigma=0.05))
    """

    def __init__(
        self,
        net,
        *,
        spatial: int = 2,
        channels: int = 1,
        parts: str | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        if not callable(net):
            raise TypeError(f"a denoiser is called as net(x) or net(x, sigma), and {net!r} is not")
        spatial = int(spatial)
        if spatial not in (2, 3):
            raise ValueError(
                f"a network takes planes or volumes, so spatial is 2 or 3, not {spatial}"
            )
        channels = int(channels)
        if parts is None:
            parts = "channels" if 2 == channels else "separate"
        if parts not in _PARTS:
            raise ValueError(f"parts is one of {_PARTS}, not {parts!r}")
        if "channels" == parts and 2 != channels:
            raise ValueError(
                "parts='channels' hands the real and imaginary planes to the network's "
                f"channels, so it takes two of them, not {channels}"
            )
        if "channels" != parts and channels not in (1, 3):
            raise ValueError(
                f"parts={parts!r} hands the network one plane, which a grayscale network takes "
                f"as it is and an RGB one takes repeated; {channels} channels is neither"
            )

        self.net = net
        self.spatial = spatial
        self.channels = channels
        self.parts = parts
        self.normalize = bool(normalize)

    def forward(self, input: torch.Tensor, sigma=None) -> torch.Tensor:
        """Denoise ``input``, returning its own shape and dtype.

        Parameters
        ----------
        input : torch.Tensor
            Complex or real, of shape ``(batch, *rest, *spatial)``.  The
            leading axis is the batch over which each scale is measured; a
            tensor of exactly ``spatial`` axes is a single image.
        sigma : float or torch.Tensor, default=None
            Passed to the network as its second argument when supplied, and
            omitted from the call otherwise.
        """
        if input.ndim < self.spatial:
            raise ValueError(
                f"an image for this denoiser carries at least its {self.spatial} spatial axes, "
                f"and {tuple(input.shape)} has {input.ndim}"
            )
        real = not input.is_complex()
        x = input.to(torch.complex64) if real else input
        scale = self._scale(x)
        out = self._denoise(x / scale, sigma) * scale
        return out.real.to(input.dtype) if real else out.to(input.dtype)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Peak modulus of each item of the leading axis, shaped to divide ``x``."""
        if not self.normalize:
            return torch.ones((), dtype=x.real.dtype, device=x.device)
        lead = 1 if x.ndim > self.spatial else 0
        peak = x.detach().abs().reshape(*x.shape[:lead], -1).amax(-1).clamp_min(_TINY)
        return peak.reshape(*x.shape[:lead], *(1,) * (x.ndim - lead))

    def _denoise(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """Apply the network to the planes ``parts`` specifies and recombine the result."""
        spatial = tuple(x.shape[-self.spatial :])

        if "magnitude" == self.parts:
            modulus = x.abs()
            phase = x / modulus.clamp_min(_TINY).to(x.dtype)
            made = self._net(modulus.reshape(-1, 1, *spatial), sigma, 1)
            return made.reshape(x.shape).to(x.dtype) * phase

        if "separate" == self.parts:
            pair = torch.stack([x.real, x.imag]).reshape(-1, 1, *spatial)
            made = self._net(pair, sigma, 1).reshape(2, -1)
            return torch.complex(made[0], made[1]).reshape(x.shape)

        planes = torch.stack([x.real, x.imag], -self.spatial - 1).reshape(-1, 2, *spatial)
        made = self._net(planes, sigma, 2).movedim(1, 0).reshape(2, -1)
        return torch.complex(made[0], made[1]).reshape(x.shape)

    def _net(self, planes: torch.Tensor, sigma, wanted: int) -> torch.Tensor:
        """A single call, matching the network's channel count and checking its output."""
        if 3 == self.channels and 1 == planes.shape[1]:
            planes = planes.repeat(1, 3, *(1,) * self.spatial)
        made = self.net(planes) if sigma is None else self.net(planes, sigma)
        if 3 == self.channels and 3 == made.shape[1]:
            made = made.mean(1, keepdim=True)
        if made.shape[1] != wanted or made.shape[2:] != planes.shape[2:]:
            raise ValueError(
                f"the network answered {tuple(made.shape)} for {tuple(planes.shape)}; a denoiser "
                f"returns {wanted} channel(s) on the shape it was given"
            )
        return made

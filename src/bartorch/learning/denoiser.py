"""A network trained on real images, applied to the images a reconstruction carries."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Denoiser"]

#: What a modulus is held above, so that an empty image divides by something.
_TINY = 1e-12

_PARTS = ("channels", "separate", "magnitude")


class Denoiser(nn.Module):
    """``net``, which takes real images, applied to a complex one of this package's.

    A denoiser from an image-restoration library takes a real tensor of
    ``(n, channels, height, width)`` whose values are around the unit range.
    An image here is complex, carries frames, contrasts or subspace
    coefficients in front of its spatial axes, is sometimes a volume, and is
    scaled by whatever the k-space was scaled by.  This module is the four
    conversions between the two, and nothing else:

    * the last ``spatial`` axes are the image the network sees, and every axis
      in front of them but the first is folded into its batch;
    * the complex values become real planes, as ``parts`` says;
    * a plane is repeated across ``channels`` where the network takes three,
      and the three that come back are averaged;
    * each item of the leading axis is divided by its own peak modulus before
      the network and multiplied by it after.

    It is called as ``denoiser(x)`` or ``denoiser(x, sigma)``, which is what
    :class:`~bartorch.priors.ImplicitPrior` asks of a denoiser, so it stands
    wherever a regularizer does.  Nothing here is specific to one library: a
    ``deepinv`` denoiser, a ``monai`` network and a network written by hand
    are all ``nn.Module``\\s called the same way.

    Parameters
    ----------
    net : callable
        Called as ``net(planes)``, or as ``net(planes, sigma)`` when a
        ``sigma`` reaches :meth:`forward`, on a real tensor of
        ``(n, channels, *spatial)``.
    spatial : int
        How many trailing axes of the image the network takes as one of its
        own: 2 for a network trained on slices, 3 for one trained on volumes.
    channels : int
        What the network's first layer takes.  1 is a grayscale network and 3
        an RGB one, each of which takes one plane at a time; 2 is a network
        that takes the real and imaginary planes together, as MoDL's does.
    parts : {"channels", "separate", "magnitude"}, optional
        How the complex values become real planes.  ``"channels"`` puts the
        real and imaginary parts in the network's two channels.
        ``"separate"`` passes each of them through on its own, in one call
        over a doubled batch.  ``"magnitude"`` denoises the modulus and
        multiplies the phase back in, which leaves the phase untouched.  The
        default is ``"channels"`` for a two-channel network and
        ``"separate"`` otherwise.
    normalize : bool
        Whether each image is scaled to unit peak modulus around the network.
        A denoiser trained on images in the unit range has a noise level that
        means nothing without this.  The scale is measured rather than
        learned: no gradient reaches the image through it.

    Notes
    -----
    ``sigma`` is in the units of the scaled image, which is the convention a
    plug-and-play noise level follows; it is passed to the network as it
    stands.

    A real input is treated as a complex image with no imaginary part, and
    what comes back is the real part of the answer.

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
        """``input`` denoised, of its own shape and dtype.

        Parameters
        ----------
        input : torch.Tensor
            ``(batch, *rest, *spatial)``, complex or real.  The leading axis is
            the batch each scale is measured over; a tensor of exactly
            ``spatial`` axes is one image.
        sigma : float or torch.Tensor, optional
            Passed to the network as its second argument when it is given, and
            not passed at all when it is not.
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
        """The peak modulus of each item of the leading axis, ready to divide by."""
        if not self.normalize:
            return torch.ones((), dtype=x.real.dtype, device=x.device)
        lead = 1 if x.ndim > self.spatial else 0
        peak = x.detach().abs().reshape(*x.shape[:lead], -1).amax(-1).clamp_min(_TINY)
        return peak.reshape(*x.shape[:lead], *(1,) * (x.ndim - lead))

    def _denoise(self, x: torch.Tensor, sigma) -> torch.Tensor:
        """The network over the planes ``parts`` makes of ``x``, put back together."""
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
        """One call, with the channels the network takes and the planes it owes back."""
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

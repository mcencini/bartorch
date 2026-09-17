"""Adapters between neural networks and this package's images and iterations.

:class:`Denoiser` applies a network trained on real-valued images to the
complex, possibly multi-frame images a reconstruction carries, and satisfies
the interface :class:`bartorch.priors.ImplicitPrior` requires of a denoiser.
:class:`Unrolled` applies one of :mod:`bartorch.optim`'s iteration blocks a
fixed number of times, together with the differentiation strategies a deep
stack requires.  :func:`as_real` and :func:`as_complex` convert between complex
tensors and the leading real channel axis used by convolutional networks and by
``torchio`` images.

Training loops, datasets, augmentation, patch sampling, networks, losses and
metrics are not implemented here.  ``lightning``, ``torchio``, ``monai``,
``deepinv`` and ``torchmetrics`` provide them, and this subpackage imports none
of them.
"""

from __future__ import annotations

from bartorch.learning.channels import as_complex, as_real
from bartorch.learning.denoiser import Denoiser
from bartorch.learning.unrolled import Unrolled

__all__ = ["Denoiser", "Unrolled", "as_complex", "as_real"]

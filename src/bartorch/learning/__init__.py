"""What a network needs of this package, and nothing that a training library already has.

Training loops belong to ``lightning``, datasets and their augmentation to
``torchio``, networks, losses and metrics to ``monai``, ``deepinv`` and
``torchmetrics``.  What none of them knows is this package's data: images that
are complex, that carry frames, contrasts or subspace coefficients in front of
their spatial axes, and that a reconstruction is an iteration over.  So this
subpackage is the two adapters between the two, and no more:
:class:`Denoiser` puts a real-valued image network where a regularizer goes,
and :class:`Unrolled` makes a network of one of :mod:`bartorch.optim`'s
iterations.  :func:`as_real` and :func:`as_complex` are the layout a
``torchio`` image and a convolution both want.
"""

from __future__ import annotations

from bartorch.learning.channels import as_complex, as_real
from bartorch.learning.denoiser import Denoiser
from bartorch.learning.unrolled import Unrolled

__all__ = ["Denoiser", "Unrolled", "as_complex", "as_real"]

"""BART, the Berkeley Advanced Reconstruction Toolbox, in-process on torch tensors.

Shapes are C order: an axis argument is an index into a tensor's shape, never
a BART bitmask.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from bartorch import (
    _settings,
    fourier,
    interop,
    interp,
    io,
    linop,
    metrics,
    nlop,
    optim,
    priors,
    registration,
    thresh,
    tools,
    util,
    wavelet,
)
from bartorch._dispatch import BartError
from bartorch._settings import *  # noqa: F401,F403
from bartorch.fourier import *  # noqa: F401,F403
from bartorch.interp import *  # noqa: F401,F403
from bartorch.metrics import *  # noqa: F401,F403
from bartorch.registration import *  # noqa: F401,F403
from bartorch.thresh import *  # noqa: F401,F403
from bartorch.util import *  # noqa: F401,F403
from bartorch.wavelet import *  # noqa: F401,F403

try:
    __version__ = version("bartorch")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "BartError",
    "__version__",
    "interop",
    "io",
    "linop",
    "nlop",
    "optim",
    "priors",
    "tools",
    *_settings.__all__,
    *fourier.__all__,
    *interp.__all__,
    *metrics.__all__,
    *registration.__all__,
    *thresh.__all__,
    *util.__all__,
    *wavelet.__all__,
]

"""BART's applications.

Five sections: simulation, sampling and trajectories, coil calibration,
reconstruction, and the array operations a reconstruction is surrounded by.

Commands without a hand-written wrapper are built from BART's own declaration
of the command and take its options under their long names.
"""

from __future__ import annotations

from bartorch.tools import calib, process, recon, sampling, simulate
from bartorch.tools.calib import *  # noqa: F401,F403
from bartorch.tools.process import *  # noqa: F401,F403
from bartorch.tools.recon import *  # noqa: F401,F403
from bartorch.tools.sampling import *  # noqa: F401,F403
from bartorch.tools.simulate import *  # noqa: F401,F403

__all__ = sorted(
    {*simulate.__all__, *sampling.__all__, *calib.__all__, *recon.__all__, *process.__all__}
)

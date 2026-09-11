"""Every BART command, as a function on tensors.

A command is reached by its own name::

    import bartorch.tools as bt

    kspace = bt.phantom(128, coils=8, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    image = bt.pics(kspace, maps, l2=0.01, solver="fista")

Shapes are C order, so the last axis is the one BART calls the first, and an
axis argument is an index into that shape rather than a bitmask.

Two kinds of wrapper
--------------------
Most of what is here is written by hand, against the catalogue of what BART's
own sources declare: an axis instead of a bitmask, one ``solver`` instead of
five flags, a tensor wherever BART names a file.  The rest is built from that
same catalogue and is shaped like the command line -- complete, and honest
about being a transliteration.  :func:`describe` prints what either takes, and
``.is_derived`` says which kind a wrapper is.

A handful of commands are not here at all, because they read or write
something that is not an array: see ``bartorch.tools._coverage.NOT_EXPOSED``,
which names each and why.  ``tests/test_tools.py`` asserts that the three
groups together are every command BART builds.
"""

from __future__ import annotations

from bartorch._catalogue import COMMANDS
from bartorch._options import describe as describe
from bartorch.tools import _call, _coverage
from bartorch.tools.arith import *  # noqa: F401,F403
from bartorch.tools.array import *  # noqa: F401,F403
from bartorch.tools.calib import *  # noqa: F401,F403
from bartorch.tools.fourier import *  # noqa: F401,F403
from bartorch.tools.recon import *  # noqa: F401,F403
from bartorch.tools.sampling import *  # noqa: F401,F403
from bartorch.tools.simulate import *  # noqa: F401,F403

_curated = _coverage.curated_wrappers()

# Every command that has no wrapper of its own, built from its catalogue entry.
# Eagerly, so that `dir()` and an editor's completion see all of them, and so
# that a command whose entry cannot be turned into a signature fails at import
# rather than the first time someone calls it.
for _name in sorted(_coverage.derived_names()):
    globals().setdefault(_name, _call.build(_name))

#: The name each wrapper goes by, which is the command's own except where a
#: command's name is not an identifier.
__all__ = sorted({"describe", *(n for n in globals() if n in COMMANDS or n in ("ifft",))})

del _name

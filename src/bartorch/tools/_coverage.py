"""Which BART commands this package exposes, and how.

Every command BART builds is in exactly one of three places, and
``tests/test_tools.py`` asserts that the three together are all of them.  A
command added by a submodule bump lands in :data:`DERIVED` on its own and is
reachable the same day; one that cannot work here has to be named, with a
reason, rather than quietly going missing.

curated
    A wrapper written by hand, because the command needed a judgement the
    catalogue cannot make: that an axis is better than a bitmask, that two
    options are one choice, that a flag BART calls a file takes a tensor.

derived
    Built from the catalogue by :mod:`bartorch.tools._call`.  Complete, and
    shaped like the command line.

not exposed
    Reads or writes something that is not an array: a file on disk, standard
    input, a graph.  Nothing here has filenames, so there is nothing to pass.
"""

from __future__ import annotations

from bartorch._catalogue import COMMANDS

__all__ = ["NOT_EXPOSED", "curated_names", "curated_wrappers", "derived_names"]

#: Commands that work on something other than arrays, and why each one cannot
#: be a function on tensors.  A path is the thing this package does not have.
NOT_EXPOSED: dict[str, str] = {
    "bart": "the dispatcher that runs the other commands, not a command itself",
    "tee": "copies standard input to files on disk",
    "multicfl": "combines and splits .cfl files on disk",
    "tensorflow": "loads a TensorFlow graph from a path",
    "twixread": "reads a Siemens .dat file from disk, which is not a CFL array",
    "ismrmrd": "reads and writes ISMRMRD files on disk",
    "toimg": "writes PNG or proto-DICOM files to a path prefix",
    "toraw": "writes raw samples to standard output",
    "stl": "reads and writes .stl files on disk",
}


def curated_wrappers() -> dict[str, object]:
    """Every hand-written wrapper, by the BART command it stands for.

    A curated module marks each of its wrappers with the command it wraps, so
    this finds them rather than being told twice.
    """
    # A module may not be named after a BART command: `sim` and `sample` are
    # both commands, and `bartorch.tools.sim` would then be two things.
    from bartorch.tools import arith, array, calib, fourier, recon, sampling, simulate

    found: dict[str, object] = {}
    for module in (arith, array, calib, fourier, recon, sampling, simulate):
        for name in getattr(module, "__all__", ()):
            wrapper = getattr(module, name)
            command = getattr(wrapper, "bart_command", None)
            if command is not None:
                found[command] = wrapper
    return found


def curated_names() -> frozenset[str]:
    """The commands a hand-written wrapper covers."""
    return frozenset(curated_wrappers())


def derived_names() -> frozenset[str]:
    """The commands built from the catalogue, which is everything left over."""
    return frozenset(COMMANDS) - curated_names() - frozenset(NOT_EXPOSED)

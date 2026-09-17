"""Where each BART command is exposed, or why it is not.

Every command BART builds is exactly one of: wrapped by hand (marked with
:func:`bartorch._call.curated`), derived from the catalogue into a
:mod:`bartorch.tools` section, or private.  ``tests/test_tools.py`` checks the
partition, so a command added by a BART update must be placed before the
suite passes.  A private command still runs through
:func:`bartorch._dispatch.dispatch`.
"""

from __future__ import annotations

from importlib import import_module

from bartorch._catalogue import COMMANDS

__all__ = [
    "CURATED_MODULES",
    "PRIVATE",
    "TOOLS_MODULES",
    "curated_names",
    "curated_wrappers",
    "derived_names",
]

#: The :mod:`bartorch.tools` sections; each lists its derived commands in ``_DERIVED``.
TOOLS_MODULES = (
    "bartorch.tools.simulate",
    "bartorch.tools.sampling",
    "bartorch.tools.calib",
    "bartorch.tools.recon",
    "bartorch.tools.process",
)

#: Every module holding hand-written wrappers.
CURATED_MODULES = (
    "bartorch.fourier",
    "bartorch.util",
    "bartorch.wavelet",
    "bartorch.thresh",
    "bartorch.interp",
    "bartorch.io",
    "bartorch.priors.denoise",
    *TOOLS_MODULES,
)

_FILES = "reads or writes files or streams rather than arrays"
_NETWORK = "trains or applies BART's own networks, whose weights are files"
_TORCH = "an array operation torch provides"
_DEMO = "a demonstration or a diagnostic"
_LATER = "not wrapped yet"

#: Commands without a public wrapper, and why.
PRIVATE: dict[str, str] = {
    "bart": "the dispatcher that runs the other commands",
    "ismrmrd": "needs libismrmrd, which this build does not compile",
    **dict.fromkeys(("tee", "multicfl", "tensorflow", "twixread", "toimg", "toraw"), _FILES),
    **dict.fromkeys(("stl", "pol2mask", "morphop"), "mesh and mask geometry, not needed here"),
    **dict.fromkeys(
        ("cunet", "mnist", "nnet", "reconet", "nlinvnet", "sample", "onehotenc"), _NETWORK
    ),
    **dict.fromkeys(("conway", "mandelbrot", "bench", "show"), _DEMO),
    **dict.fromkeys(
        (
            "cabs",
            "carg",
            "conj",
            "creal",
            "cpyphs",
            "invert",
            "spow",
            "zexp",
            "saxpy",
            "scale",
            "sdot",
            "fmac",
            "avg",
            "std",
            "var",
            "zeros",
            "ones",
            "index",
            "vec",
            "copy",
            "reshape",
            "squeeze",
            "flatten",
            "transpose",
            "repmat",
            "join",
            "slice",
            "extract",
            "calc",
            "poly",
            "svd",
            "hist",
            "compress",
        ),
        _TORCH,
    ),
    **dict.fromkeys(("fftrot", "gmm"), _LATER),
    "ictv": "fails for every input in this BART (ictv.c:97 reshapes the wrong side)",
    "version": "bartorch.bart_version() reports it",
    "bitmask": "converts bitmasks, which the Python API does not use",
    "crop": "bartorch.resize covers it",
    "delta": "an identity tensor, which torch.eye makes",
    "denoise": "an optim solver over an identity operator solves the same problem",
}


def curated_wrappers() -> dict[str, list]:
    """Every hand-written wrapper, grouped by the BART command it runs."""
    found: dict[str, list] = {}
    for name in CURATED_MODULES:
        module = import_module(name)
        for attr in getattr(module, "__all__", ()):
            wrapper = getattr(module, attr)
            if getattr(wrapper, "is_derived", True):
                continue
            for command in getattr(wrapper, "bart_commands", ()):
                found.setdefault(command, []).append(wrapper)
    return found


def curated_names() -> frozenset[str]:
    return frozenset(curated_wrappers())


def derived_names() -> frozenset[str]:
    return frozenset(
        name for module in TOOLS_MODULES for name in getattr(import_module(module), "_DERIVED", ())
    )


# Checked here rather than in a test, so that a module cannot name a command
# twice without the import failing.
assert frozenset(PRIVATE) <= frozenset(COMMANDS), sorted(frozenset(PRIVATE) - frozenset(COMMANDS))

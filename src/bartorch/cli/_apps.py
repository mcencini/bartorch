"""The commands an app answers, and what each of BART's flags means to it.

An app is a re-expression of its application, so which of the two ran is a
question about speed and not about the answer; the table below is what decides,
and an argument missing from it sends the whole command line back to BART
rather than being ignored.
"""

from __future__ import annotations

from typing import Any

from bartorch.cli._argv import Unsupported, axes, regularizer

__all__ = ["ADAPTERS"]


def _only(values: list[Any]) -> Any:
    """A flag BART takes once, as it was given."""
    return values[-1]


def _pics(options: dict[str, list[Any]], ndim: int) -> dict[str, Any]:
    """``pics``'s flags as :func:`bartorch.apps.pics` takes them."""
    from bartorch import priors

    made: dict[str, Any] = {}
    terms = [regularizer(text, ndim) for text in options.pop("R", [])]

    # `-l1` and `-l2` are one term apiece over the three transformed axes, and
    # they take the weight `-r` carries rather than one of their own
    # (`grecon/optreg.c:260`); a `-r` left over after that is the plain
    # Tikhonov penalty below.
    weight = options["r"][-1] if "r" in options else 0.0
    for kind in options.pop("l", []):
        if str(kind) == "1":
            terms.append(priors.Wavelet(axes(7, ndim), weight))
        elif str(kind) == "2":
            terms.append(priors.L2(weight))
        else:
            raise Unsupported(f"pics -l takes 1 or 2, got {kind!r}")
        options.pop("r", None)

    for flag, keyword in (
        ("i", "maxiter"),
        ("s", "step"),
        ("u", "admm_rho"),
        ("C", "cg_maxiter"),
        ("w", "scaling"),
        ("t", "traj"),
        ("p", "pattern"),
        ("B", "basis"),
        ("W", "initial"),
    ):
        if flag in options:
            made[keyword] = _only(options.pop(flag))
    if "r" in options:
        made["l2"] = _only(options.pop("r"))
    if "e" in options:
        options.pop("e")
        made["eigen_step"] = True
    if "no_toeplitz" in options:
        options.pop("no_toeplitz")
        made["toeplitz"] = False

    for arm in ("ist", "fista", "admm", "pridu"):
        if arm in options:
            options.pop(arm)
            made["solver"] = arm
    if "eulermaruyama" in options:
        raise Unsupported("-R eulermaruyama is not one of the app's iterations")

    if terms:
        made["regularizers"] = terms
    if options:
        raise Unsupported(f"pics {sorted(options)} is not something the app takes")
    return made


def _pocsense(options: dict[str, list[Any]], ndim: int) -> dict[str, Any]:
    """``pocsense``'s flags as :func:`bartorch.apps.pocsense` takes them."""
    made: dict[str, Any] = {}
    if "i" in options:
        made["maxiter"] = _only(options.pop("i"))
    if "r" in options:
        made["alpha"] = _only(options.pop("r"))
    if "o" in options:
        made["robust"] = _only(options.pop("o"))
    if "l" in options:
        kind = _only(options.pop("l"))
        if kind not in (1, 2):
            raise Unsupported(f"pocsense -l takes 1 or 2, got {kind}")
        made["wavelet"] = kind == 1
    if "g" in options:
        options.pop("g")
    if options:
        raise Unsupported(f"pocsense {sorted(options)} is not something the app takes")
    return made


#: The commands an app answers, and the reader that fills its arguments.  A
#: command absent from here is BART's own.
ADAPTERS = {
    "pics": _pics,
    "pocsense": _pocsense,
}

"""BART's reconstruction pipelines, assembled from this package.

An app is what a BART application does, written against :mod:`bartorch.linop`,
:mod:`bartorch.optim` and :mod:`bartorch.priors` rather than by running the
command.  It takes tensors and Python arguments instead of a string of flags,
and it returns the bits the command returns: the work around the solve is the
command's own -- the sampling pattern, the modulation, the scaling it
estimates -- so an app is a re-expression of the application and not an
arithmetic that agrees with it.

:mod:`bartorch.cli` wraps these back up in BART's own command line.
"""

from __future__ import annotations

from bartorch.apps.pics import pics

__all__ = ["pics"]

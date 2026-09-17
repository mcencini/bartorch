# Applications

```{eval-rst}
.. currentmodule:: bartorch.apps
```

BART's reconstruction pipelines, assembled from this package rather than run as
commands. An app takes tensors and Python arguments in place of a string of
flags, and returns what the application returns: the work around the solve is
the application's own -- the sampling pattern, the modulation into the
convention BART iterates in, the scaling it estimates -- so an app is a
re-expression of the application and not an arithmetic that agrees with it.

On a Cartesian grid `tests/test_apps.py` holds the two to `torch.equal` across
fifteen configurations. Off the grid they agree to single-precision round-off,
because the scaling the application estimates and
{func}`bartorch.optim.data_scaling` are the same number computed by different
expressions.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pics
```

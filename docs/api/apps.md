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
fifteen configurations.

Off the grid they agree to round-off instead, and no assertion there could say
more: a non-uniform transform spread over threads sums in the order the threads
finish in, so nothing off the grid is bit-reproducible.
{func}`bartorch.tools.pics` run twice over the same data differs from itself by
about 6e-07 of the peak -- the same size as its difference from the app.

`apps.pics` covers what `bartorch.tools.pics` covers apart from three: `real`
and `lowmem` want a BART operator constructor this package does not wrap yet,
and `psf` cannot work here at all -- importing a point spread function reads
the NUFFT operator's internals, and the substituted operator refuses.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pics
```

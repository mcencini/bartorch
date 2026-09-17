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

`apps.mobafit` is the one app that answers in different numbers from its
command, and deliberately: the model it fits is TorchSim's rather than BART's,
because what a fit needs is a forward it can differentiate, with bounds and a
starting state. The method is the command's -- the same Gauss-Newton loop over
the same linearized least-squares problem -- and what it returns is named maps
in their own units rather than a stack of coefficients. `tests/test_apps.py`
holds it against decays and recoveries written out in the test.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   mobafit
   pics
```

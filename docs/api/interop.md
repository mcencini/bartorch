# Interoperability

`bartorch.interop`.  Adapters that hand bartorch's operators to other
libraries.  The other direction needs no adapter: a denoiser goes in
{class}`bartorch.priors.ImplicitPrior`, and an operator in
{meth}`bartorch.linop.LinearOperator.from_callbacks`.

```{eval-rst}
.. currentmodule:: bartorch.interop
```

## deepinv

A `deepinv.physics.LinearPhysics`, for deepinv's losses and samplers.  deepinv
is the `bartorch[deepinv]` extra.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   to_deepinv
```

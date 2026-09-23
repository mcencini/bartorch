# Interoperability

`bartorch.interop` adapts this package's operators to other libraries.  The
reverse direction needs no adapter: a denoiser is accepted by
{obj}`~bartorch.priors.ImplicitPrior`, and a Python operator by
{meth}`bartorch.linop.LinearOperator.from_callbacks`.

```{eval-rst}
.. currentmodule:: bartorch.interop
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.interop.to_deepinv` | A {obj}`~bartorch.linop.LinearOperator` as a `deepinv.physics.LinearPhysics`, with a leading batch axis and a conjugate-gradient `A_dagger` |

DeepInverse is required only for its algorithms that evaluate a physics
operator, such as its diffusion samplers and its measurement-consistency,
equivariant-imaging and SURE losses; install it with
`pip install 'bartorch[deepinv]'`.

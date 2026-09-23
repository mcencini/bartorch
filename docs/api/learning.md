# Learning

`bartorch.learning` converts between neural networks and the complex images
and iterations of this package: an adapter that applies a real-valued image
network to a complex image, a module that unrolls an iteration block into a
network, and the real/complex channel conversions.  It imports no training
library.  {doc}`../explanation/differentiation` describes how gradients pass
through an unrolled iteration and the memory each backward strategy uses.

```{eval-rst}
.. currentmodule:: bartorch.learning
```

## Networks as regularizers

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.Denoiser` | Adapter applying a network on real `(n, channels, *spatial)` planes to a complex image; accepted by {obj}`~bartorch.priors.ImplicitPrior` |

## Iterations as networks

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.Unrolled` | A fixed number of applications of an iteration block from {mod}`bartorch.optim` or {mod}`bartorch.nlop`, with shared or per-iteration parameters |

| `Unrolled` setting | Graph recorded | Memory | Gradient |
| --- | --- | --- | --- |
| default | The whole stack | Iterations × one step | End to end |
| `detach=True` | One iteration at a time | One step | Of each step alone (greedy training with {meth}`Unrolled.steps`) |
| `checkpoint=True` | The states between iterations | States + one step | End to end; each step is computed twice |

## Complex values as channels

| Object | Description |
| --- | --- |
| {obj}`~bartorch.learning.as_real` | Real and imaginary parts on a new leading axis |
| {obj}`~bartorch.learning.as_complex` | Inverse of {obj}`~bartorch.learning.as_real` |

{doc}`../auto_examples/05-deep-learning/01-modl-with-admm` trains an unrolled
network built from these objects.

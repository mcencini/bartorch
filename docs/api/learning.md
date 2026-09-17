# Learning

`bartorch.learning`.  Adapters between neural networks and this package's
images and iterations.

Training loops are provided by [PyTorch
Lightning](https://lightning.ai/docs/pytorch/stable), datasets, augmentation
and patch sampling by [torchio](https://torchio.org/), and networks, losses and
metrics by [MONAI](https://project-monai.github.io/),
[deepinv](https://deepinv.org/) and torchmetrics.  None of these represents an
image that is complex, carries frames, contrasts or subspace coefficients in
front of its spatial axes, and is reconstructed by an iteration.  This module
supplies the conversions between the two, and imports none of those libraries.

```{eval-rst}
.. currentmodule:: bartorch.learning
```

## Networks as regularizers

An image-restoration network operates on a real tensor of shape
`(n, channels, height, width)` whose values are of order unity.
{class}`Denoiser` performs the conversion: the spatial axes are retained, the
axes in front of them are folded into the network's batch axis, the complex
values are laid out as real planes, a plane is replicated where the network
takes three channels, and each image is scaled to unit peak modulus around the
call.  Instances satisfy the calling convention
{class}`bartorch.priors.ImplicitPrior` requires of a denoiser.

The wrapped network may come from any library.  A `deepinv` denoiser, a `monai`
network and a network defined locally are `nn.Module` objects with the same
calling convention, differing only in input layout.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Denoiser
```

## Iterations as networks

{class}`Unrolled` applies one of {mod}`bartorch.optim`'s iteration blocks a
fixed number of times.  With every parameter frozen the stack reproduces the
solver it was constructed from; calling `requires_grad_()` on a step size, a
penalty weight or a denoiser's weights makes it a trainable network.  A single
shared block gives the weight sharing usual in unrolled networks; a sequence
gives each iteration its own parameters.

`detach` and `checkpoint` select how the backward pass is taken.  Neither
alters the value the network computes:

| | recorded | memory | gradient |
| --- | --- | --- | --- |
| default | the whole stack | iterations x one step | end to end |
| `detach=True` | one iteration | one step | each step's own |
| `checkpoint=True` | the states between iterations | states + one step | end to end |

These correspond to the stages in which a large unrolled network is trained: a
denoiser pretrained in isolation, then greedy per-iteration training against a
loss on each image yielded by {meth}`Unrolled.steps`, then end-to-end
fine-tuning with gradient checkpointing.
{class}`bartorch.optim.FixedPoint` is an alternative with bounded memory: it
drives the block to its fixed point and differentiates there, with no iteration
count to unroll.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Unrolled
```

## Complex values as channels

`torchio.ScalarImage` and a convolution both place channels in a leading axis,
and neither accepts a complex tensor.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   as_real
   as_complex
```

## Scope

Losses and metrics are not provided.  A supervised loss on a reconstructed
image requires nothing from this package, the reconstruction being an ordinary
tensor by that point; `torchmetrics`, `monai.losses`, `monai.metrics` and
`deepinv.loss` apply directly.  A loss that evaluates the forward model is
written over the operator, which is already a callable with an adjoint:

```python
consistency = (A(x_net) - y).abs().square().mean()
```

Datasets, augmentation and patch sampling are not provided either.  `torchio`
applies a transform consistently to every image of a `Subject`, which keeps an
augmented image registered with its coil sensitivities; {func}`as_real` is the
only conversion its images require.

A complete unrolled reconstruction assembled from these objects is in
{doc}`../auto_examples/05-deep-learning/01-modl-with-admm`.

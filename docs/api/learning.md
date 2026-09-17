# Learning

`bartorch.learning`.  What a network needs of this package, and nothing a
training library already has.

Training loops belong to [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable),
datasets, their augmentation and their patch sampling to
[torchio](https://torchio.org/), and networks, losses and metrics to
[MONAI](https://project-monai.github.io/), [deepinv](https://deepinv.org/) and
torchmetrics.  None of them knows this package's data: images that are complex,
that carry frames, contrasts or subspace coefficients in front of their spatial
axes, and that a reconstruction is an iteration over.  So this module is the
adapters between the two, and no more.

```{eval-rst}
.. currentmodule:: bartorch.learning
```

## A network where a regularizer goes

A denoiser from an image-restoration library takes a real tensor of
`(n, channels, height, width)` whose values are around the unit range.
{class}`Denoiser` is the conversion: the spatial axes are kept, the axes in
front of them are folded into the network's batch, the complex values become
real planes, a plane is repeated where the network takes three, and each image
is scaled to unit peak modulus around the call.  What comes back is called as
`denoiser(x)` or `denoiser(x, sigma)`, which is what
{class}`bartorch.priors.ImplicitPrior` asks of a denoiser.

Nothing in it is specific to one library.  A `deepinv` denoiser, a `monai`
network and a network written by hand are `nn.Module`s called the same way, so
none of them needs an adapter of its own -- only its input layout does.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Denoiser
```

## An iteration as a network

{class}`Unrolled` applies one of {mod}`bartorch.optim`'s blocks a fixed number
of times.  The block is BART's step and the loop is the module, so with every
parameter frozen the stack is the solver it was built from, and
`requires_grad_()` on a step size, a penalty weight or a denoiser's weights is
what makes it a network.  One block shared by every iteration is the weight
sharing an unrolled network usually means; a sequence gives each iteration its
own.

`detach` and `checkpoint` decide how the backward pass is taken, and neither
changes what the network computes:

| | what is recorded | memory | gradient |
| --- | --- | --- | --- |
| the default | the whole stack | iterations x one step | end to end |
| `detach=True` | one iteration | one step | each step's own |
| `checkpoint=True` | the states between iterations | states + one step | end to end |

Those are the stages a large unrolled network is trained in -- a denoiser
pretrained on its own, then greedily per iteration against a loss on each of
{meth}`Unrolled.steps`, then the whole stack fine-tuned with checkpointing --
and each is set by construction rather than by rebuilding the model.
{class}`bartorch.optim.FixedPoint` is the other route to a memory that does not
grow: it drives the block to its fixed point and differentiates there, with no
iteration count to unroll at all.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Unrolled
```

## Complex values as channels

A `torchio.ScalarImage` and a convolution both carry their channels in a
leading axis, and neither takes a complex tensor.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   as_real
   as_complex
```

## What is not here

Losses and metrics.  A supervised loss over a reconstructed image is
`torchmetrics`', `monai.losses`' or `deepinv.loss`', and needs nothing of this
package: the images are ordinary tensors by then.  A loss that reads the
forward model -- measurement consistency, k-space splitting, equivariant
imaging -- is written over the operator itself, which is already a callable
with an adjoint:

```python
consistency = (A(x_net) - y).abs().square().mean()
```

Datasets, augmentation and patch sampling.  `torchio` does all three over
`Subject`s, and applies one transform consistently to every image in a subject,
which is what keeps an augmented image and its coil sensitivities together.
{func}`as_real` is the only thing between its images and this package's.

Training loops, checkpointing, logging and devices.  `lightning` does those,
and an unrolled model is an `nn.Module` like any other.

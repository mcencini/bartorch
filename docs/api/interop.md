# Interoperability

`bartorch.interop`.  Adapters that hand bartorch's operators to other
libraries.  The other direction needs no adapter: a denoiser goes in
{class}`bartorch.priors.ImplicitPrior`, and an operator in
{meth}`bartorch.linop.LinearOperator.from_callbacks`.

```{eval-rst}
.. currentmodule:: bartorch.interop
```

## deepinv

A `deepinv.physics.LinearPhysics`, for the deepinv algorithms that evaluate a
forward model: its samplers (`DiffPIR`, `DPS`, `DDRM`, `ULA`) and the losses
defined in terms of the physics -- measurement consistency, measurement
splitting, equivariant imaging, SURE.  deepinv is the `bartorch[deepinv]`
extra.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   to_deepinv
```

No other use of deepinv requires the adapter, and nothing else in this package
imports it.

A denoiser is an `nn.Module` called as `net(x)` or `net(x, sigma)` and is
therefore accepted directly by {class}`bartorch.priors.ImplicitPrior`;
{class}`bartorch.learning.Denoiser` adapts its input layout to the complex
images of a reconstruction.

Supervised losses and metrics take a reconstructed image and a reference, with
no reference to the forward model, so `torchmetrics`, `monai.losses` and
`monai.metrics` apply directly, as do `deepinv.loss.SupLoss`, `MSE`, `PSNR` and
`SSIM`.

A loss that does evaluate the forward model can be written over the operator
itself:

```python
consistency = (A(x_net) - y).abs().square().mean()
```

Training loops, datasets, augmentation and patch sampling belong to `lightning`
and `torchio`, neither of which represents a forward operator.  {doc}`learning`
holds the conversions between a network and this package's data.

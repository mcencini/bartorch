# Interoperability

`bartorch.interop`.  Adapters that hand bartorch's operators to other
libraries.  The other direction needs no adapter: a denoiser goes in
{class}`bartorch.priors.ImplicitPrior`, and an operator in
{meth}`bartorch.linop.LinearOperator.from_callbacks`.

```{eval-rst}
.. currentmodule:: bartorch.interop
```

## deepinv

A `deepinv.physics.LinearPhysics`, for what deepinv does *with* a forward
model: its samplers (`DiffPIR`, `DPS`, `DDRM`, `ULA`), and the losses that read
one -- measurement consistency, k-space splitting, equivariant imaging, SURE.
deepinv is the `bartorch[deepinv]` extra.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   to_deepinv
```

Nothing else about deepinv needs it, and nothing in this package uses it.

A denoiser is an `nn.Module` called as `net(x)` or `net(x, sigma)`, so it goes
straight into {class}`bartorch.priors.ImplicitPrior`;
{class}`bartorch.learning.Denoiser` is the layout between one and a complex
image here, and imports neither deepinv nor anything else.

A supervised loss or a metric sees a reconstructed tensor and does not know
where it came from, so `torchmetrics`, `monai.losses` and `monai.metrics`
serve, as does `deepinv.loss` -- `SupLoss`, `MSE`, `PSNR`, `SSIM` and the rest
take `(x_net, x)` and no physics.

A loss that does read the forward model is written over the operator, which is
already a callable with an adjoint:

```python
consistency = (A(x_net) - y).abs().square().mean()
```

Training loops, datasets, augmentation and patch sampling are `lightning`'s and
`torchio`'s, and neither has a notion of a forward operator to adapt to.
{doc}`learning` is what stands between a network and this package.

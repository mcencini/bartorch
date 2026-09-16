# Nonlinear operators

`bartorch.nlop`.  A nonlinear operator maps *many* inputs to *many* outputs and
carries its derivative, and that derivative's adjoint, at the last evaluated
point.  Arguments are counted outputs first, then inputs.  Applying an operator
to a tensor that requires a gradient records the application for autograd; the
backward pass is the adjoint of the derivative at that point.

```{eval-rst}
.. currentmodule:: bartorch.nlop
```

## MRI encoding operators

The coil sensitivities are a second unknown rather than a fixed tensor, so the
operator maps an image and a coil representation to data, and reconstruction is
a joint estimation problem solved by Gauss-Newton.  In
{class}`NonlinearSense` the coil unknown is not the sensitivity maps themselves
but a Sobolev-weighted k-space representation of them; see its documentation.
{func}`CoilSense` is the same product of two unknowns in front of any linear
encoding.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearSense
   CartesianSense
   NoncartesianSense
   CoilSense
```

## Signal models

Quantitative signal models over
[TorchSim](https://github.com/FiRMLAB-Pisa/torchsim)'s simulators, on
TorchSim's parameterisation.  Each maps parameter maps to a series of
contrasts, and is fitted by {class}`IRGNM`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorchSim
   InversionRecovery
   MultiEcho
   Bloch
```

## Nonlinear operators and operator algebra

`a @ b` composes, applying `b` first, and accepts a
{class}`~bartorch.linop.LinearOperator` on either side.  Beyond composition an
operator with several arguments is rearranged by methods rather than by
symbols: `F.chain(G, output=i, input=j)` feeds one output into one input,
`F.combine(G)` places two side by side, and `F.link`, `F.dup`,
`F.stack_inputs`, `F.permute_inputs`, `F.permute_outputs`, `F.reshape_input`,
`F.reshape_output` and `F.flatten` rearrange what is left.  {func}`chain` and
{func}`combine` are the same two operations as functions.  The concrete types
these return are implementation detail; each is a {class}`NonlinearOperator`.

{class}`Derivative` is one output's derivative by one input as a linear
operator, also reached as `F.jacobian(...)`.  {class}`Bundle` declares an
operator's derivative and adjoint with the linearization point as an explicit
argument, the form a Gauss-Newton step is assembled over.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
   chain
   combine
   Derivative
   Bundle
```

## Elementary operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Multiply
   Divide
   Weighted
   Constant
   Exp
   Log
   Sqrt
   Power
   Add
   Inverse
   Abs
   SmoothAbs
   Phase
   Sum
   RootSumOfSquares
```

## User-defined operators

A function of several tensors becomes an operator of several inputs, so a
denoiser's weights become arguments of the operator graph rather than values
captured by closure.  {meth}`NonlinearOperator.from_callbacks` takes the
forward, the derivative and the adjoint as functions; {class}`FromTorch` takes
one differentiable function and obtains the other two from autograd.

A tensor captured by closure inside a Python callback is not an input of the
operator, so no cotangent is propagated to it.  Pass such weights as explicit
operator arguments instead; {class}`Parameters` packs a module's parameters
into one argument for this purpose.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorch
   Parameters
```

## Gauss-Newton methods

{class}`IRGNM` solves `F(x) = y` by iteratively regularized Gauss-Newton, and
covers two forms.  Without `inner=` the linearized problem is solved by
conjugate gradients inside the library.  With `inner=` it is handed to any
solver in {mod}`bartorch.optim`, so the Gauss-Newton step can be regularized by
any term in {mod}`bartorch.priors`.

```python
nlop.IRGNM(inner=optim.CG())                                 # the unregularized form
nlop.IRGNM(inner=optim.FISTA(priors.Wavelet(axes, 0.001)))   # wavelet-regularized
```

{class}`IRGNMBlock` is one Gauss-Newton step as a torch module, with the
interface of the blocks in {mod}`bartorch.optim`: `state = block.start(y, F, x0,
xref)` prepares the data and the model, `state = block(state, F)` takes one
step, and `block.output(state, F)` returns the unknowns.  {class}`IRGNM` is
these calls in a loop.  Without `inner=` a step is differentiable with respect
to the data, the iterate, the regularization centre and `alpha`.  With
`inner=` the inner solver runs over the derivative linearized at the iterate,
and the gradient reaches the iterate through it.  A block's `alpha` is the
weight of the first step, and the block taking step `k` applies it decayed `k`
times: blocks with the same `alpha` follow BART's schedule, and blocks whose
`alpha` requires a gradient learn one weight per step.

A leading axis on the data is a batch of independent items, each stepped on its
own.  {func}`CoilSense` with `items=True` holds the items inside one model
instead: the model is applied to all of them at once, and the inner conjugate
gradients keep their scalars per item, which is faster where an item is small.

```python
blocks = nn.ModuleList(nlop.IRGNMBlock(cg_maxiter=30) for _ in range(2))
state = blocks[0].start(y, F)
for block in blocks:
    state = block(state, F)
    state = dataclasses.replace(state, x=denoise(state.x))
x = blocks[-1].output(state, F)
```

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IRGNM
   IRGNMBlock
   irgnm
```

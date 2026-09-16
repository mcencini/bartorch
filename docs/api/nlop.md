# Nonlinear operators

`bartorch.nlop`.  A nonlinear operator maps one or more inputs to one or more
outputs and supplies its derivative and the derivative's adjoint.  Applying an
operator to a tensor that requires a gradient records the application for
autograd; the backward pass is the adjoint of the derivative at that point.

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

   SignalModel
   InversionRecovery
   MultiEcho
   Bloch
```

## Nonlinear operators

`F(x)` applies an operator and, for a tensor that requires a gradient, records
the application.  `a @ b` composes, applying `b` first; either side may be a
{class}`~bartorch.linop.LinearOperator`.  `F.partial(i, value)` fixes input
`i` to `value`, so `nlop.Exp(s) @ nlop.Multiply(s, s).partial(0, -t)` is
`exp(-t x)`.

`F.linearize(*x)` is the derivative at `x` as a linear operator.  It holds
`x`: it answers the same whatever is evaluated afterwards, and it is
differentiable with respect to `x`.  `input=` selects the input the derivative
is taken by, with the others held at their values; without it the derivative
is taken by all inputs laid end to end, which is how a Gauss-Newton step sees
them.  `output=` selects the output.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
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

{meth}`NonlinearOperator.from_callbacks` takes the forward, the derivative and
the adjoint as functions.  {class}`TorchOperator` takes one differentiable
function and obtains the other two from autograd.  Either can be fitted by
{class}`IRGNM`, composed with `@`, and linearized.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   TorchOperator
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

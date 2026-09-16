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

{meth}`IRGNM.operator` exposes a single Gauss-Newton step as one operator,
differentiable with respect to its data, iterate, regularization centre and
regularization weight.  An unrolled network is then itself one operator:
chaining the steps leaves the whole graph inside the library, with one crossing
into Python per step for the prior.

```python
first, second = (nlop.IRGNM(iterations=1).operator(F) for _ in range(2))
prior = nlop.FromTorch(lambda x, w: w * x, [state, weights.shape], state)
whole = nlop.chain(nlop.chain(first, prior, output=0, input=0), second, output=0, input=1)
whole(y, x0, alpha, packed, ...).abs().square().sum().backward()
```

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IRGNM
   irgnm
```

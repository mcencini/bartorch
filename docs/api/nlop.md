# Nonlinear operators

`bartorch.nlop`.  An operator maps *many* inputs to *many* outputs, as BART's
`nlop_s` does, and carries its derivative and that derivative's adjoint at the
last evaluated point.  Arguments are counted BART's way throughout -- outputs
first, then inputs.

```{eval-rst}
.. currentmodule:: bartorch.nlop
```

## Operator class

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
```

## Operator algebra

The algebra of `nlops/chain.h`.  BART applies a combination back to front, so
an operator that produces a value goes *second*; {func}`chain` arranges that
itself.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   chain
   combine
   Chain
   FromLinear
   Derivative
   Bundle
   Plan
```

## Elementary nonlinear operators

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

## Nonlinear MRI forward models

The coil sensitivities are a second unknown rather than a fixed tensor, so the
operator maps an image and a coil representation to data and the reconstruction
is a joint estimation problem, solved by Gauss-Newton as in `nlinv`.  In
{class}`NonlinearSense` the coil unknown is not the sensitivity maps themselves
but a Sobolev-weighted k-space representation of them; see its documentation.

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

`moba`'s families over [TorchSim](https://github.com/FiRMLAB-Pisa/torchsim)'s
simulators, on TorchSim's parameterisation rather than `moba`'s.  The suite
holds the curves against `bart signal` and the fits against `bart mobafit`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorchSim
   InversionRecovery
   MultiEcho
   Bloch
```

## Gauss-Newton methods

BART implements iteratively regularized Gauss-Newton in two forms, and
{class}`IRGNM` covers both.  Without `inner=` it is `irgnm`, whose linearized
problem is solved by conjugate gradients inside the library.  With `inner=` it
is `irgnm2`, whose linearized problem is handed to any solver in
{mod}`bartorch.optim`, so the Gauss-Newton step can be regularized by any term
in {mod}`bartorch.priors`.

```python
nlop.IRGNM(inner=optim.CG())                                 # iter4_irgnm2, bit for bit
nlop.IRGNM(inner=optim.FISTA(priors.Wavelet(axes, 0.001)))   # the form moba -l1 uses
```

For a {class}`NonlinearSense`, {meth}`IRGNM.operator` exposes one `nlinv`
Gauss-Newton step as a single operator, differentiable with respect to its
data, iterate, regularization centre and regularization weight.  An unrolled
network is then itself one `nlop`: chaining the steps leaves BART driving the
whole graph, with one crossing into Python per step for the prior.

A tensor captured by closure inside a Python callback is not an input of the
operator, so no cotangent is propagated to it.  Pass such weights as explicit
operator arguments instead; {class}`Parameters` packs a module's parameters into
one argument for this purpose.

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

## User-defined operators

A function of several tensors becomes an `nlop` of several inputs, so a
denoiser's weights become arguments of a BART operator graph rather than values
captured by closure.
{meth}`NonlinearOperator.from_callbacks` takes the forward, the derivative and
the adjoint as functions; {class}`FromTorch` takes one differentiable function.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorch
   Parameters
```

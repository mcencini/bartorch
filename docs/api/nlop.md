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

## Composition

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
```

## Basic

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

## MRI encodings

The coils are a second unknown rather than a fixed tensor, which is what
`nlinv` inverts.

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

## Gauss-Newton

BART has Gauss-Newton in two forms and {class}`IRGNM` is both: without
`inner=` it is `irgnm`, run inside the library, and with one it is `irgnm2`,
whose linearized problem goes to any solver in {mod}`bartorch.optim`.

```python
nlop.IRGNM(inner=optim.CG())                               # iter4_irgnm2, to the bit
nlop.IRGNM(inner=optim.FISTA(priors.Wavelet(axes, 0.001)))   # moba -l1's shape
```

For a {class}`NonlinearSense`, {meth}`IRGNM.operator` is BART's own step of
`nlinv` as one operator, differentiable by its data, iterate, regularisation
centre and weight.  An unrolled network is then a *single* `nlop`: chain the
cells and BART drives the whole thing, crossing into Python once a step for
the prior.  A weight the prior *closes over* reaches no gradient; give it the
weights as arguments instead, which is what {class}`Parameters` packs.

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

## Python-defined

A function of several tensors becomes an `nlop` of several inputs, which is
what lets a denoiser's weights be *arguments* of a BART graph rather than
something the function closed over.
{meth}`NonlinearOperator.from_callbacks` takes the forward, the derivative and
the adjoint as functions; {class}`FromTorch` takes one differentiable function.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FromTorch
   Parameters
```

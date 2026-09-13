# Nonlinear operators

`bartorch.nlop`.  A nonlinear operator carries its derivative and the adjoint
of that derivative at the last evaluated point, which is what BART's
Gauss-Newton solver and torch's autograd both use.

An operator maps *many* inputs to *many* outputs, as BART's `nlop_s` does: the
model `nlinv` inverts takes an image and a set of coil profiles and returns
k-space.  {attr}`~bartorch.nlop.NonlinearOperator.ishapes` and
{attr}`~bartorch.nlop.NonlinearOperator.oshapes` give the shape of each
argument, and `ishape` and `oshape` are the whole of it when there is one of
each.  Arguments are counted BART's way throughout -- outputs first, then
inputs.

```{eval-rst}
.. currentmodule:: bartorch.nlop
```

## Nonlinear operator class

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonlinearOperator
```

## Composition

The algebra of `nlops/chain.h`.  {func}`combine` puts two operators side by
side and is where every other combination starts; {func}`chain` feeds one
output into one input; {meth}`~NonlinearOperator.link` ties an output of one
operator back into an input of another, and
{meth}`~NonlinearOperator.dup` makes two inputs one.

BART applies a combination back to front -- in `combine(a, b)` it is `b` that
runs first -- so an operator that produces a value goes *second*, and the one
that consumes it first.  {func}`chain` arranges that itself; a
{meth}`~NonlinearOperator.link` the other way round is refused rather than
left to read a buffer nothing has written.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   chain
   combine
   Chain
   FromLinear
   Derivative
```

## Basic operators

The tensor product and the elementwise maps, each one BART constructor.
{class}`Multiply` is the two-input product every model is built out of.

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

## Python-defined operators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Callback
   FromTorch
```

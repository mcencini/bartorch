# Regularization and denoising

`bartorch.prox`.  A regularization term is what a solver in
{mod}`bartorch.optim` takes; BART builds its proximal operator.  The denoisers
are functions on images.

```{eval-rst}
.. currentmodule:: bartorch.prox
```

## Regularizer class

What a solver computes is `prox(transform(x))`, and the shapes do not say
which of the two carries the work.  BART's own `iter2_ist` ignores the
transform, which is why BART's IST and FISTA cannot take a total-variation
term.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Regularizer
```

## In a solve that is being differentiated

BART's proximal operators carry no derivative, so
{meth}`Regularizer.prox` refuses a tensor that does; a denoiser is what goes
in that slot instead.  {func}`frozen` is for the mixed solve, where the
gradient is meant to reach the denoiser and another term is furniture.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   frozen
```

## Sparsity terms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   L1
   Wavelet
   FourierL1
   TotalVariation
   Laplace
   ImaginaryL1
```

## Quadratic terms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   L2
   ImaginaryL2
```

## Low rank

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   LocallyLowRank
```

## Constraints

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NonNegative
```

## Hard thresholding

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   WaveletNIHT
   ImageNIHT
```

## Terms that add unknowns

These split into several penalties over a variable larger than the image, so
one cannot be built alone.  {func}`bartorch.tools.pics`,
{class}`bartorch.optim.ADMM` and {class}`bartorch.optim.PRIDU` take them; the
solve then runs in the library and does not unroll.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   TotalGeneralizedVariation
   InfimalConvolutionTV
   InfimalConvolutionTGV
```

## Denoisers

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   rof
   tgv
   nlmeans
```

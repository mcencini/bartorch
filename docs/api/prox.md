# Regularization and denoising

`bartorch.prox`.  A regularization term is what a solver in
{mod}`bartorch.optim` takes; BART builds its proximal operator.  The denoisers
are functions on images.

```{eval-rst}
.. currentmodule:: bartorch.prox
```

## Regularizer class

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Regularizer
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

## Terms for pics only

{func}`bartorch.tools.pics` takes these; a solver in {mod}`bartorch.optim`
does not, because they add variables BART counts across the whole set of
terms.

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

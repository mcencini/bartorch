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

A term can also be applied on its own, which is what an iteration written
outside the library needs: `Regularizer.prox(x, gamma)` is the proximal
operator a solver calls between its gradient steps, `prox_shape` says what it
works on, and `transform` is the operator BART puts in front of it.

Two things the shapes do not say.  A term's proximal operator usually works on
the image, because the transform is folded inside it -- a wavelet term is like
this -- but total variation's is on the components of a gradient instead, an
axis of their own beyond the image's.  And the Laplace term's transform is a
real convolution whose codomain *is* shaped like the image, so a caller that
guessed from the shape would quietly leave it out.  What a solver computes is
`prox(transform(x))`; BART's own `iter2_ist` is the exception, applying the
proximal operator to the image and ignoring the transform, which is why BART's
IST and FISTA cannot take a total-variation term.

`transform_is_identity` asks BART the same question `iter2_chambolle_pock`
asks of the first term before making it the primal proximal step rather than a
dual, and `rewind` puts a term's own random generator back where a fresh term
would have it -- which is what a wavelet threshold's cycle spinning draws on,
and what makes a term kept across solves answer as the tool does.

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

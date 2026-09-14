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
outside the library needs.  What a solver computes is `prox(transform(x))`,
and the shapes do not say which of the two carries the work: a wavelet term
folds its transform inside the proximal operator, total variation's works on
the components of a gradient instead, and the Laplace term's transform is a
convolution whose codomain *is* shaped like the image.  So a caller that
guessed from the shapes would leave a transform out.  BART's own `iter2_ist`
does exactly that, which is why BART's IST and FISTA cannot take a
total-variation term.

## In a solve that is being differentiated

An iteration in {mod}`bartorch.optim` is a torch graph over BART's arithmetic,
so an unrolled network or a deep-equilibrium fixed point differentiates
through the operators in it -- including a term's transform, whose backward
pass is its transpose.  The proximal operator is where that stops.  BART's are
`operator_p_s`, and `operator_p_fun_t` is `(data, mu, dst, src)` with nowhere
for a derivative to live, so `Regularizer.prox` refuses a tensor that carries
one rather than contributing a wrong gradient: soft thresholding is not the
constant map, and treating it as one zeroes the whole path through the prior.

What goes in that slot instead is a denoiser, which differentiates on its own
terms.  {func}`frozen` is for the mixed solve -- a denoiser in one slot and a
term of BART's own in another -- where the gradient is meant to reach the
denoiser and the other term is furniture.

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

These split into several penalties over a variable larger than the image:
BART carries the supporting fields alongside it and counts them across the
whole set of terms, so a term cannot be built on its own the way the others
are.  {func}`bartorch.tools.pics` takes them, and so do
{class}`bartorch.optim.ADMM` and {class}`bartorch.optim.PRIDU` -- the two
iterations BART hands a term's transform to.  The solve then runs in the
library, where that larger vector is laid out, so these terms do not unroll
into a network and the other solvers refuse them.

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

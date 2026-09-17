# Regularization and denoising

`bartorch.priors`.  A regularizer specifies a regularization functional for the
solvers in {mod}`bartorch.optim`; {doc}`../explanation/inverse-problems`
introduces the functionals and their proximal operators.  BART builds the corresponding proximal
operator and, where the functional acts on a transform of the image, the linear
transform in front of it.  The denoisers are functions on images.

```{eval-rst}
.. currentmodule:: bartorch.priors
```

## Regularizers

A regularizer is a pair: a linear transform `G` and the proximal operator of a
functional `g`, together representing `g(G x)`.  `G` is the identity for terms
that penalize the image directly ({class}`L1`, {class}`L2`) or that carry their
own transform inside the proximal operator ({class}`Wavelet`), and a genuine
operator for terms such as {class}`TotalVariation`, whose functional acts on
finite differences.

Only the alternating-direction and primal-dual iterations are given `G`.
BART's `iter2_ist` takes a single term and ignores the transform array
entirely, so {class}`~bartorch.optim.IST` and {class}`~bartorch.optim.FISTA`
admit only terms whose transform is the identity; a term with a non-trivial
`G` would otherwise be applied as though `G` were the identity.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Regularizer
```

## Differentiation

Proximal operators are evaluated inside BART and no backward pass is
implemented for them, so {meth}`Regularizer.prox` raises on a tensor that
requires a gradient rather than contributing an incorrect one: soft
thresholding is not the identity, and differentiating as though it were zeroes
the whole path through the regularizer.

{class}`ImplicitPrior` substitutes a differentiable denoiser for the proximal
operator, as plug-and-play regularization does.  {func}`frozen` detaches a
BART term's proximal step, so that term stays fixed while the rest of the
iteration -- including a denoiser in another term -- is differentiated.

{class}`ImplicitPrior` accepts a `transform` like any other term, so that the
denoiser may be applied on a domain other than the image: the
alternating-direction and primal-dual iterations then split the variable at
`G x`, and the denoiser is applied on the codomain of `G`.
{class}`bartorch.learning.Denoiser` adapts a network operating on real planes
to that interface.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ImplicitPrior
   frozen
```

## Sparsity regularization

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

## Quadratic regularization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   L2
   ImaginaryL2
```

## Low-rank regularization

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

## Hard-thresholding terms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   WaveletNIHT
   ImageNIHT
```

## Regularizers with auxiliary variables

These functionals are defined by a minimization over auxiliary variables, and
BART realizes them by extending the optimization variable with those variables
(`ropts->svars`).  The extension is counted across the whole set of terms, so
such a term cannot be built in isolation.  {func}`bartorch.tools.pics`,
{class}`bartorch.optim.ADMM` and {class}`bartorch.optim.PRIDU` accept them; the
solution vector is the image followed by the auxiliary variables, and the image
alone is returned.

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

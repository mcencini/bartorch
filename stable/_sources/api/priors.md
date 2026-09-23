# Regularization and denoising

`bartorch.priors` provides the regularization terms of BART's reconstructions.
A term represents a functional $g(Gx)$: BART builds the proximal operator of
$g$ and the linear transform $G$, and the solvers of {mod}`bartorch.optim`
apply them.  An {obj}`~bartorch.priors.ImplicitPrior` takes the place of a term
with a denoiser, and three functions apply BART's denoisers to an image
directly.  {doc}`../explanation/inverse-problems` introduces functionals,
proximal operators and the splitting that a nontrivial $G$ requires.

```{eval-rst}
.. currentmodule:: bartorch.priors
```

| Term | Functional $g(Gx)$ | Transform $G$ | Solvers |
| --- | --- | --- | --- |
| {obj}`~bartorch.priors.L1` | $\lambda\lVert x\rVert_1$ | $I$ | IST, FISTA, ADMM, PRIDU |
| {obj}`~bartorch.priors.Wavelet` | $\lambda\lVert \Psi x\rVert_1$ | $I$; $\Psi$ is inside the proximal operator | IST, FISTA, ADMM, PRIDU |
| {obj}`~bartorch.priors.LocallyLowRank` | $\lambda\sum_b \lVert B_b x\rVert_*$ | $I$; the blocks are inside the proximal operator | IST, FISTA, ADMM, PRIDU |
| {obj}`~bartorch.priors.L2` | $\tfrac{\lambda}{2}\lVert x\rVert_2^2$ | $I$ | IST, FISTA, ADMM, PRIDU |
| {obj}`~bartorch.priors.NonNegative` | Indicator of $\operatorname{Re} x \ge 0$ and $\operatorname{Im} x \ge 0$ | $I$ | IST, FISTA, ADMM, PRIDU |
| {obj}`~bartorch.priors.TotalVariation` | $\lambda\sum_r \lVert (\nabla x)_r\rVert_2$ (isotropic) | Finite differences $\nabla$ | ADMM, PRIDU |
| {obj}`~bartorch.priors.FourierL1` | $\lambda\lVert F x\rVert_1$ | Fourier transform $F$ | ADMM, PRIDU |
| {obj}`~bartorch.priors.Laplace` | $\lambda\lVert L x\rVert_1$ | Laplacian $L$ | ADMM, PRIDU |
| {obj}`~bartorch.priors.ImaginaryL1` | $\lambda\lVert \operatorname{Im} x\rVert_1$ | Imaginary part | ADMM, PRIDU |
| {obj}`~bartorch.priors.ImaginaryL2` | $\tfrac{\lambda}{2}\lVert \operatorname{Im} x\rVert_2^2$ | Imaginary part | ADMM, PRIDU |
| {obj}`~bartorch.priors.TotalGeneralizedVariation` | Second-order TGV | Extends the variable | ADMM, PRIDU |
| {obj}`~bartorch.priors.InfimalConvolutionTV` | Infimal convolution of two TV terms | Extends the variable | ADMM, PRIDU |
| {obj}`~bartorch.priors.InfimalConvolutionTGV` | Infimal convolution of two TGV terms | Extends the variable | ADMM, PRIDU |
| {obj}`~bartorch.priors.WaveletNIHT` | Keep the $K$ largest wavelet coefficients | Wavelet transform | NIHT |
| {obj}`~bartorch.priors.ImageNIHT` | Keep the $K$ largest image entries | $I$ | NIHT |

$\lambda$ is the term's `weight`, relative to data divided by
{func}`~bartorch.optim.data_scaling`; $\lVert\cdot\rVert_1$ of a complex array
is the sum of the moduli, and a term's `joint_axes` group entries into an
$\ell_2$ norm first.  Every term is also accepted by
{func}`bartorch.tools.pics` and {func}`bartorch.apps.pics`.

## Term classes

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.Regularizer` | Base class: the proximal operator and the transform of a BART term |
| {obj}`~bartorch.priors.ImplicitPrior` | A denoiser in place of a proximal operator (plug-and-play), optionally through a transform |
| {obj}`~bartorch.priors.frozen` | A term whose proximal step is excluded from differentiation |

## Sparsity-promoting terms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.L1` | $\ell_1$ norm of the image |
| {obj}`~bartorch.priors.Wavelet` | $\ell_1$ norm of the wavelet coefficients |
| {obj}`~bartorch.priors.FourierL1` | $\ell_1$ norm of the Fourier coefficients |
| {obj}`~bartorch.priors.TotalVariation` | Total variation |
| {obj}`~bartorch.priors.Laplace` | $\ell_1$ norm of the Laplacian |
| {obj}`~bartorch.priors.ImaginaryL1` | $\ell_1$ norm of the imaginary part |

## Quadratic terms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.L2` | Squared $\ell_2$ norm of the image (Tikhonov) |
| {obj}`~bartorch.priors.ImaginaryL2` | Squared $\ell_2$ norm of the imaginary part |

## Low-rank terms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.LocallyLowRank` | Nuclear norm of image blocks |

## Constraints

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.NonNegative` | Projection clamping the real and imaginary parts at zero |

## Hard-thresholding terms

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.WaveletNIHT` | Retention of the largest wavelet coefficients |
| {obj}`~bartorch.priors.ImageNIHT` | Retention of the largest image entries |

## Terms with auxiliary variables

The optimization variable is the image followed by auxiliary fields, which BART
counts across the whole set of terms; these terms cannot be built alone, and
the solvers return the image only.

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.TotalGeneralizedVariation` | Second-order total generalized variation |
| {obj}`~bartorch.priors.InfimalConvolutionTV` | Infimal convolution of total variation |
| {obj}`~bartorch.priors.InfimalConvolutionTGV` | Infimal convolution of total generalized variation |

## Denoisers

| Object | Description |
| --- | --- |
| {obj}`~bartorch.priors.rof` | Total-variation (Rudin-Osher-Fatemi) denoising |
| {obj}`~bartorch.priors.tgv` | Second-order total generalized variation denoising |
| {obj}`~bartorch.priors.nlmeans` | Non-local means filtering |

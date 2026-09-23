# Nonlinear forward models

```{admonition} TL;DR
:class: tldr

- Joint estimation of the image and the coil sensitivities (`nlinv`) and model-based parameter estimation (`moba`) have forward operators that are nonlinear in the unknowns.
- Both are solved by iteratively regularized Gauss-Newton: each step solves a linearized least-squares problem with a Tikhonov term whose weight decreases geometrically, and the number of steps acts as a regularization parameter.
- `nlinv` resolves the ambiguity between image and sensitivities with a Sobolev weighting of the coils; the data term of a signal model is nonconvex in its parameters, so the result depends on the starting point.
- A model-based reconstruction estimates parameter maps directly; a subspace reconstruction keeps the forward model linear and fits the parameters afterwards.
```

{doc}`inverse-problems` assumes a known, linear forward operator.  Two common
MRI reconstructions do not satisfy that assumption: the coil sensitivities can
be unknowns alongside the image, and the image can be a function of physical
parameters.  Both lead to a nonlinear operator $F$ and are solved by the same
Gauss-Newton method.

## Two nonlinear models

**Joint estimation of image and sensitivities.**  When the acquisition has no
fully sampled calibration region, or the sensitivities are to be estimated
from all of the data, both are unknowns:

$$
y_c = P F \left( S_c \cdot x \right).
$$

The operator is bilinear: linear in $x$ for fixed $S$, linear in $S$ for fixed
$x$, and nonlinear in the pair.  This is **nonlinear inversion**, BART's
`nlinv`.[^nlinv]

**Model-based parameter estimation.**  A quantitative experiment acquires a
series of contrasts whose dependence on tissue parameters $\theta$ is given by
a signal model $M$ — a mono-exponential decay in $T_2$, an inversion recovery
in $T_1$, or a Bloch simulation of the sequence:

$$
y_{c,e} = P_e F \left( S_c \cdot M_e(\theta) \right),
$$

with $e$ the contrast index.  The unknowns are the parameter maps.  This is
**model-based reconstruction**, BART's `moba`.[^block][^sumpf][^wang]

A Gauss-Newton solver requires the value $F(x)$, the derivative $DF_x$ at a
point as a linear operator, and its adjoint $DF_x^H$.  A
{class}`~bartorch.nlop.NonlinearOperator` provides the three, and
`F.linearize(x)` returns $DF_x$ as a {class}`~bartorch.linop.LinearOperator`.

## Iteratively regularized Gauss-Newton

Each Gauss-Newton step replaces $F$ by its linearization at the current iterate
$x_k$ and solves the resulting linear problem with a Tikhonov term centred on a
reference $x_{\mathrm{ref}}$:

$$
x_{k+1} = \arg\min_x \;
\left\lVert DF_{x_k}(x - x_k) - \left(y - F(x_k)\right) \right\rVert_2^2
+ \alpha_k \lVert x - x_{\mathrm{ref}} \rVert_2^2 ,
$$

that is,
$x_{k+1} = x_k + (DF^H DF + \alpha_k I)^{-1}\left[DF^H (y - F(x_k)) - \alpha_k (x_k - x_{\mathrm{ref}})\right]$
with $DF = DF_{x_k}$.  The weight decreases geometrically,
$\alpha_{k+1} = (\alpha_k - \alpha_{\min})/q + \alpha_{\min}$ with $q = 2$ by
default.[^bakushinsky]  The early steps are strongly regularized and change
the iterate only along well-determined directions; later steps admit the
poorly determined ones.  The number of steps is a regularization parameter:
stopping early gives a smoother estimate, and continuing eventually fits the
noise.

{class}`~bartorch.nlop.IRGNM` implements this loop and
{class}`~bartorch.nlop.IRGNMBlock` one step of it.  Without an inner solver
the linear problem is solved by conjugate gradients inside BART, as in
`nlinv`; with `inner=` it is passed to a solver of {mod}`bartorch.optim`, and
that solver's regularization terms add a penalty to the step.

## Identifiability

The bilinear model has a symmetry: $S_c \cdot x$ is unchanged by
$S_c \mapsto \gamma S_c$ and $x \mapsto x / \gamma$ for any nonzero function
$\gamma(r)$, so the data do not determine how structure is shared between the
two factors.  `nlinv` resolves this with the prior knowledge that
sensitivities are smooth, built into the model: the coil unknown is a k-space
representation $\hat{s}$, and the sensitivities are

$$
S = \mathcal{F}^{-1}\!\left[ (1 + a \lvert k \rvert^2)^{-b/2}\, \hat{s} \right],
$$

a **Sobolev weighting** that attenuates the high spatial frequencies of the
coil estimate.  The IRGNM penalty on $\hat{s}$ is then a penalty on the
Sobolev norm of $S$, and the image receives the high-frequency structure.  The
product $x \cdot \sqrt{\sum_c \lvert S_c \rvert^2}$ is invariant to $\gamma$
up to its phase, which is why `nlinv` reports the image multiplied by the
root sum of squares of the estimated sensitivities.
{class}`~bartorch.nlop.NonlinearSense` is this model;
{func}`~bartorch.nlop.CoilSense` is the unweighted product in front of any
linear encoding, which leaves the regularization of the coils to the caller.

Signal models have no such symmetry — the model fixes the meaning of each
map — but the data term is nonconvex in $\theta$, so the result depends on the
starting point.  The models of {mod}`bartorch.nlop` impose bounds by solving
for a transformed variable, so that every iterate stays within the bounds;
{meth}`~bartorch.nlop.SignalModel.initial` builds a starting point from
parameter values and {meth}`~bartorch.nlop.SignalModel.split` converts a
solution back to named maps in physical units.

## Approaches to parameter mapping

| Approach | Unknown | Forward model | Cross-contrast information in the reconstruction | Problem | Output |
| --- | --- | --- | --- | --- | --- |
| Reconstruction, then voxel-wise fit | One image per contrast | Linear, $P_e F S$ | Only through a joint regularizer, if one is used; none if the contrasts are reconstructed separately | Linear reconstruction, then a nonlinear fit per voxel | Contrast images, then parameter maps |
| Subspace reconstruction | Coefficient maps of a low-dimensional basis $\Phi$ | Linear, $P_e F S\, \Phi$ | The signal evolution is restricted to the span of $\Phi$[^huang][^tamir] | Linear; convex with a convex regularizer | Coefficient maps, then parameter maps by dictionary matching or fitting |
| Model-based reconstruction | Parameter maps $\theta$ | Nonlinear, $P_e F S\, M(\theta)$ | The signal model couples every contrast to the same parameters | Nonlinear, nonconvex | Parameter maps |

A model-based reconstruction estimates fewer unknowns from the same data —
for a multi-echo experiment, a few maps rather than one image per echo — and
applies regularization to the maps.  This improves the estimate when the
model describes the signal; a signal the model does not describe, such as
partial volume of two tissues or an imperfect refocusing, becomes a bias in
the maps.  A subspace reconstruction restricts the signal to a basis derived
from a dictionary of simulated signals, keeps a linear forward model, and
defers the nonlinear estimation of the parameters to a separate step.
{doc}`../auto_examples/03-applications/02-subspace-t1-mapping` and
{doc}`../auto_examples/04-model-based/02-quantitative-models` apply the
second and third routes.

## Signal models and their derivatives

The signal models — {func}`~bartorch.nlop.InversionRecovery`,
{func}`~bartorch.nlop.MultiEcho`, {func}`~bartorch.nlop.Bloch` — are
[TorchSim](https://github.com/FiRMLAB-Pisa/torchsim) simulators presented as
BART nonlinear operators.  A signal model is voxel-wise: the signal of a voxel
depends only on that voxel's parameters, so one forward-mode automatic
differentiation pass gives the derivative for the whole volume and no Jacobian
matrix is formed.  {class}`~bartorch.nlop.TorchOperator` does the same for any
differentiable PyTorch function.  Composition with `@` places a model in front
of an encoding; the derivative of `E @ M` at $\theta$ is $E\, DM_\theta$.
{doc}`differentiation` describes how gradients pass through the Gauss-Newton
steps themselves.

## References

[^nlinv]: Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear inversion—joint estimation of coil sensitivities and image content. *Magn Reson Med* 60(3):674–682 (2008). [doi:10.1002/mrm.21691](https://doi.org/10.1002/mrm.21691)

[^block]: Block KT, Uecker M, Frahm J. Model-based iterative reconstruction for radial fast spin-echo MRI. *IEEE Trans Med Imaging* 28(11):1759–1769 (2009). [doi:10.1109/TMI.2009.2023119](https://doi.org/10.1109/TMI.2009.2023119)

[^sumpf]: Sumpf TJ, Uecker M, Boretius S, Frahm J. Model-based nonlinear inverse reconstruction for T2 mapping using highly undersampled spin-echo MRI. *J Magn Reson Imaging* 34(2):420–428 (2011). [doi:10.1002/jmri.22634](https://doi.org/10.1002/jmri.22634)

[^wang]: Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. Physics-based reconstruction methods for magnetic resonance imaging. *Phil Trans R Soc A* 379(2200):20200196 (2021). [doi:10.1098/rsta.2020.0196](https://doi.org/10.1098/rsta.2020.0196)

[^bakushinsky]: Bakushinsky AB, Kokurin MY. *Iterative Methods for Approximate Solution of Inverse Problems.* Springer (2004). [doi:10.1007/978-1-4020-3122-9](https://doi.org/10.1007/978-1-4020-3122-9)

[^huang]: Huang C, Graff CG, Clarkson EW, Bilgin A, Altbach MI. T2 mapping from highly undersampled data by reconstruction of principal component coefficient maps using compressed sensing. *Magn Reson Med* 67(5):1355–1366 (2012). [doi:10.1002/mrm.23128](https://doi.org/10.1002/mrm.23128)

[^tamir]: Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2 shuffling: sharp, multicontrast, volumetric fast spin-echo imaging. *Magn Reson Med* 77(1):180–195 (2017). [doi:10.1002/mrm.26102](https://doi.org/10.1002/mrm.26102)

# Differentiation through reconstruction

```{admonition} TL;DR
:class: tldr

- The backward pass of a linear operator is its adjoint, $A^H g$, and that of a nonlinear operator is $DF_x^H g$.
- Conjugate gradients are differentiated implicitly, at constant memory; the proximal iterations and ADMM are unrolled, with memory linear in the iteration count.
- BART's proximal operators have no backward pass: {class}`~bartorch.priors.ImplicitPrior` replaces one with a differentiable denoiser, and {func}`~bartorch.priors.frozen` holds one fixed.
- {class}`~bartorch.learning.Unrolled` applies a fixed number of iteration blocks, differentiated end to end, per iteration, or with checkpointing; {class}`~bartorch.optim.FixedPoint` differentiates at a fixed point with constant memory.
```

A reconstruction written with operators and solvers can be part of a larger
PyTorch computation: a loss on the reconstructed image can be differentiated
with respect to the data, to learned parameters of the iteration, or to the
point at which a nonlinear model is linearized.  This page states how each
object computes its backward pass, which parts are excluded from
differentiation, and how memory grows with the number of iterations.  The
functions of {mod}`bartorch.tools` record nothing for autograd.

## Operators

For a complex tensor PyTorch propagates the conjugate Wirtinger gradient, so
the vector–Jacobian product of $y = Ax$ with incoming gradient $g$ is
$A^H g$, not $A^T g$.  Applying a {class}`~bartorch.linop.LinearOperator` to a
tensor that requires a gradient records the application with the adjoint as
its backward pass; the test suite compares this with PyTorch's own gradient of
a matrix multiplication and checks that the transpose would differ.

For a {class}`~bartorch.nlop.NonlinearOperator` the backward pass at $x$ is
$DF_x^H g$, the adjoint of the derivative at the evaluated point.
`F.linearize(x)` returns $DF_x$ as a linear operator that holds $x$: applying
it is differentiable with respect to its argument and to $x$.

## Solvers and iterations

| Object | Forward computation | Backward pass | Memory in the iteration count |
| --- | --- | --- | --- |
| {class}`~bartorch.optim.CG` | BART's conjugate gradients, inside the library | Implicit: one further CG solve with the same normal operator, then one application of $A$ | Constant |
| {class}`~bartorch.optim.IST`, {class}`~bartorch.optim.FISTA`, {class}`~bartorch.optim.PRIDU`, {class}`~bartorch.optim.POCS` | The iteration block applied `maxiter` times in Python | Unrolled: every step is recorded | Linear |
| {class}`~bartorch.optim.ADMM` | As above | Unrolled; each x-update, a CG solve, is differentiated implicitly by one further solve | Linear |
| {class}`~bartorch.nlop.IRGNM`, {class}`~bartorch.nlop.IRGNMBlock` without `inner` | Gauss-Newton steps; the inverse in each is BART's `norm_inv` | Each step recorded; the inverse differentiated implicitly with respect to the data, the iterate, $x_{\mathrm{ref}}$ and $\alpha$ | Linear in the steps |
| {class}`~bartorch.nlop.IRGNM`, {class}`~bartorch.nlop.IRGNMBlock` with `inner` | The inner solver on the linearized problem | That solver's route, including the dependence of $DF_x$ on the iterate | Linear in the steps |
| {class}`~bartorch.learning.Unrolled` | A fixed number of iteration blocks | See the table below | See the table below |
| {class}`~bartorch.optim.FixedPoint` | A block iterated to its fixed point, without recording | Implicit differentiation at the fixed point | Constant |

**Implicit differentiation of a linear solve.**  With
$N = A^H A + \lambda I$, the solution $x = N^{-1} A^H y$ is linear in $y$,
and its vector–Jacobian product is $A N^{-1} g$.  The backward pass is exact
for the solution of the normal equations; for a solve truncated by `maxiter`
or `tol` it is the gradient of that solution, not of the returned iterate, and
the backward solve is itself truncated.  The starting point receives no
gradient, and neither do the data held by the operator — sensitivities,
trajectory, a term's weight — except through a linearized nonlinear operator,
where the dependence on the linearization point is included.  No second
derivatives are available.

**Explicit unrolling** records every operation of every iteration.  The
backward pass of an operator application is a further application of the
adjoint and stores nothing beyond its input, so the stored graph is dominated
by the activations of any network in the iteration.

## Terms excluded from differentiation

BART's proximal operators have no backward pass.
{meth}`Regularizer.prox <bartorch.priors.Regularizer.prox>` raises an error
for an input that requires a gradient rather than contribute a wrong one, so a
solver whose term is a BART regularizer cannot be differentiated through that
term.  The following objects permit differentiation:

| Object | Effect |
| --- | --- |
| {class}`~bartorch.priors.ImplicitPrior` | Replaces the proximal operator by a differentiable denoiser, as in plug-and-play reconstruction;[^pnp] the denoiser's parameters receive gradients |
| {func}`~bartorch.priors.frozen` | Applies a BART term's proximal operator to a detached input: the term acts in the forward pass and is held fixed in the backward pass, for a solve in which another term is learned |

The residual norms that drive adaptive steps — an adaptive ADMM penalty, an
adaptive primal-dual step size — are detached: they control the iteration and
contribute no gradient.

## Unrolled networks

{class}`~bartorch.learning.Unrolled` applies an iteration block a fixed number
of times.  With all parameters frozen it reproduces the corresponding solver;
calling `requires_grad_()` on a step size, a regularization weight or a denoiser's
parameters makes it a trainable network, such as MoDL.[^modl]

| Setting | Recorded graph | Memory | Gradient |
| --- | --- | --- | --- |
| default | Every iteration | Proportional to the iteration count | End to end |
| `detach=True` | One iteration; each starts from a detached state | One iteration | Of each iteration alone; with a loss on each image from {meth}`~bartorch.learning.Unrolled.steps`, greedy per-iteration training |
| `checkpoint=True` | The states between iterations; each iteration is recomputed in the backward pass | The states plus one iteration | End to end, at the cost of a second forward pass per iteration |

Checkpointing assumes that recomputing a step reproduces it.  A term that draws
random shifts, such as a wavelet term with cycle spinning, draws new shifts on
recomputation, so checkpointing is valid only for deterministic steps.
Pretraining a denoiser, then training greedily per iteration, then fine-tuning
end to end with checkpointing is a staged procedure reported for fully
three-dimensional unrolled reconstruction.[^urman]

## Fixed-point differentiation

{class}`~bartorch.optim.FixedPoint` iterates a block $z \mapsto \Phi(z)$
without recording until the relative change is below a tolerance, and
differentiates the fixed point $z^\star = \Phi(z^\star)$ implicitly — a deep
equilibrium model.[^deq][^gilton]  The vector–Jacobian product solves
$w = J^H w + g$, with $J$ the Jacobian of one step at $z^\star$, by the
fixed-point iteration $w \leftarrow J^H w + g$, which converges when the
spectral radius of $J$ is below one.  Memory is that of a single step,
independent of the iteration count.  An iteration that does not apply the same
map $\Phi$ at every step is not a fixed-point iteration and is refused: FISTA's
momentum, a moving ADMM penalty, adaptive or decaying primal-dual steps.

## References

[^pnp]: Venkatakrishnan SV, Bouman CA, Wohlberg B. Plug-and-play priors for model based reconstruction. *IEEE Global Conference on Signal and Information Processing* 945–948 (2013). [doi:10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048)

[^modl]: Aggarwal HK, Mani MP, Jacob M. MoDL: model-based deep learning architecture for inverse problems. *IEEE Trans Med Imaging* 38(2):394–405 (2019). [doi:10.1109/TMI.2018.2865356](https://doi.org/10.1109/TMI.2018.2865356)

[^urman]: Urman Y, Nishimura M, Abraham DR, Cao X, Setsompop K. Fully 3D unrolled magnetic resonance fingerprinting reconstruction via staged pretraining and implicit gridding. *Magn Reson Med* 96(5):2516–2529 (2026). [doi:10.1002/mrm.70500](https://doi.org/10.1002/mrm.70500)

[^deq]: Bai S, Kolter JZ, Koltun V. Deep equilibrium models. *Advances in Neural Information Processing Systems* 32:688–699 (2019). [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)

[^gilton]: Gilton D, Ongie G, Willett R. Deep equilibrium architectures for inverse problems in imaging. *IEEE Trans Comput Imaging* 7:1123–1133 (2021). [doi:10.1109/TCI.2021.3118944](https://doi.org/10.1109/TCI.2021.3118944)

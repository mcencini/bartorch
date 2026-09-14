# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration is BART's, the one `pics` or `nlinv`
runs.

Where the iteration has been written out in `bartorch.optim.iterators` --
`IST`, `FISTA`, `ADMM` and `PRIDU` -- the loop runs here and each step calls
the library; `CG`, `NIHT` and `EulerMaruyama` are one call into it.  Either
way the answer is the same bits, and `solver.in_library(y, A)` runs BART's own
loop when that is what is wanted.  A term that adds unknowns to the
optimization is the one exception: `ADMM` and `PRIDU` take those, but the
larger vector is laid out inside the library, so the solve goes there.

```{eval-rst}
.. currentmodule:: bartorch.optim
```

## Linear least squares

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CG
   Tikhonov
```

`CG` minimizes $\|Ax-y\|^2 + \lambda\|x\|^2 + \sum_i w_i\|G_ix-b_i\|^2$.
The first two are BART's own: `iter2_conjgrad` takes one weight and asserts it
is handed no regularizing operators and no biases, and `lsqr2_create` builds
$A^HA+\lambda I$.  The rest are `Tikhonov` terms, which are not passed to the
iteration at all -- they are stacked under the encoding, so that what BART
solves is still an ordinary least-squares problem.

Stacking builds the normal operator rather than letting BART derive it, as
$A^HA+\sum_i w_iG_i^HG_i$ over each operator's own normal.  A Toeplitz
encoding therefore stays one inside a regularized solve: `A.gram()` is the
point-spread convolution, not the transform and its adjoint.

## Regularized least squares

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IST
   FISTA
   ADMM
   PRIDU
   NIHT
   EulerMaruyama
   maxeigen
```

## The same solvers as functions

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ist
   fista
   admm
   pridu
   cg
```

`optim.FISTA(term, maxiter=30)(y, A)` is a solver configured and then called,
which is the shape to keep when the same configuration is used twice -- inside
a training loop, say.  For a reconstruction run once, the function says the
same thing in one expression:

```python
x = optim.fista(y, A, term, maxiter=30)
```

They take one more kind of argument than the classes do, which is the point of
having them: a `deepinv` prior or a bare denoiser goes wherever a
{mod}`bartorch.prox` term goes, with `g_param` as its own parameter -- a
denoiser's noise level, say.

```python
from deepinv.models import DRUNet, to_complex_denoiser

x = optim.admm(y, A, to_complex_denoiser(DRUNet()), g_param=0.03)
```

What runs is the iteration in `bartorch.optim.iterators`, which is BART's step
for step, so a plug-and-play reconstruction is BART's ADMM with the threshold
replaced rather than a second implementation of it.  A denoiser and a term of
BART's own can go in the same alternating-direction solve, each with its own
split variable.  Two things to know: an image here is complex and most
denoisers are not, so `to_complex_denoiser` is usually needed; and there is no
library route for a solver holding one -- BART has no way to be handed a
denoiser, and `in_library` says so rather than running something else.

## Iterations for unfolding and fixed points

`bartorch.optim.iterators`.  BART's proximal iterations written as
`deepinv.optim.optim_iterators.OptimIterator` classes: the same arithmetic in
the same order, with every operator still the library's, held against the
library's own answer to the bit at every iteration count.

What that buys is the shape.  An iteration running inside the library cannot
be unrolled into a network or driven to a fixed point, because there is
nothing to differentiate through; one written out can be, and `optim_builder`
takes these straight into `BaseOptim` with `unfold=True` or `DEQ`.  The cost
is a few axpys on an image per step, which is nothing beside a transform.

### The solvers as networks

`unrolled` builds a network of `maxiter` steps and `fixed_point` a
deep-equilibrium model of the same step, each a `torch.nn.Module` taking
`(y, physics)`:

```python
net = optim.FISTA(denoiser, maxiter=10, step=0.9).unrolled(
    (1, 256, 256), trainable=["stepsize"]
)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
image = net(kspace[None], bartorch.to_deepinv(A))
```

A network is not built around an encoding -- it is handed one per call -- so
the one thing it has to be told is the shape its terms are configured for.
`trainable` names the parameters to learn, one value per step: `"stepsize"`
for the proximal-gradient solvers, `"rho"` for the alternating directions,
`"sigma"` and `"tau"` for the primal-dual. `net.parameters()` reaches those
*and* the weights of whatever stands where a term goes, which is what there is
to train.

With nothing trainable in it a network answers with the solver's bits, batch
or no batch -- so what an unrolled network here is, is BART's iteration with a
denoiser in the threshold's place, and not an architecture that resembles one.
Two things make that hold. It starts at zero, where BART starts, rather than
at $A^Hy$ where `deepinv` starts an optimizer; `custom_init=None` restores the
other start, which is usually what a network about to be trained wants. And a
batch is the solver run once per item, the conjugate gradients inside an
alternating-direction step included -- one solve over the stack would be the
same operator, since it is block diagonal, but not the same iteration, because
it would stop on the whole stack's residual and let the items steer each
other's stopping.

A *learned* parameter is a tensor, and an iteration with one in it works its
scalars out in single precision throughout rather than in a double rounded at
the end. So a trained network does not answer with the library's bits, and
could not: the numbers in it are no longer the library's.

Two solvers refuse a fixed point, because a step has to be the same map every
time for one to mean anything. FISTA's momentum carries the iteration number,
`t <- (p + sqrt(q + r t^2)) / 2`; `optim.IST` is the same iteration without the
ravine and takes `fixed_point`. And an alternating-direction step's fixed
point is in $(x, z, u)$ rather than in the image: its x-update reaches the
previous image only as the warm start of the inner solve, which carries no
gradient, so a model built on the image alone would be differentiating a map
that does not depend on its argument.

### What differentiates, and what does not

Every operator a step applies is recorded, so an unrolled iteration is a torch
graph over BART's arithmetic: the encoding, its normal operator
(`LinearOperator.A_adjoint_A`, the recording counterpart of `normal`), the
transform in front of a term, and the linear solve in ADMM's x-update.  Put a
denoiser where a term goes and the gradient reaches whatever is inside it:

```python
x = optim.admm(y, A, my_network, maxiter=10, cg_maxiter=8)
x.abs().square().sum().backward()      # reaches my_network's parameters
```

Two things are deliberately not in the graph.

BART's proximal operators have no derivative to give -- `operator_p_fun_t` is
`(data, mu, dst, src)`, with nowhere for one to live -- so
`Regularizer.prox` refuses a tensor that carries a gradient rather than
contributing the gradient of the constant map, which is not what soft
thresholding is.  {func}`bartorch.prox.frozen` says that a constant is what
was meant, for the mixed solve where a denoiser is in one slot and a term of
BART's own is furniture in another.

And the residual norms that steer $\rho$, $\tau$ and the stopping test are
read as numbers.  They are BART's schedule for the iteration rather than part
of the model it solves, and an unrolled network differentiates through the
iterate, not through the schedule that steered it.

`ADMMIteration`'s x-update differentiates by implicit differentiation rather
than by unrolling the conjugate gradients inside it: $x=N^{-1}A^Hy$ is linear
in $y$, so the backward pass is one more solve with the same operator.  That
is what BART does for the one solver it made an `nlop` -- `norm_inv_der_src`
and `norm_inv_adj_src` in `src/nlops/norm_inv.c` each run a solve of their own
-- and it means a truncated forward pass gets the converged derivative, and a
warm start carries no gradient because the solution of a linear system does
not depend on where the iteration began.  `optim.CG` is differentiable for the
same reason and on its own, which is the data-consistency layer of a MoDL.

```{eval-rst}
.. currentmodule:: bartorch.optim.iterators

.. autosummary::
   :toctree: generated
   :nosignatures:

   ISTIteration
   FISTAIteration
   ADMMIteration
   PRIDUIteration
   NormalEquations
   TermPrior
   AsTerm
```

`NormalEquations` differentiates $\tfrac12\|Ax-y\|^2$ as $A^H A x - A^H y$
rather than $A^H (A x - y)$, so a Toeplitz encoding answers with its point-spread
convolution instead of a transform and its adjoint.  $A^Hy$ is computed once
per solve: BART's gridding reduces in whatever order its threads finish, so
two adjoints of the same data differ in the last bits, and computing it once
is what makes the iteration repeatable as well as quick.

`ADMMIteration` is the one that takes several terms, each with its own
transform and bias, because that is what BART's ADMM solves.  Its x-update is
BART's conjugate gradients on $A^HA+\rho\sum_j G_j^HG_j$, warm-started, so the
encoding keeps its own normal there too.  Total variation reaches it through
`Regularizer.apply_transform`: its gradient puts the components on an axis
past BART's sixteen, so there is no operator to hand over, and the transform is
applied in place instead.

Its x-update sums the terms first, each scaled by $\rho$ as it goes, and adds
the encoding's normal last.  That is `admm_normaleq`'s order and not the
obvious one; with a single term the two orders are the same two numbers added
up, and with two they differ in the last place of the answer.

`maxiter` there is two limits rather than one, which is worth knowing because
`maxiter=30` does not mean thirty steps.  `admm`'s loop runs at most `maxiter`
times *and* breaks when `nr_invokes > maxiter`, where `nr_invokes` counts
conjugate-gradient iterations across the whole run -- so thirty with ten inner
iterations is about five outer steps, and on a well-conditioned problem, where
the inner solve takes one iteration a step, it really is thirty.  The
iterations are counted in C and handed back: `CG.__call__` takes a `steps`
list, which is where `ADMMIteration` gets the number, and nothing outside the
library could have worked it out.

One term is not a function of its argument.  A `prox.Wavelet` with
`randshift` on -- BART's default, and `pics`'s -- shifts its transform by a
draw of BART's own before every application, so the same term applied twice to
the same image answers twice differently.  Two loops therefore agree to the
bit only when they apply it the same number of times in the same order, which
is one more thing the comparison against the library checks.

### What `optim.ADMM` takes, and where it came from

`admm.c` has more in it than `pics` has flags for, and more than the wrapper
used to pass on.  All of it is reachable now, in two groups.

What `italgo_config` can be told -- `dynamic_rho`, `dynamic_tau`,
`relative_norm`, `fast` -- goes through `bartorch_solve` as well, so
`in_library` answers with the same bits.  `dynamic_rho` moves the penalty with
the residuals and rescales the dual variables to match; `dynamic_tau` chooses
how far by `sqrt(r / s)`, clipped to `[1 / tau_max, tau_max]`; `relative_norm`
compares each residual to its own scaling first.  Those three together are the
residual balancing of Wohlberg (2017).

What it cannot be told -- `alpha`, `mu`, `tau_max`, `abstol`, `reltol`, a
`bias` per term, and `cg_maxiter_first` -- is reachable only from the
iteration written here, and `in_library` refuses it rather than dropping it
quietly.  `abstol` and `reltol` are worth a word: `iter_admm_defaults` carries
Boyd's 1e-4 and 1e-3, and `italgo_config` overwrites both with zero, so
`pics`'s ADMM never stops on its residuals at all -- the budget is what stops
it.

The comparison against [riesling](https://github.com/spinicist/riesling)'s
ADMM, which this was asked to make, comes out in BART's favour almost
throughout.  Riesling has the residual balancing, the over-relaxation and a
combined tolerance; BART has all of that plus biases, Boyd's absolute and
relative tolerances separately, `dynamic_tau` as a setting of its own,
hogwild, a warm start, and a budget counted in applications of the normal
operator rather than outer steps.  Two things differ in riesling's favour: its
x-update is LSMR with a preconditioner rather than conjugate gradients --
deliberately not followed here, because a Toeplitz normal is the point of this
package's encodings -- and `iters0`, a separate budget for the first outer
step, where there is no warm start to build on.  That one is worth having, so
`cg_maxiter_first` is it, and it is the only setting in `optim.ADMM` that is
nobody's but riesling's.
`PRIDUIteration` is the one iteration whose answer depends on how BART was
compiled.  `vecops.c` has a single kernel behind `axpy`, `xpay` and `axpbz`,
`dst[i] = a1 * src1[i] + a2 * src2[i]`, and clang folds the first product into
the add where the hardware has a fused multiply-add -- arm64 does, the x86-64
baseline does not -- which is one rounding where this package computes two.
The other three iterations escape it because their updates are `axpy`, whose
`a1` is one, and folding an exact product in changes nothing; the data term's
resolvent here is an `xpay` and an `axpbz` with two real coefficients.  So on
arm64 this iteration is within a few times $10^{-7}$ of the library rather
than the same bits, and the tests ask the library which arithmetic it was
compiled with rather than assuming.

`PRIDUIteration` carries the data term as a dual of its own rather than
differentiating it, and gives each regularization term a dual too -- except
the first, when its transform is the identity, which becomes the primal
proximal step instead.  That split is `iter2_chambolle_pock`'s, reproduced
rather than chosen.  Its tolerance is absolute, unlike every other iteration
here: `iter2_chambolle_pock` leaves `eps` at one where the others scale it by
the norm of $A^Hy$.

`maxeigen` is the estimate `pics -e` divides the step by: a power iteration
over the encoding's normal with the quadratic weight on its diagonal, and --
for the primal-dual iteration alone -- the dual terms' transforms added to it.
It starts from a random vector, so it is a draw rather than a number: two
solves with `eigen=True` do not agree to the bit, in this package or in BART.

Reading BART's arithmetic off its source is most of the work in these
iterations, and two habits account for nearly all of it.  Every scalar is a C
`float` unless the library declares a `double`, and a scalar worked out in a
double and rounded once at the end is a different number -- which is what made
FISTA's ravine diverge at the thirteenth iteration.  And a vector is scaled by
a coefficient rather than divided by its reciprocal: `chambolle_pock` works
out `1 / sigma`, `1 / (1 + sigma)` and `-sigma / (1 + sigma)` once, in a
double, and rounds each to a float.  Dividing the tensor instead agrees while
`sigma` is where `pics` starts it, and stops agreeing once the adaptive step
has moved it.

`TermPrior` puts a {mod}`bartorch.prox` term where `deepinv` expects a prior;
`AsTerm` is the other direction, a prior or a bare denoiser where a term goes.
Either way a complex image needs `to_complex_denoiser` around a denoiser that
is not complex-capable, which neither wrapper does for you.

```{eval-rst}
.. currentmodule:: bartorch.optim
```

## Preconditioning

BART's `conjgrad` has **no preconditioner argument at all**.  What BART calls
preconditioning is three unrelated things, and only one of them touches the
linear solvers.

**`lsqr2_create`'s `precond_op`** is the real one: it is chained onto the
normal operator and onto the adjoint, so what the iteration sees is
$M(A^HA+\lambda)x = MA^Hy$ -- left preconditioning by composition.  It is
plumbed all the way through `sense_recon_create`, and `pics.c` passes `NULL`,
so nothing on the command line has ever used it.  Every solver here takes it:

```python
M = linop.Diagonal(1.0 / weights, shape)      # any operator image -> image
optim.CG(maxiter=30, precond=M)(kspace, A)
```

For conjugate gradients to mean anything, $M$ has to be positive definite;
BART composes it without symmetrizing, so an $M$ that is not gives an
iteration that is not conjugate gradients on anything.

**`pics --precond`** is not preconditioning.  `opt_precond_configure` adds a
`prox_weighted_leastsquares` term with the inverse sampling pattern as weights
and chains it through the model operator -- a reformulation of the data
fidelity, not a change of metric.  The tell is that `pics` asserts the
algorithm is ADMM or the primal-dual one; a preconditioner would not care.  It
is reachable through {func}`bartorch.tools.pics`, where it belongs.

**`eulermaruyama_precond`** is a genuinely preconditioned sampler, and the one
place in BART where a preconditioned conjugate-gradient solve really runs:
every step solves $(M^HM + \text{diag})o = x$ with `conjgrad`.  `pics` has no
flag for it, so {class}`EulerMaruyama`'s `sampler_precond=` is the only way to
reach it.

```python
optim.EulerMaruyama(
    term, step=0.1,
    sampler_precond=M, sampler_precond_diag=1.0,
    sampler_precond_tol=1e-4, sampler_precond_maxiter=10,
)(kspace, A)
```

BART enters that path on a positive diagonal rather than on the operator, so a
preconditioner without one would be silently ignored; this refuses it instead.

## Nonlinear least squares

BART has Gauss-Newton in two forms.  `irgnm` solves the linearized problem
with its own conjugate gradients and nothing else, which is what `nlinv` runs.
`irgnm2` pays an extra application of the derivative and hands that problem to
a *generic* regularized least-squares solver, which is how a regularized
`nlinv` or `moba` works: `noir/recon2.c` and `moba/iter_l1.c` build the inner
solver with `iter2_fista`, `iter2_admm` or `iter2_chambolle_pock` over
`nlop_get_derivative` and pass it in.

{class}`IRGNM` is both.  Without `inner=` it is the first form, run entirely
inside the library.  With one it is the second, and the inner problem goes to
any solver here:

```python
optim.IRGNM(inner="cg")                                     # iter4_irgnm2, to the bit
optim.IRGNM(inner=optim.FISTA(prox.Wavelet(axes, 0.001)))   # moba -l1's shape
optim.IRGNM(inner=optim.ADMM([prox.Wavelet(axes, w), prox.TotalVariation(axes, v)]))
```

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   IRGNM
```

## Data scaling

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   data_scaling
```

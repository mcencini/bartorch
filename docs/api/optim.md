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

`CG` minimizes $\|Ax-y\|^2 + \lambda\|x\|^2 + \sum_i w_i\|G_ix-b_i\|^2$,
the sum being `Tikhonov` terms.  Those are stacked under the encoding rather
than passed to the iteration, so what BART solves is still an ordinary
least-squares problem and a Toeplitz encoding keeps its own normal inside a
regularized solve.

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

`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term,
maxiter=30)(y, A)`.  They take one argument the classes do not: a `deepinv`
prior or a bare denoiser goes wherever a {mod}`bartorch.prox` term goes, which
is what makes a plug-and-play reconstruction the same call with the denoiser
in the term's place.

```python
from deepinv.models import DRUNet, to_complex_denoiser

x = optim.admm(y, A, to_complex_denoiser(DRUNet()), g_param=0.03)
```

A denoiser and a term of BART's own can go in the same alternating-direction
solve, each with its own split variable.  An image here is complex and most
denoisers are not, so `to_complex_denoiser` is usually needed; and a solver
holding a denoiser has no library route, which `in_library` says rather than
running something else.

## Iterations for unfolding and fixed points

`bartorch.optim.iterators`.  BART's proximal iterations written as
`deepinv.optim.optim_iterators.OptimIterator` classes: the same arithmetic in
the same order, with every operator still the library's, held against the
library's own answer to the bit at every iteration count.  An iteration
running inside the library cannot be unrolled into a network or driven to a
fixed point, because there is nothing to differentiate through; one written
out can be.

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

With nothing trainable in it a network answers with the solver's bits, batch
or no batch -- so what an unrolled network here is, is BART's iteration with a
denoiser in the threshold's place, and not an architecture that resembles one.
Two things make that hold: it starts at zero, where BART starts, rather than
at $A^Hy$ where `deepinv` starts an optimizer; and a batch is the solver run
once per item, the conjugate gradients inside an alternating-direction step
included, because one solve over the stack would stop on the whole stack's
residual and let the items steer each other's stopping.

A *learned* parameter is a tensor, and an iteration with one in it works its
scalars out in single precision throughout rather than in a double rounded at
the end.  So a trained network does not answer with the library's bits, and
could not: the numbers in it are no longer the library's.

{class}`~bartorch.optim.FISTA` refuses a fixed point, because its momentum
carries the iteration number and a step has to be the same map every time for
one to mean anything; {class}`~bartorch.optim.IST` is the same iteration
without the ravine and takes one.  So does
{class}`~bartorch.optim.ADMM`, over $(x, z, u)$ rather than over the image
alone.

### What differentiates, and what does not

Every operator a step applies is recorded, so an unrolled iteration is a torch
graph over BART's arithmetic: the encoding, its normal operator
({meth}`~bartorch.linop.LinearOperator.A_adjoint_A`, the recording counterpart
of `normal`), the transform in front of a term, and the linear solve in
ADMM's x-update.  Put a denoiser where a term goes and the gradient reaches
whatever is inside it:

```python
x = optim.admm(y, A, my_network, maxiter=10, cg_maxiter=8)
x.abs().square().sum().backward()      # reaches my_network's parameters
```

Two things are deliberately not in the graph.  BART's proximal operators carry
no derivative, so {meth}`bartorch.prox.Regularizer.prox` refuses a tensor that
carries one rather than contributing the gradient of a constant map;
{func}`bartorch.prox.frozen` says that a constant is what was meant.  And the
residual norms that steer $\rho$, $\tau$ and the stopping test are read as
numbers: they are BART's schedule for the iteration rather than part of the
model it solves.

The linear solve inside ADMM's x-update differentiates by implicit
differentiation rather than by unrolling the conjugate gradients inside it,
which is what BART does for the one solver it made an `nlop`; see
{mod}`bartorch.optim.autograd`.  {class}`~bartorch.optim.CG` is
differentiable for the same reason and on its own, which is the
data-consistency layer of a MoDL.

### Reading BART's arithmetic

Two habits of the library's decide whether an iteration written out here
answers with its bits, and both are easy to undo by accident.

Every scalar is a C `float` unless the library declares a `double`, and a
scalar worked out in a double and rounded once at the end is a different
number.  And a vector is scaled by a coefficient rather than divided by its
reciprocal: `chambolle_pock` works out `1 / sigma`, `1 / (1 + sigma)` and
`-sigma / (1 + sigma)` once, in a double, and rounds each to a float.
Dividing the tensor instead agrees while `sigma` is where `pics` starts it,
and stops agreeing once the adaptive step has moved it.

One term is also not a function of its argument: a `prox.Wavelet` with
`randshift` on -- BART's default, and `pics`'s -- shifts its transform by a
draw of BART's own before every application.  Two loops therefore agree to the
bit only when they apply it the same number of times in the same order.

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

BART has Gauss-Newton in two forms and {class}`IRGNM` is both: without
`inner=` it is `irgnm`, run entirely inside the library, and with one it is
`irgnm2`, whose linearized problem goes to any solver here.

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

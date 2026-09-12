# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration is BART's, the one `pics` or `nlinv`
runs.

Where the iteration has been written out in `bartorch.optim.iterators` --
`IST`, `FISTA`, `ADMM` and `PRIDU` -- the loop runs here and each step calls
the library; `CG`, `NIHT` and `EulerMaruyama` are one call into it.  Either
way the answer is the same bits, and `solver.in_library(y, A)` runs BART's own
loop when that is what is wanted.

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
```

`NormalEquations` differentiates $\tfrac12\|Ax-y\|^2$ as $A^HAx - A^Hy$
rather than $A^H(Ax-y)$, so a Toeplitz encoding answers with its point-spread
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

`TermPrior` puts a {mod}`bartorch.prox` term where `deepinv` expects a prior,
and a `deepinv` denoiser goes in the same place -- wrapped in
`to_complex_denoiser`, since an image here is complex and most denoisers are
not.

```{eval-rst}
.. currentmodule:: bartorch.optim
```

## Nonlinear least squares

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

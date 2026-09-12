# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration runs inside BART, as the one `pics` or
`nlinv` runs.

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
   NormalEquations
   TermPrior
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

Two things to know about `maxiter`.  On `optim.ADMM` it is BART's, and BART's
is a budget on applications of the normal operator rather than a count of
outer steps -- `admm` breaks when `nr_invokes > maxiter`, and `nr_invokes`
counts conjugate-gradient iterations across the whole run, so thirty with ten
inner iterations is about five outer steps.  `ADMMIteration` counts outer
steps instead, which is what `BaseOptim` expects; the step itself is BART's
either way, to the bit.

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

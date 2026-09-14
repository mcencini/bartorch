# Optimization

`bartorch.optim`.  A solver is configured once and called as
`solver(y, A, x0=None)`; the iteration is BART's, the one `pics` or `nlinv`
runs.  `solver.in_library(y, A)` runs BART's own loop instead, and answers
with the same bits.

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

`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term, maxiter=30)(y, A)`.
They take one argument the classes do not: a `deepinv` prior or a bare
denoiser goes wherever a {mod}`bartorch.prox` term goes.

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

## Iterations for unfolding and fixed points

`bartorch.optim.iterators`.  The same iterations written as
`deepinv.optim.optim_iterators.OptimIterator` classes, so that a solver here
can be unrolled into a network ({meth}`~CG.unrolled`) or driven to a fixed
point ({meth}`~CG.fixed_point`).

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

Every operator a step applies is recorded, so an unrolled iteration is a torch
graph over BART's arithmetic.  Two things are deliberately not in it: BART's
proximal operators, which carry no derivative, and the residual norms that
steer $\rho$, $\tau$ and the stopping test.

```python
net = optim.FISTA(denoiser, maxiter=10, step=0.9).unrolled(
    (1, 256, 256), trainable=["stepsize"]
)
image = net(kspace[None], bartorch.to_deepinv(A))
```

```{eval-rst}
.. currentmodule:: bartorch.optim
```

## Preconditioning

BART calls three unrelated things preconditioning.  `precond=` is
`lsqr2_create`'s `precond_op`, which every solver here takes; `pics --precond`
reformulates the data fidelity instead; and {class}`EulerMaruyama`'s
`sampler_precond=` is the one preconditioned solve BART really runs.

## Nonlinear least squares

BART has Gauss-Newton in two forms and {class}`IRGNM` is both: without
`inner=` it is `irgnm`, run inside the library, and with one it is `irgnm2`,
whose linearized problem goes to any solver here.

```python
optim.IRGNM(inner="cg")                                     # iter4_irgnm2, to the bit
optim.IRGNM(inner=optim.FISTA(prox.Wavelet(axes, 0.001)))   # moba -l1's shape
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

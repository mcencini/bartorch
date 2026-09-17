# Optimization

`bartorch.optim`.  {doc}`../explanation/inverse-problems` states the problems
these algorithms solve and which one applies where.  A *block* is one step of a
BART iteration as a torch module.
A solver loops a block, reproducing BART's iteration schedule -- step sizes,
penalty updates and stopping -- and is called as `solver(y, A, x0=None)`.  The
functional forms, such as `optim.fista(y, A, term)`, construct a solver and call
it in one expression.  A leading axis on `y` beyond `A`'s codomain is a batch of
independent problems, which a solver takes one run at a time: its stopping
rule, its adaptive steps and a term's random shifts belong to the run.

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
   maxeigen
```

An {class}`~bartorch.priors.ImplicitPrior` is accepted wherever a regularizer
is.

{class}`NIHT` cannot be run.  BART's `niht` applies the normal operator in
place, and the operator `lsqr2` supplies asserts that its arguments are not
aliased (`iter/niht.c:85`, `iter/lsqr.c:60`), so every NIHT solve terminates in
an assertion.

## Functional interface

Each solver has a function that constructs it and calls it in one expression;
`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term, maxiter=30)(y, A)`.
The class form is needed where a solver object is passed as an argument, as in
{class}`bartorch.nlop.IRGNM`'s `inner=`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cg
   ist
   fista
   admm
   pridu
   niht
```

## Iteration blocks

`state = block.start(y, A, x0)` initializes a run, `state = block(state, A)`
takes one step, and `block.output(state, A)` returns the image.  A solver is
these three calls in a loop, so a stack of blocks with frozen parameters
reproduces the solver's output bit for bit.  Step sizes and penalty weights are
`torch.nn.Parameter`s, frozen until `requires_grad_()` is called on them.

```python
blocks = nn.ModuleList(
    optim.FISTABlock(priors.ImplicitPrior(UNet(), sigma=0.05), step=0.9) for _ in range(10)
)
for block in blocks:
    block.step.requires_grad_()

state = blocks[0].start(kspace, A)
for block in blocks:
    state = block(state, A)
image = blocks[-1].output(state, A)
```

A block over a batch applies the operator item by item, and a term that draws
random shifts draws them per item; the scalars that drive ADMM's and PRIDU's
adaptive steps are computed over the whole batch.

The operator may be the derivative of a nonlinear operator at a point,
`F.linearize(x)`.  The gradient then reaches `x` as well: through each
application in IST, FISTA and PRIDU, and through the implicit x-update in ADMM.

Two parts of a step are outside the graph.  BART's proximal operators have no
implemented backward pass.  The residual norms that drive the schedule -- an
adaptive `rho`, an adaptive step size -- are deliberately detached, so they
control the iteration without contributing gradients.  Every other tensor
operation in a step is recorded.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ISTBlock
   FISTABlock
   ADMMBlock
   PRIDUBlock
```

## Differentiation through iterations

The solvers produce gradients in one of two ways.  The proximal solvers
unroll: `solver(y, A)` runs its block `maxiter` times in Python, and the whole
iteration is recorded.  {class}`CG` instead records the solve as a single
operation and obtains the gradient analytically -- `x = N^-1 A^H y` is linear in
`y`, so the vector-Jacobian product is one further solve with the same normal
operator followed by a forward application.  Over `F.linearize(x)` the solve is
differentiated by `x` too, through one further application of the normal
operator at `x`.  {class}`ADMM`'s x-update is a conjugate-gradient solve inside
the unrolled iteration, and is differentiated the same way.

{class}`FixedPoint` is a third route, and wraps a block rather than being a
solver.  It drives the block to its fixed point and differentiates implicitly
there, solving the adjoint fixed-point equation instead of unrolling, so memory
does not grow with the iteration count -- a deep-equilibrium model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   FixedPoint
```

## Data scaling

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   data_scaling
```

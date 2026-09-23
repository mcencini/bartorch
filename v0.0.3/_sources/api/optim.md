# Optimization

`bartorch.optim` provides BART's iterative algorithms for linear inverse
problems.  A solver is configured by its constructor and called as
`solver(y, A, x0=None)`, with `y` the data and `A` a
{obj}`~bartorch.linop.LinearOperator`; it reproduces BART's iteration,
including its step sizes, penalty updates and stopping rules.  An iteration
block is one step of the same iteration as a `torch.nn.Module`.  A leading
axis of `y` beyond the codomain of `A` is a batch of independent problems.
{doc}`../explanation/inverse-problems` states the problems and algorithms, and
{doc}`../explanation/differentiation` the backward pass of each solver.

```{eval-rst}
.. currentmodule:: bartorch.optim
```

| Solver | Problem | Regularization | Backward pass |
| --- | --- | --- | --- |
| {obj}`~bartorch.optim.CG` | $\min_x \lVert Ax-y\rVert^2 + \lambda\lVert x\rVert^2 + \sum_i w_i \lVert G_i x - b_i\rVert^2$ | {obj}`~bartorch.optim.Tikhonov` terms | Implicit: one further conjugate-gradient solve |
| {obj}`~bartorch.optim.IST` | $\min_x \tfrac12\lVert Ax-y\rVert^2 + g(x)$ | One term with $G = I$ | Unrolled |
| {obj}`~bartorch.optim.FISTA` | As IST, with momentum | One term with $G = I$ | Unrolled |
| {obj}`~bartorch.optim.ADMM` | $\min_x \tfrac12\lVert Ax-y\rVert^2 + \sum_j g_j(G_j x)$ | Any number of terms, any $G_j$ | Unrolled; each x-update implicit |
| {obj}`~bartorch.optim.PRIDU` | As ADMM | Any number of terms, any $G_j$ | Unrolled |
| {obj}`~bartorch.optim.NIHT` | Sparsity-constrained least squares | Hard-thresholding terms | Not runnable (BART assertion) |
| {obj}`~bartorch.optim.POCS` | Feasibility: repeated projections | Projections and terms at unit weight | Unrolled |

Terms are the objects of {mod}`bartorch.priors`; an
{obj}`~bartorch.priors.ImplicitPrior` is accepted wherever a regularization term is.

## Linear least squares

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.CG` | Conjugate gradients on the normal equations, with Tikhonov terms |
| {obj}`~bartorch.optim.Tikhonov` | Quadratic penalty $w \lVert G x - b\rVert^2$ for {obj}`~bartorch.optim.CG` |

## Regularized least squares

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.IST` | Iterative soft thresholding (proximal gradient) |
| {obj}`~bartorch.optim.FISTA` | Fast iterative soft thresholding (accelerated proximal gradient) |
| {obj}`~bartorch.optim.ADMM` | Alternating direction method of multipliers |
| {obj}`~bartorch.optim.PRIDU` | Chambolle-Pock primal-dual iteration |
| {obj}`~bartorch.optim.NIHT` | Normalized iterative hard thresholding; raises `NotImplementedError` |
| {obj}`~bartorch.optim.maxeigen` | Power-iteration estimate of the largest eigenvalue of $A^H A$ |

## Projection methods

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.POCS` | Projection onto convex sets, repeated sweeps |
| {obj}`~bartorch.optim.POCSBlock` | One sweep of the projections |

## Functional interface

Each function constructs the corresponding solver and calls it:
`optim.fista(y, A, term, maxiter=30)` is `optim.FISTA(term, maxiter=30)(y, A)`.

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.cg` | {obj}`~bartorch.optim.CG` in one call |
| {obj}`~bartorch.optim.ist` | {obj}`~bartorch.optim.IST` in one call |
| {obj}`~bartorch.optim.fista` | {obj}`~bartorch.optim.FISTA` in one call |
| {obj}`~bartorch.optim.admm` | {obj}`~bartorch.optim.ADMM` in one call |
| {obj}`~bartorch.optim.pridu` | {obj}`~bartorch.optim.PRIDU` in one call |
| {obj}`~bartorch.optim.niht` | {obj}`~bartorch.optim.NIHT` in one call |
| {obj}`~bartorch.optim.pocs` | {obj}`~bartorch.optim.POCS` in one call, with the projections in place of an encoding |

## Iteration blocks

`state = block.start(y, A, x0)` initializes a run, `state = block(state, A)`
takes one step and `block.output(state, A)` returns the image.  Step sizes and
penalty weights are `torch.nn.Parameter` objects, frozen until
`requires_grad_()` is called on them.

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.ISTBlock` | One iterative soft-thresholding step |
| {obj}`~bartorch.optim.FISTABlock` | One fast iterative soft-thresholding step |
| {obj}`~bartorch.optim.ADMMBlock` | One ADMM step |
| {obj}`~bartorch.optim.PRIDUBlock` | One primal-dual step |

## Fixed-point methods

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.FixedPoint` | An iteration block iterated to its fixed point and differentiated implicitly there (deep equilibrium) |

## Data scaling

| Object | Description |
| --- | --- |
| {obj}`~bartorch.optim.data_scaling` | BART's estimate of the data scale by which `pics` divides the data before it iterates |

{doc}`../auto_examples/01-basics/02-operators-and-solvers` assembles a BART
reconstruction from an operator, a term and a solver.

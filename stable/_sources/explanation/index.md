# Explanation

The concepts behind the interfaces, their notation, and the reasons for the
choices visible in the API.  The pages assume linear algebra and numerical
computing, not MRI reconstruction or convex optimization; terms from either are
introduced where first used.  Exact interfaces are in {doc}`../api/index`, and
complete workflows in the {doc}`examples <../examples/index>`.

| Page | Question |
| --- | --- |
| {doc}`execution-model` | Which interface fits a task, and what runs underneath it? |
| {doc}`inverse-problems` | What is estimated, from what, and by which algorithm? |
| {doc}`encoding` | What does the MRI forward operator consist of, and how is it represented? |
| {doc}`non-cartesian` | How is the Fourier transform computed off the Cartesian grid, and what is its normal operator? |
| {doc}`nonlinear` | What changes when the forward operator is nonlinear in the unknowns? |
| {doc}`differentiation` | How do gradients pass through operators, solvers and unrolled iterations? |

```{toctree}
:hidden:

execution-model
inverse-problems
encoding
non-cartesian
nonlinear
differentiation
```

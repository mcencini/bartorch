# Explanation

These pages introduce the concepts the library is built on, the notation it
uses for them, and the reasons behind the choices that are visible in its
interface. They are not a tour of the API: {doc}`../api/index` states what each
object does, and the {doc}`examples <../auto_examples/index>` show complete
workflows.

| Page | Question |
| --- | --- |
| {doc}`inverse-problems` | What is being estimated, from what, and by which algorithm? |
| {doc}`encoding` | What does the MRI forward operator consist of, and what determines its form? |
| {doc}`non-cartesian` | How is a transform computed off the Cartesian grid, and what does that cost? |
| {doc}`nonlinear` | What changes when the forward operator is not linear in the unknowns? |

The reader assumed here is comfortable with linear algebra and numerical
computing, and not necessarily with MRI reconstruction or with convex
optimization. Terms from either field are introduced where they are first used.

```{toctree}
:hidden:

inverse-problems
encoding
non-cartesian
nonlinear
```

# Nonlinear fusion

The plan for giving nonlinear operators what [composed encodings](composed-encodings.md)
gave linear ones. Each operator supplies its derivative as a function of the
linearization point; a composition's follows by the chain rule; the Gauss-Newton
step is then assembled over any of them rather than over BART's coil model
alone; and a planner rewrites the composition so the transform is applied once
per step as its normal. The design was agreed on 2026-09-15.

The requirement that decides every section is that a model built from primitives
has to train as well as BART's own does, and that no noir code is reimplemented
to get there. What is written here is the algebra `noir/model_net.c` is written
in, over operators the library already exposes.

## The derivative bundle

A Gauss-Newton step asks a model for three things at the current iterate `xn`:
the residual `y - F(xn)`, its adjoint image `DF(xn)^H (y - F(xn))`, and the
inverse of `DF(xn)^H DF(xn) + alpha`. All three are functions of `xn`, and BART
supplies only the first that way. `nlop_get_derivative` -- `nlop.Derivative`,
`F.jacobian()` -- returns a linear operator that reads the point out of the
operator's own state, where the last `forward` left it.

That is enough to *solve*. `IRGNM(...)(y, F, x0)` evaluates `F` at `xn` and then
applies the derivative, in that order, every iteration. It is not enough to make
the step itself an operator, because the step's dependence on `xn` through
`DF(xn)` is then state rather than an argument, and nothing differentiates by
state:

```
  stored point                              point as an argument

  xn ──► forward ──► F(xn)                  xn ─────┐
              │                                     ▼
            [state]                   dz ──► adjoint(dz, xn) ──► DF(xn)^H dz
              │
  dz ──► adjoint ──► DF(xn)^H dz            d/d xn reaches the point
              ✗
  d/d xn sees nothing
```

This is why `noir/model_net.c` writes the point out. `noir_get_adjoint`
(`model_net.c:266`) takes `(dz, xn)` and returns `dx`; `noir_get_derivative`
(`:295`) takes `(dx, xn)`; `noir_get_normal` (`:324`) is the two chained with the
point duplicated. `_Cell` exists because those three exist for BART's coil model
and for nothing else.

A **derivative bundle** is that record, supplied by the operator rather than
built for one model:

| Member | Arguments | Returns |
| --- | --- | --- |
| `forward` | `xn` | `F(xn)` |
| `derivative` | `dx`, `xn` | `DF(xn) dx` |
| `adjoint` | `dz`, `xn` | `DF(xn)^H dz` |
| `normal` | `dx`, `xn` | `DF(xn)^H DF(xn) dx` |

Each is an ordinary `nlop`, so the bundle is BART's algebra throughout and a
member composes with anything else in it.

**The tangent is input 0 and the point is whatever follows.** That is not a
convention chosen here: `norm_inv_lambda_create` duplicates input 0 into the
Tikhonov term and appends `lambda` after every other input (`norm_inv.c:428`),
and `norm_inv_create` asserts one output whose codomain is the domain of input 0
(`:334`, `:346`). A normal written the other way round inverts the wrong
operator, and BART's assertion does not catch it because the shapes agree.

The point may be more than one argument. A model of several unknowns has a
bundle whose members take them all, and it is the caller's `flatten` that decides
whether a step runs over one vector or several -- the same choice
`noir_get_forward` makes for itself with `nlop_flatten_in_F` and
`nlop_stack_inputs_F`.

`normal` is carried rather than derived because deriving it throws away the
cheapest thing the model knows; what it saves is [The normal-equation
domain](#the-normal-equation-domain). An operator that has no bundle still
evaluates, differentiates and solves through `IRGNM(...)(y, F, x0)`, and is
refused only the operator form; [Outside the form](#outside-the-form) says what
that covers.

## Bundles of the primitives

BART applies every elementwise derivative as a multiplication by a diagonal it
stored at the forward, and the adjoint as a multiplication by that diagonal's
conjugate: `md_ztenmul` and `md_ztenmulc` in `nlop_jacobian.c:189` and `:213`.
So a one-input primitive's bundle is fixed once the diagonal is known, and the
diagonal is a function of the point and the value:

```
derivative(dx, xn) = tenmul(dx, d(xn))     adjoint(dz, xn) = tenmul(dz, conj d(xn))
```

| `bartorch` | BART | `d(x)` | Where |
| --- | --- | --- | --- |
| `Exp` | `zexp` | `F(x)` | `zexp.c:42` |
| `Log` | `zlog` | `1 / x` | `zexp.c:74` |
| `Sqrt` | `zsqrt` | `0.5 / F(x)` | `someops.c:519` |
| `Inverse(eps)` | `zinv_reg` | `-F(x)^2` | `someops.c:353` |
| `Power(p)` | `zspow` | `p F(x) / x` | `someops.c:595` |

The rest of what needs declaring is not diagonal:

| `bartorch` | BART | Bundle |
| --- | --- | --- |
| `Multiply` | `tenmul` | `derivative = da·b + a·db`; `adjoint = (conj(b)·dz, conj(a)·dz)` |
| `Weighted` | `zaxpbz` | linear in both inputs: the derivative is the operator, the point unused |
| `Add` | `zsadd`, which is `zaxpbz` against a constant (`someops.c:218`) | the identity |
| `FromLinear(L)` | `nlop_from_linop` | `L` and `L^H`, the point unused |
| `Constant` | `nlop_const` | no input, so no tangent |
| `FromTorch`, `FromTorchSim` | callbacks | `torch.func.jvp` and the reverse-mode vjp, evaluated at the point given as an argument rather than at the stored one |

`Multiply`'s row is what `noir_get_adjoint` and `noir_get_derivative` build by
hand (`model_net.c:269-287`, `:299-315`), and it is the product rule; nothing
about it is particular to a coil model.

Everything else in `nlop/basic.py` is already a composition in BART, and the
chain rule reaches it with no declaration at all:

| `bartorch` | What BART builds it from | Where |
| --- | --- | --- |
| `Divide` | `chain(zinv, tenmul)` | `someops.c:395` |
| `Sum` | `tenmul(conj x, x)` duplicated, then `zreal` | `someops.c:539` |
| `RootSumOfSquares` | `zss`, `zsadd(eps)`, `zsqrt` | `someops.c:562` |
| `Abs` | `zrss` over no axes | `someops.c:626` |
| `SmoothAbs` | `zrss` over no axes, with `eps` | `someops.c:621` |
| `Phase` | `zabs` into `zdiv`, duplicated | `someops.c:638` |

Conjugation is not complex-linear, and BART carries it as `linop_zconj_create`,
whose adjoint is itself; `linop.Conj` and `linop.Real` are already that. This is
the Wirtinger convention the rest of the library is under -- a backward pass is
the adjoint and not the transpose -- so the bundles inherit it rather than
introducing it.

## The chain rule

A composition's bundle is its operands', assembled by the algebra the composition
was written in:

| Node | `derivative(dx, x)` | `adjoint(dz, x)` |
| --- | --- | --- |
| `chain(f, g)` | `D_g(D_f(dx, x), f(x))` | `D_f^H(D_g^H(dz, f(x)), x)` |
| `combine(f, g)` | the two side by side | the two side by side |
| `dup(a, b)` | the sum of the two tangent paths | the two adjoints, added |
| `link(o, i)` | the chain rule along the tie | the chain rule along the tie |
| `pin(i, v)` | the input leaves both tangent and point | the same |
| `flatten`, `reshape`, `permute`, `stack` | relabelled axes | relabelled axes |

`chain` is the only row that needs something the operands do not have: `f(x)`,
the point the second operand is linearized at. **It is recomputed, not carried.**
That is what BART does -- `noir_get_derivative` applies `lop_im` and `lop_coil`
to `xn` every time it runs, and `noir_get_normal` chains derivative into adjoint,
so the point passes through both twice per normal application. Carrying it
instead would mean a cache keyed by the point, which is the state [The derivative
bundle](#the-derivative-bundle) exists to remove; it would also have to be
invalidated by the very gradient step that is trying to differentiate by the
point.

What recomputation costs is an extra forward evaluation of each stage per
application, and BART's answer to that is `nlop_checkpoint_create_F`, applied to
the step, to each unrolled cell and to the whole (`model_net.c:388`, `:404`,
`:423`). It trades the recomputation against memory in the backward pass, and it
is one of the two entry points this design adds to the ABI.

## The normal-equation domain

`noir2_net` never applies the transform. `noir_get_forward` is a `tenmul` chained
into `linop_get_normal(model->lop_fft)` (`model_net.c:255-256`), and
`noir_get_adjoint` carries no transform at all. `noir2` -- what `NonlinearSense`
is built on -- chains the whole `lop_fft` instead (`model2.c:185`). The two
models differ in exactly this.

Write `F = E ∘ G` with `E` linear. Then for the step's three quantities:

```
  y        stored as        E^H y            once, before the first step
  F(xn)                     E^H E G(xn)      one normal application
  DF^H r                    DG^H r           no transform at all
  DF^H DF                   DG^H (E^H E) DG  one normal application
```

Every appearance of `E` has become one appearance of `E^H E`, and the residual
`E^H y - E^H E G(xn)` is `E^H (y - E G(xn))` -- the rewrite is exact, not an
approximation.

| | Transforms per normal application | Data argument |
| --- | --- | --- |
| `E` applied as a pair | forward and adjoint | samples |
| `E` applied as its normal | one | coil images |

This is the same trade `pics` makes between its Toeplitz normal and its transform
pair, and `AGENTS.md` already records what that is worth there: 1.06 s against
2.33 s on a 256x256 eight-coil radial dataset on the host. Whether it is worth
the same inside a Newton step is measured in [Verification](#verification) and is
not claimed here.

It is a trade and not a free win. The data argument moves from samples to coil
images, so an acquisition with far fewer samples than voxels stores more, and an
encoding whose normal has no kernel gains nothing but pays the same storage. The
plan says which happened.

This is also where `_Cell`'s companions go. `prepare()` is `E^H` applied once,
which any linear operator has; `split`, `join` and `decompose` are the flattening
of a multi-unknown state and the model's own linear parts, which `flatten`,
`stack_inputs` and the operator's parts already give. None of them needs the noir
model to exist.

## The generic step

BART's step is one expression, and every operator in it outside the bundle is
public:

```
x = xn + ( DF(xn)^H DF(xn) + alpha )^-1 [ DF(xn)^H (y - F(xn)) - alpha (xn - x0) ]
```

`alpha` is a vector as long as the state, multiplied by --
`norm_inv_lambda_create(..., ~0UL)` selects every axis (`norm_inv.c:425`), which
is what `_Cell.weight` already builds.

| `noir_gauss_newton_step_create_s` | Over a bundle |
| --- | --- |
| `noir_get_forward(model)` (`:363`) | `bundle.forward` |
| `nlop_zaxpbz_create(N, kdims, 1, -1)` (`:377`) | unchanged |
| `noir_get_adjoint(model)` (`:378`) | `bundle.adjoint` |
| `nlop_zaxpbz_create(1, dims, 1, -1)` and `nlop_tenmul_create` (`:380-383`) | unchanged |
| `nlop_checkpoint_create_F` (`:388`) | unchanged |
| `norm_inv_lambda_create(conf, noir_get_normal(model), ~0UL)` (`:354`) | `bundle.normal` |
| `nlop_zaxpbz_create(1, dims, 1, 1)` and the `dup`s (`:396-397`) | unchanged |

Three substitutions in an expression that is otherwise BART's own calls in BART's
own order -- which is what "no noir code is copied" means here, and what makes
the same reading hold for `noir_gauss_newton_iter_create_s` (`:402`), the
decaying-`alpha` loop `_Cell` already exposes.

The step asserts its state is one flat vector (`model_net.c:367`), so a model of
several unknowns is flattened first, as `noir_get_forward` flattens its own two.

## Fusion of the coil model

Two things in `nlop` are the same description, and the planner is what says so:

```
  NonlinearSense              CoilSense(E)

  image ──► lop_im ─┐         image ─┐
                    ├► tenmul ─► lop_fft ─► data
  coils ──► lop_coil┘          coils ┘
```

`bartorch_noir_coils`, `bartorch_noir_image` and `bartorch_noir_transform` hand
out `model.lop_coil`, `model.lop_im` and `model.lop_fft` (`ops.c:2190-2193`),
which are exactly the three `noir_get_forward` is built from. So
`NonlinearSense` declares itself as that composition rather than as an opaque
`nlop`, and the chain rule over `Multiply` produces `noir_get_forward`,
`noir_get_adjoint` and `noir_get_derivative` without any of them being written
for it.

What the planner matches, and what it does:

| In the composition | Lowered to |
| --- | --- |
| a linear operator after a `Multiply` of two unknowns | `E` folded into the bundle as `E^H E`, per [The normal-equation domain](#the-normal-equation-domain) |
| a linear operator before either unknown | folded into that input's own linear part |
| a `Diagonal` on either side | folded the way the linear planner folds one |
| anything else | the bundle by the plain chain rule, transform pair and all |

`E` is whatever the linear planner made of it, so a fused `NoncartesianSense`
with a point spread function arrives as one operator with a normal kernel, and
the coil slab loop of `src/csrc/ops/sense.c` runs inside the Newton solve rather
than around it. No C is written for this: the executor is the linear one,
reached through `LinearOperator.normal`.

**The chosen plan is never silent**, for the same reason it is not in the linear
case: a fallback answers with the same numbers several times slower. `F.plan`
names the bundle's source for each stage (declared, chain rule, or torch),
whether the step runs in the normal-equation domain, and the linear part's own
`plan`; `plan.fused` is false when the rewrite was not taken. The executor
counters stay the linear ones, because it is the linear encoding that runs.

`IRGNM.operator(F)` then takes any operator with a bundle, `_Cell` is retired
with `bartorch_noir_net_*` left in the ABI unused by the operator layer, and the
restriction list the network model imposed -- `oversampling_coils=1.0`, no
`optimized`, no `oversampled_coils`, no separate coefficient shape -- goes with
it, because none of it is a property of the expression.

## Outside the form

A bundle is refused rather than approximated. What has none:

- An operator BART built that this library did not declare a bundle for and
  cannot read as a composition -- a future BART constructor, or an `nlop` handed
  in from elsewhere. Its `forward`, `derivative` and `adjoint` still work at the
  stored point, so `IRGNM(...)(y, F, x0)` solves; only `IRGNM(...).operator(F)`
  is refused, naming the operator.
- A bundle member that would need a second forward per sample rather than per
  application. `FromTorch` is the boundary: `torch.func.jvp` is one extra
  evaluation, which is why it has a bundle, while anything needing a materialised
  Jacobian does not.

A composition that has a bundle but does not match the planner is not outside the
form: it is the plain chain rule, one transform each way, correct and slower, and
`plan.fused` says so.

## Verification

A numerical test pins the library against something outside BART. For bundles
that is torch, which can differentiate every primitive in the table:

| Claim | Pinned against |
| --- | --- |
| a declared bundle's `derivative` | `(F(x + eps dx) - F(x)) / eps`, and `torch.func.jvp` of the same function |
| a declared bundle's `adjoint` | the adjoint identity `<D dx, dz> = <dx, D^H dz>` over random vectors, and torch's own gradient -- with the transpose asserted to disagree, as `tests/test_linop.py` does |
| `normal` | `adjoint(derivative(dx, x), x)` |
| a composed bundle | torch's jvp and vjp of the composition written out |
| the generic step | a Gauss-Newton loop written out in torch, on a model small enough to write out -- a two-parameter exponential fit |
| the step's gradients | torch autograd through that written-out loop, for all four of `y`, `xn`, `x0`, `alpha` |
| the normal-equation rewrite | the same step assembled without it, to the last bits where the arithmetic is the same and to the solver's tolerance where it is not |
| fusion | the fused plan asserted by `F.plan`, and the fused result against the unfused one |

Two agreement checks are BART against BART and are labelled as such rather than
counted as numerical tests: a bundle's `derivative` against `nlop_get_derivative`
at the same point, and the generic step against `_Cell` for the noir composition.
They say the two routes have not diverged; they do not say either is right.

## Phases

1. **Bundles for the primitives.**
   - The bundle record, the five diagonals, `Multiply`, the linear and constant
     cases, and the torch route.
   - Done when each is checked against finite differences, the adjoint identity
     and torch's own gradient, and agrees with `nlop_get_derivative`.
   - No ABI change.
2. **The chain rule.**
   - Bundles for `chain`, `combine`, `dup`, `link`, `pin` and the relabelling
     nodes; `NonlinearSense` declared as its composition.
   - Done when a composed bundle matches torch's jvp and vjp of the written-out
     composition, and `NonlinearSense`'s matches `nlop_get_derivative` on BART's
     own model.
3. **The generic step.**
   - `bartorch_nlop_checkpoint` and `bartorch_nlop_norm_inv_lambda` added to the
     header, `_abi.py` regenerated; `IRGNM.operator(F)` assembled over any
     bundle.
   - Done when it matches the written-out torch loop, carries gradients by all
     four arguments, and reproduces `_Cell` for the noir composition.
4. **The planner and fusion.**
   - The normal-equation rewrite, `F.plan`, `_Cell` retired and the
     `NonlinearSense`-only restriction removed.
   - Done when the fused and unfused results agree, the fused plan is asserted,
     and the benchmark in [Targets](#targets) is filled in.

Names that appear: `nlop.Derivative` keeps its meaning; `IRGNM.operator(F)` loses
its type restriction; `F.plan` and `F.bundle` are new. Names that go: `_Cell` and
its `prepare`/`split`/`join`/`decompose` companions, replaced by the linear
part's adjoint and the algebra's own `flatten`.

## Targets

To be filled by a run on a machine with a card, and by a host run. One case per
row, best of five, against the same reconstruction assembled without the rewrite:

| Case | Step, paired transform (s) | Step, normal domain (s) |
| --- | --- | --- |
| Cartesian 256², 8 coils, 8 Newton steps | — | — |
| non-Cartesian 256², 8 coils, 401 spokes, 8 Newton steps | — | — |
| the same, unrolled and trained for one epoch | — | — |

## Working constraints

- **Environment.** The implementing session has no GPU. Everything above is
  checked in the CPU suite as `AGENTS.md` describes; the card tests are written
  alongside the code and marked as needing one. What needs a card run: the fused
  plan on device memory, the coil slab loop's counters inside a Newton solve, and
  every row of [Targets](#targets).
- **BART.** Not edited. The two new entry points wrap public constructors in
  `src/csrc/ops/`; nothing is compiled in BART's place.
- **Docstrings.** Two to four lines of contract for each addition. Documentation
  edits stay in `docs/api/`.
- **Commits.** One per phase step, with the suite passing; a pull request to
  `main` at the end.

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
bundle whose members take them all, and the step lays them end to end into one
state -- the same choice `noir_get_forward` makes for itself with
`nlop_flatten_in_F` and `nlop_stack_inputs_F`.

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
| `FromTorch` | callbacks | `torch.func.jvp` and the reverse-mode vjp, evaluated at the point given as an argument rather than at the stored one |
| `FromTorchSim` | TorchSim | ``A_jvp(x, dx)`` and ``A_vjp(x, dy)``, which take the point as an argument already and build no Jacobian |

`Multiply`'s row is what `noir_get_adjoint` and `noir_get_derivative` build by
hand (`model_net.c:269-287`, `:299-315`), and it is the product rule; nothing
about it is particular to a coil model.

Everything else in `nlop/basic.py` is already a composition in BART, and gets a
bundle by declaring itself as that composition -- the way `NonlinearSense` does
-- rather than by carrying one of its own.  Each is one BART constructor here,
so without the declaration the chain rule has nothing to walk into:

| `bartorch` | What BART builds it from | Where |
| --- | --- | --- |
| `Divide` | `chain(zinv, tenmul)` | `someops.c:395` |
| `Sum` | `tenmul(conj x, x)` duplicated, then `zreal` | `someops.c:539` |
| `RootSumOfSquares` | `zss`, `zsadd(eps)`, `zsqrt` | `someops.c:562` |
| `Abs` | `zrss` over no axes | `someops.c:626` |
| `SmoothAbs` | `zrss` over no axes, with `eps` | `someops.c:621` |
| `Phase` | `zabs` into `zdiv`, duplicated | `someops.c:638` |

`Phase` is the row that declares nothing: it is already written here as those
two put together, so its bundle follows once they have one.

**What `zss` conjugates it also makes real-linear.** Conjugation is not
complex-linear, and BART carries it as `linop_zconj_create`, whose adjoint is
itself; `linop.Conj` and `linop.Real` are already that, and say of themselves
that they fail a complex dot test alone. So the five built on `zss` have an
adjoint that satisfies the identity in the *real* inner product and not the
complex one, and the tests assert the complex one fails -- asserting it held
would be asserting a different operator. This is the Wirtinger convention the
rest of the library is under, inherited rather than introduced.

## The chain rule

A composition's bundle is its operands', assembled by the algebra the composition
was written in:

| Node | `derivative(dx, x)` | `adjoint(dz, x)` |
| --- | --- | --- |
| `chain(f, g)` | `D_g(D_f(dx, x), f(x))` | `D_f^H(D_g^H(dz, f(x)), x)` |
| `combine(f, g)` | the two side by side | the two side by side |
| `dup(a, b)` | the sum of the two tangent paths | the two adjoints, added |
| `pin(i, v)` | the input leaves both tangent and point | the same |
| `del_out(o)` | the output carries no tangent | its cotangent is zero |
| `reshape`, `permute` | relabelled axes | relabelled axes |

`link` has no row. Its derivative table is
`D[o][i] + D[o][ii] . D[oo][i]` (`chain.c:442`), which is over the whole graph
rather than over the node, so a composition carrying one reports no bundle and
[Outside the form](#outside-the-form) covers it. `flatten` and `stack` have none
either, because it is the step that lays a state out and it does so on the
members; see [The generic step](#the-generic-step).

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
`noir_get_adjoint` carries no transform at all.

`noir2` -- what `NonlinearSense` is built on -- does the same off the grid and
not on it. `noir2_join(ret, asym)` (`model2.c:164`) ends the model with
`linop_from_ops(lop_fft->normal, identity->adjoint)` when `asym` is set and with
`lop_fft` itself otherwise (`:180`, `:185`). So this is not a rewrite BART lacks;
it is one BART applies where the transform is expensive, and the planner's work
is to apply it wherever it pays.

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

What it costs is the adjoint identity: the model's Jacobian is no longer the
adjoint of its own derivative, because `E^H` sits in the data rather than in the
adjoint. BART's own non-Cartesian model fails that identity for the same reason.
What survives, and what the inner solve needs, is that
`DG^H (E^H E) DG` is Hermitian.

| | Applications of `E` per normal | What one costs | Data argument |
| --- | --- | --- | --- |
| as a pair | a forward and an adjoint | two transforms | samples |
| as its normal | one | whatever `linop_get_normal` is | coil images |

What the one costs is the whole question, and it is the encoding's answer rather
than this design's. Off a grid it is a point spread function, which is the trade
`pics` makes and which `AGENTS.md` already prices there: 1.06 s against 2.33 s on
a 256x256 eight-coil radial dataset on the host. On a grid `linop_get_normal` of
an FFT is the same two transforms, so there is nothing to win -- which is why
BART leaves its own Cartesian model paired. [Targets](#targets) is the
measurement.

It is a trade and not a free win. The data argument moves from samples to coil
images, so an acquisition with far fewer samples than voxels stores more, and an
encoding whose normal has no kernel gains nothing but pays the same storage. The
plan says which happened.

This is also where the step's own companions come from. `Step.prepare` is `E^H`
applied once, which any linear operator has, and `split` and `join` are the
laying out of a multi-unknown state. Neither needs the noir model to exist, which
is why they are the step's rather than a model's; what `_Cell` keeps that they do
not replace is in [Fusion of the coil model](#fusion-of-the-coil-model).

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

The step asserts its state is one flat vector (`model_net.c:367`), so each
member is laid out first, as `noir_get_forward` lays out its own two:
`nlop_flatten_in` per argument and then `nlop_stack_inputs`, and not
`nlop_flatten`, which builds at BART's sixteen axes. What the assertion is about
is the rank, so a model of one unknown is written this way too. The operators the
step puts around the state are built at `N = 1` for the same reason: `nlop_dup`
compares `iovec`s and not shapes.

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
| an asymmetric stage already there | left as it stands: BART lowered it itself |
| anything else | the bundle by the plain chain rule, transform pair and all |

The rewrite is taken wherever the composition matches, and not on a judgement
about which is faster. `linop_get_normal` is the encoding's own normal where it
has one -- a point spread function for a NUFFT, the transform's own where no
k-space factor survives -- and the two applications where it has not, which is
what the pair costs anyway. So the arithmetic is never more; what the rewrite
trades is where the data lives, and that is what the plan reports. `fuse=False`
declines it, which is what the fused answer is held against.

`E` is whatever the linear planner made of it, so a fused `NoncartesianSense`
with a point spread function arrives as one operator with a normal kernel, and
the coil slab loop of `src/csrc/ops/sense.c` runs inside the Newton solve rather
than around it. No C is written for this: the executor is the linear one,
reached through `LinearOperator.gram`.

**The chosen plan is never silent**, for the same reason it is not in the linear
case: a fallback answers with the same numbers several times slower.
`Step.plan` -- what `IRGNM(...).operator(F)` returns -- names where the bundle
came from (declared, linear, chain rule, or torch), whether the step runs in the
normal-equation domain, and the linear part's own `plan`; `plan.fused` is false
where the rewrite was not taken. The plan sits on the step rather than on the
model because that is where the decision is made and read back from, not worked
out a second time. The executor counters stay the linear ones, because it is the
linear encoding that runs.

`IRGNM.operator(F)` then takes any operator with a bundle, and the restriction
list the network model imposed -- `oversampling_coils=1.0`, no `optimized`, no
`oversampled_coils`, no separate coefficient shape -- applies only to the model
that imposes it.

**`_Cell` stays, and for two reasons that are `noir2_net`'s rather than the
expression's.** It carries BART's own batch axis, which is
`nlop_stack_multiple` over one model per item; and its `prepare()` takes the
sampling pattern as an *argument*, which is what lets one trained network answer
for several patterns -- `noir2` bakes the pattern into `lop_pattern` at
construction, so a `Step` over a `NonlinearSense` is one model per pattern.
Neither is reachable by assembling the expression differently. So
`IRGNM.operator` sends a `NonlinearSense` to `_Cell` and everything else to
`Step`, and a test holds the two to the same numbers over the composition BART's
own model writes.

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
| a declared bundle's `derivative` | a central quotient, whose error is quadratic in the step and so still resolved in single precision, and `torch.func.jvp` of the same function |
| a declared bundle's `adjoint` | the adjoint identity `<D dx, dz> = <dx, D^H dz>` over random vectors, and torch's own gradient -- with the transpose asserted to disagree, as `tests/test_linop.py` does |
| `normal` | `adjoint(derivative(dx, x), x)` |
| a composed bundle | torch's jvp and vjp of the composition written out |
| the generic step | a Gauss-Newton loop written out in torch, on a model small enough for its Jacobian to be a matrix, with each inner problem solved exactly rather than by conjugate gradients -- so what is compared is the method and not two paths through one iteration |
| the step's gradients | torch autograd through that written-out loop, for all four of `y`, `xn`, `x0`, `alpha` |
| the normal-equation rewrite | the same step assembled with `fuse=False`: to single precision on a grid, and off it to a distance that closes as the transform's tolerance is tightened, which is what says the transform and not the rewrite is what separates them |
| fusion | the plan asserted by `Step.plan`, on a coil composition and on one that is not |

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
   - The normal-equation rewrite, `Step.plan`, and the `NonlinearSense`-only
     restriction removed.
   - Done when the fused and unfused results agree, the fused plan is asserted,
     and the benchmark in [Targets](#targets) is filled in.
   - `_Cell` is not retired; see [Fusion of the coil model](#fusion-of-the-coil-model).

Names that appear: `nlop.Derivative` keeps its meaning; `IRGNM.operator(F)` loses
its type restriction; `F.bundle`, `nlop.Bundle`, `nlop.Plan` and `Step.plan` are
new, and `Step` carries `prepare`, `split` and `join` of its own. No name goes.

## Targets

`scripts/benchmark_newton.py`, best of five, one case per process, against the
same step assembled with `fuse=False`.  Measured on a four-core host with no
card: 256² with eight coils and eight Newton steps.

| Case | paired (s) | normal domain (s) | |
| --- | --- | --- | --- |
| Cartesian | 4.79 | 4.53 | 1.06x |
| non-Cartesian, 401 spokes | 14.29 | 7.52 | 1.90x |

Which is the shape the rewrite was expected to have and the reason the note does
not gate on it: on a grid `linop_get_normal` of an FFT is the same two
transforms, so there is nothing to win and nothing to lose; off it the point
spread function halves the step.

Two rows are still open, and both want a machine that is not this one: the same
cases on a card, and the backward pass, whose first measurement here --
Cartesian, 19.96 s against a 4.70 s forward -- says only that it is the four
solves and not the rewrite that a training step spends its time in.

## Working constraints

- **Environment.** The implementing session has no GPU. Everything above is
  checked in the CPU suite as `AGENTS.md` describes, and `tests/test_nlop_cuda.py`
  is what wants a card: a bundle, a composed bundle, a step and the fused coil
  step all answering on device memory what they answer on the host, the gradient
  arriving there, and the counters saying the normal off the grid was the point
  spread function rather than the pair. Written without a card they were run
  against the host with the device substituted out, which exercises every
  assertion and leaves only the dispatch untried -- which is the one thing the
  card is for. What else wants one: [Targets](#targets) again, where the host
  numbers are what a four-core container measured rather than what the design is
  worth.
- **BART.** Not edited. The two new entry points wrap public constructors in
  `src/csrc/ops/`; nothing is compiled in BART's place.
- **Docstrings.** Two to four lines of contract for each addition. Documentation
  edits stay in `docs/api/`.
- **Commits.** One per phase step, with the suite passing; a pull request to
  `main` at the end.

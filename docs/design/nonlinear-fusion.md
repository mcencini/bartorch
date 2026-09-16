# Nonlinear fusion

The design for giving nonlinear operators what [composed encodings](composed-encodings.md)
gave linear ones. Each operator supplies its derivative as a function of the
linearization point, and a composition's derivative follows by the chain rule.
`nlop.IRGNMBlock` takes BART's Gauss-Newton step over any such operator, and a
planner rewrites a coil composition so that its encoding is applied once per step
as its normal operator.

The requirement behind every section is that a model built from primitives trains
as well as BART's own model does, without reimplementing noir. The step applies
the operators `noir/model_net.c` is written in, over operators the library
already exposes; torch supplies only the additions and multiplications between
them.

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
point duplicated. BART writes those three by hand for its coil model and for
nothing else, and all three are `static`.

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
[Outside the form](#outside-the-form) covers it. `stack` has none either.
`flatten` has the bundle the step lays a state out with; see
[The generic step](#the-generic-step).

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
application. BART makes the same trade with `nlop_checkpoint_create_F` on its step
(`model_net.c:388`, `:404`, `:423`); the block makes it in torch, evaluating each
operator again at its saved inputs before its backward pass.

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

The step's companions come from the same place. `E^H` is applied to the data
once, in `IRGNMBlock.start`, and a state of several unknowns is laid out and taken
apart by `split` and `join` on the prepared model. Neither needs the noir model.

## The generic step

BART's step is one expression:

```
x = xn + ( DF(xn)^H DF(xn) + alpha )^-1 [ DF(xn)^H (y - F(xn)) - alpha (xn - x0) ]
```

`IRGNMBlock` evaluates it in torch and applies three BART operators of the
linearization point:

| `noir_gauss_newton_step_create_s` | In the block |
| --- | --- |
| `noir_get_forward(model)` (`:363`) | `bundle.forward` |
| `noir_get_adjoint(model)` (`:378`) | `bundle.adjoint` |
| `norm_inv_lambda_create(conf, noir_get_normal(model), ~0UL)` (`:354`) | `norm_inv_lambda` over `bundle.normal` |
| `nlop_zaxpbz_create`, `nlop_tenmul_create` (`:377`, `:380-383`, `:396`) | torch subtraction, multiplication and addition |
| `nlop_checkpoint_create_F` (`:388`) | each operator evaluated again at its saved inputs before its backward pass |

Each operator is a BART `nlop` whose backward pass is its adjoint derivative.
`norm_inv` differentiates the solve implicitly, which supplies the second-order
terms by the point. `alpha` is a vector as long as the state:
`norm_inv_lambda_create(..., ~0UL)` selects every axis (`norm_inv.c:425`).

The step asserts its state is one flat vector (`model_net.c:367`), so each
member is laid out first, as `noir_get_forward` lays out its own two:
`nlop_flatten_in` per argument and then `nlop_stack_inputs`, and not
`nlop_flatten`, which builds at BART's sixteen axes. The assertion concerns the
rank, so a model of one unknown is written this way too.

The weight decays between steps as `alpha <- (alpha - alpha_min) / redu + alpha_min`
(`noir_gauss_newton_iter_create_s`, `:402`). A block's `alpha` is the weight of
the first step, and the block taking step `k` applies it decayed `k` times in that
arithmetic. One block looped, or a stack of blocks with equal `alpha`, is BART's
schedule to the bit; a stack whose blocks learn `alpha` learns one weight per step.

BART writes the first form twice, with different arithmetic:

| | right-hand side |
| --- | --- |
| `iter4_irgnm` (`italgos.c:723-726`) | `DF^H r + alpha xref`, then `- alpha xn` |
| `noir`'s step (`model_net.c:380-383`) | `DF^H r - alpha (xn - x0)` |

The block follows `noir`. Without a centre the two reduce to one subtraction and
agree to the bit; with one they differ in the last bits.

`inner=` selects the second form, `irgnm2`. The solver minimizes
`||DF u - r||^2 + alpha ||u||^2 + R(u)` with `r = y - F(xn) + DF (xn - x0)`, and
the step returns `u + x0`. With `optim.CG()` as the solver the loop reproduces
`iter4_irgnm2` to the bit. The second form passes `alpha` to the solver as a
number, so `alpha` is held fixed there.

## Linearization at a point

The second form hands `DF(xn)` to a solver in `bartorch.optim`. `Bundle.at(xn)` is
that derivative as a `LinearOperator` holding `xn` as a tensor: its forward,
adjoint and normal pass the point to the bundle's members. An application
therefore does not depend on what was evaluated before it, and is differentiable
by the point as well as by its argument.

| Solver | Gradient by the point |
| --- | --- |
| IST, FISTA, PRIDU | through each unrolled application |
| ADMM | through the right-hand side, and the implicit x-update's term `-d/dp Re <w, N(p) x>` |
| CG | through `b = A^H y`, recorded by the operator, and the same implicit term with no penalty |

The linear autograd wrappers pass such an operator its argument instead of
recording `A` and `A^H` as an adjoint pair. A derivative lowered into the
normal-equation domain is not such a pair: its forward is `E^H E DG` and its
adjoint is `DG^H`. TGV and the infimal convolutions, which extend the variable,
are refused over a linearization. The implicit terms are exact only for an exact
inner solve; ADMM's default conjugate-gradient tolerance moves the gradient by a
few parts in a thousand per step.

Two properties of BART's operators constrain the implementation:

- Whether a node stores its derivative is a flag on the node
  (`nlop_der_requested`). An application that selects only some derivatives
  (`norm_inv`'s inner solves, a checkpoint, a stack) clears the flag on every node
  it reaches, including nodes another member shares, and a plain `nlop_apply`
  does not set it again. Both of the library's apply entries therefore select
  every derivative.
- `norm_inv` keeps copies of its arguments on the device of its first application
  (`norm_inv.c:65-70`). The prepared model holds one inverse per device.

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
`IRGNMBlock.plan(F)` names where the bundle came from (declared, linear, chain
rule, or torch), whether the step runs in the normal-equation domain, and the
linear part's own `plan`; `plan.fused` is false where the rewrite was not taken.
The executor counters stay the linear ones, because the linear encoding is what
runs.

The block takes any operator with a bundle, and the restriction list BART's
network model imposes -- `oversampling_coils=1.0`, no `optimized`, no
`oversampled_coils`, no separate coefficient shape -- does not apply. Off the grid
that includes BART's own default coil oversampling.

A batch is a leading axis on the data, and the block takes each item's step on
its own. Conjugate gradients couple through global inner products, so a batch laid
into one long state answers something else; and a leading axis on the encoding of
`CoilSense` is read as coils, which describes one shared image rather than
independent items. Applying a batch as one fused operator, as the sensitivity
batch of the linear encodings does, therefore needs two things this design does
not have: an item axis on both unknowns of the model, and a conjugate-gradient
iteration whose scalars are kept per item. Until then the cost is linear in the
batch.

A sampling pattern is `linop_gdiag_set_diag`, which writes into the existing
`cdiag` and drops its cached normal, so a composition, a gram and a model prepared
for steps all answer for the new pattern without being rebuilt. BART's own
sampling operator is already a settable diagonal -- `linop_sampling_create`
(`sense/model.c:48`) is `linop_cdiag_create(NULL)` followed by
`linop_gdiag_set_diag_ref` -- and `noir_adjoint_fft_fun` (`model_net.c:729`)
writes a pattern into `noir`'s model between calls.
On a card, a 32 by 32 four-coil model at two steps is solved three to four times
faster after a swap than after building the model again, and the two answers are
equal under `torch.equal`.

The pattern is not differentiable: BART refuses the adjoint derivative of
`noir_adjoint_fft_s` by its second input, and the pattern belongs to the model.
What pins the noir composition is the reconstruction of a phantom, measured
against the truth rather than against BART.

## Outside the form

A bundle is refused rather than approximated. What has none:

- An operator BART built that this library did not declare a bundle for and
  cannot read as a composition -- a future BART constructor, or an `nlop` handed
  in from elsewhere. Its `forward`, `derivative` and `adjoint` still work at the
  stored point, and `F.jacobian()` is its derivative there; `IRGNMBlock` refuses
  it, naming the operator.
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
| the schedule | `iter4_irgnm` without a centre and, with `optim.CG()` as the solver, `iter4_irgnm2`, both to the bit; a stack of blocks against one block looped, to the bit |
| a linearization at a point | every proximal block and `optim.CG` over `Bundle.at` against the same solver over the derivative written in torch; ADMM, CG and a whole second-form fit against central differences with exact inner solves |
| the normal-equation rewrite | the same step with `fuse=False`: to single precision on a grid, and off it to a distance that closes as the transform's tolerance is tightened, which says the transform and not the rewrite separates them |
| fusion | the plan asserted by `IRGNMBlock.plan`, on a coil composition and on one that is not |
| the device | each of the above on a card against the host, and a model stepped on the host and then on the card |

One agreement check is BART against BART and is labelled as such rather than
counted as a numerical test: a bundle's `derivative` against
`nlop_get_derivative` at the same point. It says the two routes have not
diverged; it does not say either is right. The noir composition is pinned
instead by the reconstruction of a phantom, against the truth.

## Targets

`scripts/benchmark_newton.py`: eight steps of `IRGNMBlock` over a coil composition
of 256² with eight coils, against the same steps with `fuse=False`.  One variant
runs per process, the order of the two alternates, and each cell is the range
over the repetitions of two processes.

Measured on the same laptop's host, with the card hidden.  Other work ran on the
host during part of the measurement, so the table gives only whether the ranges
separate and, where they do, the ratio of their ends:

| Case | forward | backward |
| --- | --- | --- |
| Cartesian | within the spread | within the spread |
| non-Cartesian, 401 spokes | 1.4-2.1 | 1.6-3.1 |

Measured on an otherwise idle RTX 4060 Laptop:

| Case | pass | paired (s) | normal domain (s) | ratio |
| --- | --- | --- | --- | --- |
| Cartesian | forward | 0.34-0.42 | 0.34-0.40 | within the spread |
| Cartesian | backward | 1.09-1.34 | 1.08-1.22 | within the spread |
| non-Cartesian, 401 spokes | forward | 0.62-0.64 | 0.40-0.44 | 1.5 |
| non-Cartesian, 401 spokes | backward | 1.89-2.02 | 1.25-1.44 | 1.5 |

A range is reported rather than a best because repeating one variant on the card
gives a run-to-run spread of ten to fifteen per cent: ranges that overlap resolve
no difference, and ranges that are disjoint do.  The absolute times also depend on
what else the machine is running -- a second compute job on the host cores
inflates them by tens of per cent without changing whether the ranges separate --
so the card is measured idle.

On a grid `linop_get_normal` of an FFT is the same two transforms, and neither
pass separates from the spread.  Off the grid the normal domain is faster in the
forward and in the backward pass alike.  On the grid the two domains hold the same
device memory; off it the normal domain holds a few per cent more, for the point
spread function.

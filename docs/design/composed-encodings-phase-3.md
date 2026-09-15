# Composed encodings, phase 3: handover

Phases 1 and 2 of [the design note](composed-encodings.md) are merged.  This is what phase 3
needs, what was established about it without a card, and what could not be
settled here.  It is written for someone continuing the work on a machine that
has one.

Main is at the merge of #63.  The suite is 1580 collected, 1523 passing, 58
skipped; 31 of those skips are card-gated and the rest are optional packages.

## Profile first

Nothing in phases 1 or 2 has been timed.  Every cost claim so far is *counted*
-- applications of the slab executor, read back from
`bartorch_encoding_counter` -- which shows that a composition is fused but says
nothing about what fusing bought.  Each phase 3 item is a performance claim, so
the numbers decide which of them is worth its risk.

```sh
pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON
python scripts/check_device.py                    # nine checks, in dependency order
BARTORCH_LIBRARY=... PYTHONPATH=src pytest tests/ -q     # the 31 card tests stop skipping
python scripts/benchmark_encodings.py cart3d      # one case per process
```

The cases are `cart2d`, `cart2d_sub`, `cart3d`, `cart3d_sub`,
`cart3d_sampled`, `wave3d`, `wave3d_sub`, `wave3d_sampled`, `noncart3d_sub`,
`field_cart3d`, `field_noncart3d`, `field_wave3d`.  Each prints the plan it was
lowered into beside its times, so a case that fell back is not read as if it
had not.  The numbers to beat are the targets table in [the design note](composed-encodings.md).

Two things worth measuring that no target covers:

- **A composition against its own chain.** `plan.materialise` builds the sum of
  chains deliberately (`match=False`), so the same description can be timed
  both ways in one process.  That is the direct measurement of what the phase 2
  fusion bought, and it is the number that says whether the gap below is worth
  the CUDA risk.
- **A batch on the sensitivities against one operator per item.**  `(nz, 1, c,
  y, x)` is one operator over the dataset; the comparison is a Python loop of
  `nz` operators.  On a grid and off it.

## Fewer sets of frequencies for a Cartesian axis: blocked

The design asks that a trajectory axis whose samples are whole numbers cost no
set of frequencies of its own, so a three-dimensional stack-of-stars function
decomposes into four sets rather than eight.

**This is not reachable by substituting an entry point.**  It was tried and
reverted; the evidence is specific:

| Where | What it says |
| --- | --- |
| `nufft.c:954` | `data->factors[i] = conf.decomp ? 2 : 1` -- gated per axis only by `MD_IS_SET(data->flags, i)` |
| `nufft.c:984` | the doubled grid, gated the same way |
| `nufft.c:851` | `assert(!((!conf.decomp) && conf.toeplitz))` -- `decomp` is one boolean for every axis, and a Toeplitz normal requires it |
| `nufft.c:1229` | `md_check_bounds(N, ~conf.flags, cim_dims, ksp_dims)` |

So `flags` is the only per-axis control, and it is simultaneously the layout
contract.  Clearing kz to skip the doubling makes BART compare the sample count
against the image's z extent: on a stack of stars of 4 slices and 9 spokes it
asserts at `nufft.c:1229` with 36 against 4.

`conf.cfft` has exactly the right shape -- `nufft.c:989` builds the transform
over `flags | cfft`, so a `cfft` axis is transformed without being doubled or
decomposed, and `toeplitz_mult` uses that transform.  But `nufft.c:850` asserts
the two are disjoint, and `cfft` means an axis that is a k-space axis *of the
data*, not a component of the trajectory.  Reaching it means lifting the kz
samples out of the raveled point set into a k-space axis of their own -- which
is this phase's *other* bullet (decoupling a pure stack), not a cheaper route
to this one.

Three ways forward, in increasing order of what they cost:

1. **Decouple the stack properly.**  Where the in-plane trajectory is the same
   at every kz, the operator is an FFT along kz over a batch of 2D NUFFTs.
   That is a different operator shape, not a tuning of this one: a 2-component
   trajectory, kz as a k-space axis, `conf.cfft` set for it.  It subsumes the
   coset saving and is what the design's second bullet describes.
2. **A normal operator written here**, which the "nothing here is an algorithm"
   rule weighs against but does not forbid for a substitution that exists to be
   faster.  The seam is `toeplitz_for` in `nufft_finufft.c:2470`.
3. **A BART edit**, which the design rules forbid.

Measure before choosing.  The saving is half the function's memory and half its
build transforms for a 3D stack; whether that matters is what
`noncart3d_sub` and a stack-of-stars case will say.

## Kernel stacks: larger than a stacked kernel

A per-item trajectory with the image carrying the item is declined today, and
correctly: `nufft_finufft.c:2732` refuses an image that varies along an axis
the trajectory indexes, because that needs one plan per item rather than one
plan over all of them.

```python
per_item = torch.stack([bartorch.tools.traj(x=16, y=9) for _ in range(3)])
linop.NoncartesianSense(maps, (3, 16, 16), traj=per_item)
# BartError: the images vary across frames as well as the trajectory
```

So a stacked normal kernel comes with per-item plans in the substitution, not
only a stack of functions.  Both halves are phase 3 work:

- **The plans.**  A side (`struct side`, the pair of plans and the coordinates)
  becomes a stack of sides, built per item.  FINUFFT plans them independently;
  what is shared is nothing, which is the point.
- **The function.**  `install_psf` (`nufft_finufft.c:2161`) stores one function
  into BART's operator through `noncart/nufft_priv.h`.  A stack means one per
  item and an index into it at apply time, which is BART's `data->psf` and its
  strides.

The design's caveat holds and is worth keeping in view: a subspace with a
trajectory per frame is *not* this case.  It contracts its frames, so it has
one kernel built over every frame's samples, and that already works.

## The gap: an image factor that differs between map sets

Only a slice picked whole may vary along the sets today.  Any other image-side
weight along them is left to the sum of the terms, with `plan.contraction`
reporting `chained`.  The reason is placement: the contraction is attached
around the *transform* (`sense.c:1203`, `contracted`), and the sensitivities
have summed the sets away before it is reached --
`md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, ..., d->img_strs, c->src, mstrs, map)`
at `sense.c:621`, contracting `MAPS` because `cim_dims` lacks it.

To fix it the term loop moves inside the slab, before the maps:

| Function | Line | What changes |
| --- | --- | --- |
| `forward_slab` | `sense.c:615` | multiply `c->src` by `image_l` into a buffer, then `md_ztenmul2` |
| `adjoint_slab` | `sense.c:628` | the conjugate factor after `md_zfmacc2` |
| `normal_slab` | `sense.c:666` | both |
| `normal_slab_coset` | `sense.c:687` | both, per coset |
| `normal_slab_folded` | `sense.c:708` | both |
| `forward_slab_gridded` | `sense.c:644` | the factor into the transform's load callback |
| `adjoint_slab_gridded` | `sense.c:655` | the same, store side |
| `normal_slab_gridded` | `sense.c:720` | the same |

The last three are why this was not done here.  They call
`bartorch_grid_forward_sense` / `_adjoint_sense` / `_normal_sense`
(`grid.c:1029`, `1043`, `1010`), which fold the sensitivities into the
transform's callbacks -- and the forward and adjoint of those are `#ifdef
USE_CUDA` with `error()` otherwise.  The fused path is card-only, so half this
change cannot be run on a host at all.  It lands in `sampled_fused_forward`
(`grid.c:789`), `sampled_fused_adjoint` (`grid.c:843`) and `grid_fused`
(`grid.c:517`), and from there into `grid.cuh`.

**What it buys is smaller than it looks.**  The transform count is the same
either way -- one per term, whether the terms are inside the coil loop or
outside it.  What moving them inside saves is re-walking the coil bank per
term: real where the bank is held as kernels and inflated per slab, small where
it is maps.  Time a chained contraction against a fused one before spending the
CUDA risk on it.

Where to hook the Python side: `_slice_layout` (`mri.py`) already recognises
the selector-and-phase spelling and builds `Form.slice_phase`; a general
per-set image factor would be the sibling of `_segment_layout`, with
`_Segmentable._image_dims` gaining `_layout.MAPS` again.  Phase 2 deliberately
left `MAPS` out of it, and there is a test holding that line --
`test_an_image_weight_that_differs_between_sets_is_left_to_the_sum` in
`tests/test_plan.py`.  That test is the one to change when the executor can
take it, and it exists because putting `MAPS` back *without* changing the
executor fused a wrong answer: relative error 1.2, not a small one.

## Things found the hard way

- **The trajectory's dimension vector must not carry a batch.**
  `_encoding_vector` in `sense.py` serves the k-space vector, the trajectory
  and the weights alike.  A batch on the trajectory says it varies across the
  batch, which makes it a sample axis the image also varies along, which is
  `DECLINE(16)`.  `test_the_trajectory_of_a_batch_is_shared_rather_than_one_per_item`
  pins it.
- **A fused wrong answer is the failure mode to fear.**  Both times an image
  factor was put somewhere the executor could not apply it, the operator built
  and ran and returned numbers.  Assert the plan *and* the numbers, and check
  that a test fails when the bug is reintroduced.
- **The Toeplitz normal closing with the tolerance is the signature of a
  correct function.**  1.15e-02, 3.32e-03, 5.51e-05 at a hundredth, a
  thousandth and a hundred-thousandth.  A function built over the wrong
  dimensions gives a fixed offset instead.  Use it on anything that touches the
  point spread function.
- **Card tests are marked, not skipped by accident.**  `requires_cuda` in
  `tests/test_plan.py`, `test_cuda.py`, `test_finufft.py` and
  `test_linop_mri.py`; 31 of the 58 skips.

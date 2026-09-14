# Composed MRI encodings

This is the plan for reorganising bartorch's MRI encoding operators. bartorch provides three built-in encodings, and they compose with generic base operators into any encoding. A planner then lowers each composition into the fused, streamed execution that hand-built operators have today. The design was agreed on 2026-09-14, and this note records it together with the phases that implement it.

The requirement that decides every section is that composing must cost nothing. A composed encoding has to match its hand-built counterpart within run-to-run spread on the benchmarks in [Targets](#targets). Composing in Python only builds a description. Matching and lowering happen once, when the operator is built. Each application is then one C call from host arrays to host arrays.

## Built-in operators

bartorch provides three MRI encodings. Each owns its coil sensitivities, its transform and its sampling.

| Operator | Transform | Sampling and k-space side |
| --- | --- | --- |
| `CartesianSense` | centred FFT | dense pattern, or a table of phase-encode positions; optional subspace basis |
| `NoncartesianSense` | NUFFT | trajectory, density weights; optional subspace basis |
| `WaveSense` | readout FFT, point-spread function, phase-encode FFT | dense pattern or table; optional basis. The point-spread function comes from `psf=` or from the gradient wave (`max_grad`, `max_slew`, `cycles`, `adc`, `resolution`, `offset`, `delay`, `scale`) |

All three take sensitivities as maps or as k-space kernels (`kernels=True`). Kernels are inflated a slab of coils at a time, so the bank is never resident whole, and composition keeps that. Each encoding also defines how its fast normal kernel is built ([Normal kernel construction](#normal-kernel-construction)).

Everything else is a composition with generic base operators: `Diagonal`, `Matrix`, `MultiplySum`, `Sum`, `Repeat`, `FFT`, `NUFFT`.
- `Coils` and `Sampling` stop being exported base operators, because sensitivities and sampling belong inside the encodings.
- `FieldCorrected` becomes a builder rather than an operator: it fits the segment weights with mri-nufft and returns `Σ_l Diagonal(b_l) @ E @ Diagonal(c_l)`.
- SMS, echo-resolved phase and shot phase become documented compositions, not operators.

## Encoding form

Every supported encoding reduces to one expression, for coil `c`, encoding frame `t` and sample `k`:

```
y[c, t, k] = Σ_a  O[a, t](k) · T_t( I[c, a, t](r) · x[a](r) )(k)
```

| Part | What it is | Varies along | Examples |
| --- | --- | --- | --- |
| `I` | image-side element-wise factor | voxels `r`, and any of coil `c`, contraction axis `a`, frame `t` | coil maps or coil kernels; field-correction spatial weights `c_a`; echo phase `P_t`; shot phase; per-slice maps |
| `T_t` | the transform | may vary with frame `t`, never with `c` or `a` | centred or uncentred FFT; NUFFT (a trajectory per frame); wave |
| `O` | k-space element-wise factor | samples `k`, and any of `a`, `t` | pattern or position table; density weights; field-correction temporal coefficients `b_a(k)`; SMS slice phase; subspace basis `Φ(t, a)` |
| `Σ_a` | contraction | — | subspace coefficients; field-correction segments; SMS slices and map sets |

Each axis has a fixed role:
- **Batches** lie outside the expression: independent items that share everything in it.
- **Coils** appear only in `I`, and nothing sums over them before the adjoint.
- **Contraction axes `a`** come in two kinds. The image carries `a` when it is several images: subspace coefficients, SMS slices, map sets. The image lacks it when `I` fans one image out, as in field correction.
- **Frames `t`** are the encoding axes of the samples: echoes, shots, time points.

**Basis placement.** A subspace basis `Φ(t, a)` is one number per frame and coefficient, so it commutes with the transform. The form therefore places it among the k-space factors. There it costs `K` transforms per coil rather than `T`, and it enters the normal kernel. A per-voxel phase per frame such as `exp(i·2π·B0·TE_t)` cannot move, and stays among the image-side factors.

**Sets and slices.** Map sets and SMS slices are the same contraction: the sensitivities vary along it, the image carries it and the samples do not. SMS adds a k-space phase per slice. Map sets are that same contraction with a unit phase.

In `sensitivities`, an axis directly in front of the coils is a set. A batch axis of sensitivities is written in front of a sets axis, which may have size one:
- `(3, c, y, x)` is an SMS group of three slices;
- `(nz, 1, c, y, x)` is independent slices, each with its own maps;
- per-slice kernels follow the same rule, e.g. `(nz, 1, c, ky, kx)`.

Calibration returns full-volume maps `(c, nz, ny, nx)`, which are rearranged into one of these layouts.

| Encoding | `x[a]` | `I` | `T_t` | `O` |
| --- | --- | --- | --- | --- |
| SENSE | image | maps or kernels | FFT, NUFFT or wave | pattern or table, weights |
| subspace | coefficients | maps | shared, or a trajectory per frame | `Φ(t, a)`, pattern, weights |
| field correction | one image, fanned out | maps · `c_a` | FFT, NUFFT or wave | `b_a(k)`, pattern, weights |
| SMS | slices | maps per slice | FFT, NUFFT or wave | slice phase, pattern |
| echo phase with a basis | coefficients | maps · `P_t` | FFT | `Φ(t, a)`, pattern |
| multishot | one image | maps · shot phase | FFT | positions per shot |

**Storage record.** For each factor, the form records:
- its axes and shape;
- whether it is held whole or compressed, and how it is inflated (coil kernels);
- whether it lives on the host, on page-locked host memory or on the card;
- whether it is shared by every slab and frame, or differs per coil, frame or contraction term.

It also records where the input and output live. Streaming, fusion and traffic are decided from this record alone.

**Outside the form.** A composition bartorch cannot express in this form is still applied correctly, as BART's plain chain. That covers non-element-wise operations on either side (convolutions such as SPIRiT kernels, motion warps, finite differences) and transforms other than the three. Coil compression is not among them: it is preprocessing, and virtual channels are channels.

## Matching

Composing builds a graph of bartorch operators: products `@`, sums `+`, adjoints and the base operators. The built-in encodings are leaves that expose their parts. When the operator is built, the planner walks the graph once and lowers it.

| In the composition | Becomes in the form |
| --- | --- |
| `Diagonal` (or `MultiplySum` that sums nothing) before an encoding | image-side factor `I` |
| `Diagonal` after an encoding | k-space factor `O` |
| `Repeat` before an encoding | the image fanned out along a contraction axis |
| `Matrix` or `MultiplySum` contracting an axis after an encoding | a contraction, its matrix a k-space factor |
| `Sum` over an axis after an encoding | a contraction with unit weights (SMS slices) |
| `+` of terms sharing one encoding and differing in factors | a contraction over the terms (field-correction segments) |
| `Reshape`, `Permute` between those | relabelled axes, copied only where memory order changes |

A subgraph that does not fit stays BART's plain chain, and the fused parts around it are unaffected.

The chosen plan is never silent:
- The operator reports it, e.g. `A.plan` naming the transform, the factors on each side, the contraction, the streaming and the normal kernel.
- C counters record which executor path ran.
- Tests assert both that the result matches the plain chain and that the fused plan was taken. A silent fallback passes every correctness test; the first native field correction did exactly that.

## Normal kernel construction

The fast normal is the core of the design. Its kernel collapses the k-space side over the samples at each place of a grid, once, when the operator is built:

```
K[a', a](place) = Σ_t Σ_(k at place)  conj(O[a', t](k)) · O[a, t](k)
```

| Transform | "Place" | Construction |
| --- | --- | --- |
| FFT | a grid point of the axes the pattern varies along | pattern-and-basis kernel per kept place |
| wave | the same, on the (readout, phase-encode) grid | the grid kernel; the point-spread function stays in the transform |
| NUFFT | a point of the 2× grid | point-spread function: a type-1 NUFFT of the pair weights, packed upper triangle, real when the pair products are real, built a coset at a time |

**When no kernel exists.** If an image-side factor varies along a frame axis, as echo phase `P_t(r)` does, there is no kernel. The normal is then applied frame by frame, still in the coil loop.

**Kernel stacks.** Sometimes the image carries an axis the samples also carry, and each item along it has its own trajectory or pattern. Examples are dynamic frames without a subspace, slices of a multislice acquisition, and both together.
- `K` is then a stack indexed by that axis, built one item at a time.
- It is one operator over the whole dataset, so one solve uses the redundancy between items rather than serialising a solve per item.
- A subspace with a trajectory per frame is different. It contracts its frames, so it still has one kernel, built over all frames' samples.

**Cartesian-aligned axes.** Each trajectory coordinate is tested on its own when the operator is built: if every sample's value is an integer within tolerance, the axis is Cartesian. Stack-of-stars and stack-of-spirals along kz are the usual case.
- A Cartesian axis is not oversampled, so a 3D kernel has 2^(non-Cartesian axes) cosets: 4 instead of 8.
- If the in-plane trajectory is also the same at every position along that axis, an FFT along it decouples the problem into 2D problems batched over it. The readout already decouples a grid this way.

## Normal kernel application

Once `K` exists, applying the normal does not depend on the encoding. Per slab of coils, and per coset for a NUFFT:

1. **Forward transform, per term `a`.** The load callback multiplies the image-side factors in, and the store callback gathers the kept places into a bank.
2. **Contraction.** The bank is contracted with `K`, taking the stack item's kernel where the kernel is stacked. The kernel may be the packed triangle, real or complex, in full precision or bfloat16.
3. **Inverse transform, per term.** The load callback scatters the bank, and the store callback multiplies the conjugate image-side factors and adds into the result.

FFT, NUFFT and wave differ only in the transform plan and in how places are indexed.

## Streaming

Any axis that no factor or contraction mixes is walked a slab at a time: coils, batches, kernel-stack items. The slab size is a setting, and its default is the memory-minimal one.

## Fusion

A factor adjacent to the transform, on either side, is folded into the transform's load and store callbacks, and consecutive factors on one side merge into one pass. A factor that cannot be folded becomes one fused element-wise kernel, never a separate step of BART's chain.

## Placement and traffic

- **Shared factors** are uploaded once per application.
- **Per-coil and per-item data** is streamed a slab at a time.
- **Compressed factors** (coil kernels, coset functions) are inflated or streamed per slab.
- **Around each application:** staging is page-locked, two streams run overlapping slabs, and BART's memory cache is handed back afterwards.

## Executor

One C slab executor runs every matched form. It is today's pieces parameterised by the form instead of hard-coded per operator:

| Today | Where |
| --- | --- |
| coil loop, slab streaming, map folding, segments around a slab | `src/csrc/ops/sense.c` |
| grid transform: pattern and basis kernel, sampled tables, wave front, uncentred convention | `src/csrc/ops/grid.c`, `grid.cuh` |
| NUFFT substitution, Toeplitz point-spread function, cosets, streaming, basis along samples | `src/csrc/substitute/nufft_finufft.c`, `psf.c` |
| callbacks and device kernels | `src/csrc/ops/fft_callbacks.cu`, `fft_callbacks_lto.cu`, `kernels.cu` |
| Python encodings | `src/bartorch/linop/mri.py` (`_CartesianNative`, `_CartesianSampled`, `_WaveNative`, `_Segmentable`, `_SegmentedSense`, `FieldCorrected`), `src/bartorch/linop/sense.py` (`NoncartesianSense`, `Coils`) |
| composition nodes | `src/bartorch/linop/base.py` (`_Compose`, `_Add`, `_WithNormal`) |

## Phases

1. **Planner over today's encodings.**
   - Build the form, matching and the executor's parameterisation.
   - Re-express `CartesianSense`, `NoncartesianSense`, `WaveSense` and `FieldCorrected` through them.
   - Done when the existing tests pass unchanged, the counters show the fused plan for every case in [Targets](#targets), and the benchmarks match within spread.
2. **Compositions.**
   - Field correction, SMS, echo phase with a basis and multishot, written as compositions with no C code specific to any of them.
   - Accept a batch axis on sensitivities (in front of a sets axis).
   - Remove `Coils` and `Sampling` from the exports.
   - Each is tested against its written-out sum and its plain chain, with the fused plan asserted.

   Done: `@` and `+` defer, so the planner is offered the whole description
   rather than one factor at a time, and field correction, multishot and echo
   phase with a basis each lower into one encoding whose contraction is their
   terms.  `Coils` and `Sampling` are off the exports.

   **SMS does not fit the executor as phase 1 built it.** The form places the
   contraction outside the transform with `O[a, t]` free to depend on `a`,
   which is what SMS needs; the executor attaches the contraction around the
   transform, over the *coil* images, and `md_ztenmul2` in `forward_slab` has
   contracted the sets away before the image factor is reached. So a factor
   that differs between sets cannot reach the place it belongs, and the
   planner leaves such a sum chained rather than fusing a wrong answer.

   The fix keeps MAPS through the transform where the form carries a k-space
   factor along it, and sums after that factor with `linop_sum_create` --
   one transform per slice, which is what the sum being outside the transform
   costs. Still outstanding, with the batch axis on sensitivities.
3. **Kernel stacks and Cartesian-aligned axes.**
   - Stacked kernels for per-item trajectories.
   - Detection of Cartesian-aligned trajectory axes, with fewer cosets and, for pure stacks, decoupling along the axis.

## Targets

Measured on an RTX 4060 laptop with host arrays, 8 coil kernels, and a reused output buffer. Ranges are the minimum and maximum over repeats. The script is `scripts/benchmark_encodings.py`.

| Case | Forward (s) | Adjoint (s) | Normal (s) |
| --- | --- | --- | --- |
| Cartesian 3D 256³ | 0.32-0.34 | 0.37-0.41 | 0.19-0.31 |
| Cartesian 3D subspace 256³, dense pattern | — | — | 0.72-0.87 |
| Cartesian 3D subspace 256³, positions | 0.54-0.69 | 0.67-0.76 | 0.65-0.83 |
| wave 192³, 3× readout | 0.42-0.54 | 0.44-0.58 | 0.25-0.45 |
| wave subspace 192³, dense pattern | — | — | 0.99-1.04 |
| wave subspace 192³, positions | 0.83-0.97 | 1.15-1.23 | 0.92-1.09 |
| non-Cartesian subspace 256³, 500 frames × 48 shots | — | — | 1.54-1.56 |
| field-corrected non-Cartesian 160³, 6 segments | 0.46-0.49 | 0.76-0.84 | 1.84-1.96 |
| field-corrected EPI-like Cartesian 160³, 6 segments | 0.26-0.28 | 0.26-0.32 | 0.37-0.41 |
| field-corrected wave 160³, 6 segments | 0.82-0.98 | 0.79-0.89 | 1.36-1.39 |

## Working constraints

- **Environment.** The implementing session has no GPU. It runs the CPU suite as `AGENTS.md` describes. Card tests are written alongside the code and marked as needing a card; they and the benchmarks run on the laptop afterwards.
- **BART.** It is not edited. What bartorch changes goes into `src/csrc/substitute/` or `src/csrc/ops/`.
- **Tests.** They pin numbers against something outside BART: torch reference computations, explicit sums, a written-out model.
- **Memory.** Defaults are memory-minimal everywhere, including on large cards.
- **Comments and docstrings.** They describe the code as it is.
- **Commits.** One phase per commit series, on a branch; nothing is pushed to `main` without review.
- **Deferred.** Calibration of the wave point-spread function is issue #56.

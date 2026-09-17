"""
=================================
Subspace-constrained T1 mapping
=================================

An inversion-recovery FLASH acquisition of four hundred frames, one spoke each,
reconstructed into the coefficients of a signal subspace and fitted for
:math:`T_1`.

A single spoke does not determine a frame. What makes the series recoverable is
that the frames are not arbitrary: every voxel follows an inversion-recovery
curve, and the curves of every plausible :math:`T_1` span a subspace of
dimension four or so. Writing the unknown series as :math:`x_t = \\sum_a
\\Phi_{at} c_a` turns four hundred images into four coefficient maps, and the
basis :math:`\\Phi` enters the encoding on the k-space side, after the
transform and before the samples:

.. math::

   y[c, t] = \\sum_a \\Phi_{at} \\, \\mathrm{NUFFT}_t \\!\\left( S_c \\, c_a \\right).

The subspace is estimated from a simulated dictionary, which is also what the
parameter fit matches against.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/01-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.

Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. *T2
shuffling: sharp, multicontrast, volumetric fast spin-echo imaging.* Magn Reson
Med 77(1):180-195 (2017).
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.constrained_layout.use": True,
    }
)

PAGE_WIDTH = 8.0  # inches, the width of the documentation column


def panels(rows, columns, height=1.0):
    """A grid of square image panels filling the documentation column."""
    side = PAGE_WIDTH / columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(PAGE_WIDTH, rows * side * height + 0.4)
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    return figure, axes


# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 128
COILS = 8
FRAMES = 400
RANK = 4
SLICE = 90

TR = 0.0041  # s
TE = 0.0021  # s
FLIP = 6.0  # degrees

#: The tissue classes of the BrainWeb segmentation this example keeps, by the
#: label they carry in its table.
TISSUES = {1: "CSF", 2: "grey matter", 3: "white matter", 8: "glial matter"}

# %%
#
# The dictionary and its subspace
# -------------------------------
#
# :func:`bartorch.tools.signal` evaluates BART's analytical signal models.
# ``F`` selects FLASH and ``I`` the inversion-recovery preparation in front of
# it; ``flag_1`` is BART's ``-1``, the range of :math:`T_1` values as
# ``min:max:count``. BART steps it by ``(max - min) / count`` from ``min``, so
# the last entry falls one step short of ``max``; the fit below indexes the
# same values. The result is one curve per entry, sampled at the frame times
# the repetition time implies.

T1_RANGE = (0.1, 4.5, 200)  # s, as BART's -1 takes it
t1_values = T1_RANGE[0] + (T1_RANGE[1] - T1_RANGE[0]) / T1_RANGE[2] * torch.arange(T1_RANGE[2])

dictionary = bt.signal(F=True, I=True, r=TR, e=TE, f=FLIP, n=FRAMES, flag_1=T1_RANGE).squeeze()

left = torch.linalg.svd(dictionary.T.to(torch.complex64), full_matrices=False)[0]
basis = left[:, :RANK].T.contiguous()

print(f"dictionary {tuple(dictionary.shape)}, basis {tuple(basis.shape)}")

# %%
#
# The singular values of the dictionary say how many coefficients a
# reconstruction has to carry. The rank chosen here is where they fall below a
# hundredth of the first, which is a modelling decision rather than a
# measurement: too few coefficients bias the recovered curves toward the
# dictionary, too many spend the undersampling factor the subspace was meant to
# buy.

# sphinx_gallery_start_ignore
spectrum = torch.linalg.svdvals(dictionary.T.to(torch.complex64))
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.55, 2.8))
axis.semilogy(range(1, 13), (spectrum[:12] / spectrum[0]).cpu().numpy(), marker="o", ms=4)
axis.axvline(RANK + 0.5, color="0.6", ls="--")
axis.set_xlabel("singular value")
axis.set_ylabel("magnitude, relative to the first")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Phantom
# -------
#
# Each tissue class is given the :math:`T_1` of the BrainWeb table and its own
# curve from the same signal model, and the series is the membership-weighted
# sum of them. A voxel holding two tissues therefore follows a sum of two
# recovery curves, which is not itself an inversion-recovery curve -- the
# partial-volume error any voxelwise fit carries.

# sphinx_gallery_start_ignore
table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
t1_table = np.array([float(row["T1 (ms)"]) for row in rows]) / 1000.0
proton_density = np.array([float(row["PD (ms)"]) for row in rows])

fractions = get_mri(sub_id=0, contrast="fuzzy")[:, :, SLICE]
fractions = np.flip(fractions.transpose(1, 0, 2), 0).copy()


def resampled(values):
    grid = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


memberships = torch.stack([resampled(fractions[..., label]) for label in TISSUES])
series = torch.zeros(FRAMES, SIZE, SIZE, dtype=torch.complex64)
density = torch.zeros(SIZE, SIZE)
for index, label in enumerate(TISSUES):
    curve = bt.signal(
        F=True,
        I=True,
        r=TR,
        e=TE,
        f=FLIP,
        n=FRAMES,
        flag_1=(float(t1_table[label]), float(t1_table[label]), 1),
    ).squeeze()
    weighted = memberships[index] * float(proton_density[label])
    series += weighted[None].to(torch.complex64) * curve[:, None, None]
    density += weighted
# sphinx_gallery_end_ignore

# %%
#
# Acquisition and reconstruction
# ------------------------------
#
# One golden-angle spoke per frame. The trajectory indexes frames as well as
# samples, and the image the encoding operator maps from is the four
# coefficient maps rather than the four hundred frames, with ``basis``
# contracting the one into the other.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, 1, SIZE, 3)

# sphinx_gallery_start_ignore
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

frames = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(frames(series), n=1e-7, s=5)

A = linop.NoncartesianSense(sensitivities, (RANK, SIZE, SIZE), traj=trajectory, basis=basis)
print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.contraction`` reports the subspace and its rank, and the normal
# operator is a point spread function over the basis as well as the
# trajectory, so a subspace reconstruction costs per iteration what a plain one
# costs.

data = measured / optim.data_scaling(measured[..., None], A=A)
coefficients = optim.ADMM(priors.Wavelet(axes=(-1, -2), weight=0.002), maxiter=30)(data, A)

recovered = torch.einsum("af,ayx->fyx", basis.to(torch.complex64), coefficients)

# %%
#
# Parameter fit
# -------------
#
# The recovered coefficients are matched against the dictionary projected onto
# the same subspace, by the normalized inner product, which is dictionary
# matching performed in four dimensions rather than four hundred. Matching in
# the subspace and matching the reconstructed curves differ only by the
# component of the dictionary the basis discards.

atoms = basis.to(torch.complex64) @ dictionary.T.to(torch.complex64)
atoms = atoms / atoms.norm(dim=0, keepdim=True)
voxels = coefficients.reshape(RANK, -1)
voxels = voxels / voxels.norm(dim=0, keepdim=True).clamp(min=1e-12)

matched = (atoms.conj().T @ voxels).abs().argmax(0)
t1_map = t1_values[matched].reshape(SIZE, SIZE)

# %%
#
# The fit is reported where the proton density is high enough for a curve to be
# defined, and separately for the voxels each tissue class dominates.

support = density > 0.2 * float(density.max())
dominant = memberships.argmax(0)
pure = memberships.max(0).values > 0.7

for index, label in enumerate(TISSUES):
    selected = support & pure & (dominant == index)
    if int(selected.sum()) < 20:
        continue
    estimate = 1000.0 * float(t1_map[selected].median())
    print(
        f"{TISSUES[label]:>14}  table {1000 * t1_table[label]:6.0f} ms"
        f"   fitted {estimate:6.0f} ms   ({int(selected.sum())} voxels)"
    )

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 4)
for column in range(RANK):
    magnitude = coefficients[column].abs()
    axes[0, column].imshow(
        (magnitude / magnitude.max()).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0
    )
    axes[0, column].set_title(f"coefficient {column}")
figure.suptitle("subspace coefficient maps, each on its own scale")

truth = (
    memberships * torch.as_tensor(t1_table[list(TISSUES)], dtype=torch.float32)[:, None, None]
).sum(0)
truth = truth / memberships.sum(0).clamp(min=1e-6)

figure, axes = panels(1, 3)
for axis, values, title in (
    (axes[0, 0], truth, "membership-weighted $T_1$"),
    (axes[0, 1], torch.where(support, t1_map, torch.zeros(())), "fitted $T_1$"),
):
    handle = axis.imshow(
        torch.where(support, values, torch.zeros(())).cpu().numpy(),
        cmap="magma",
        vmin=0.0,
        vmax=3.0,
    )
    axis.set_title(title)
figure.colorbar(handle, ax=axes[0, 1], label="$T_1$ [s]", fraction=0.046)

voxel = torch.nonzero(support & pure & (dominant == 2))
voxel = voxel[len(voxel) // 2]
axes[0, 2].remove()
axis = figure.add_subplot(1, 3, 3)
truth_curve = series[:, voxel[0], voxel[1]].real
fitted_curve = recovered[:, voxel[0], voxel[1]].real
# The reconstruction determines the curve up to a global scale, which is
# divided out here so the two can be read against each other.
fitted_curve = fitted_curve * float((truth_curve * fitted_curve).sum() / (fitted_curve**2).sum())
axis.plot(truth_curve.cpu().numpy(), lw=1.2, label="phantom")
axis.plot(fitted_curve.cpu().numpy(), lw=1.2, label="recovered")
axis.set_xlabel("frame")
axis.set_ylabel("signal [a.u.]")
axis.legend(fontsize=9)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The first two coefficient maps carry the magnetization and its recovery and
# look like images; the third and fourth carry what the first two cannot
# represent and look like nothing in particular, as a basis estimated from a
# dictionary rather than from anatomy will.
#
# The recovered curve is the reconstruction's, which determines it only up to a
# global scale -- the data was normalized before the solve -- so the panel
# beside the maps compares the two after dividing that scale out. The fit is
# invariant to it, being a normalized inner product.
#
# The fitted :math:`T_1` agrees with the table in the voxels one tissue
# dominates. It is biased where two tissues meet, because the sum of two
# recovery curves is not a recovery curve, and it is bounded by the range the
# dictionary covers -- a fit cannot return a value it was never offered.
#
# Estimating the parameters directly from k-space, without an intermediate
# series or a subspace, is :doc:`../04-model-based/02-quantitative-models`.

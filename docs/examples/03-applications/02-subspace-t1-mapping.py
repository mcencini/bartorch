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
from cmap import Colormap
from matplotlib.colors import ListedColormap

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
# mygbm, turned so that zero phase is yellow and +/-pi is blue.
MYGBM = Colormap("colorcet:CET_C2").to_matplotlib().reversed()
PHASE = ListedColormap(MYGBM([((step + 60) % 256) / 255 for step in range(256)]))

# Colormap, window and unit per parameter.  Both relaxation windows stop short
# of cerebrospinal fluid, so that white and grey matter -- 500 against 833 ms
# in T1, 70 against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "$T_1$ [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "$T_2$ [ms]"),
}

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


def show(axis, values, title=None, vmax=None, cmap="gray", vmin=0.0):
    """One panel, of a magnitude by default."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


def parameter(axis, values, name, title=None):
    """One relaxation map in the colormap and window its parameter is read in."""
    cmap, limits, _ = STYLE[name]
    return show(axis, values, title, vmax=limits[1], cmap=cmap, vmin=limits[0])


def domain(axis, values, title=None):
    """A complex map the way a coil sensitivity is read: phase in colour,
    magnitude in brightness, so an unsupported corner reads as background
    rather than as a phase."""
    values = values.detach().cpu()
    colours = PHASE((values.angle() / (2 * np.pi) + 0.5).numpy())[..., :3]
    magnitude = values.abs().numpy()
    magnitude = magnitude / max(float(magnitude.max()), 1e-12)
    axis.imshow(colours * magnitude[..., None])
    if title is not None:
        axis.set_title(title)


def scalebar(figure, axes, handle=None, label=None, name=None):
    """One colorbar for a group of panels, so none gives up width to its own."""
    if name is not None:
        cmap, limits, label = STYLE[name]
        handle = plt.cm.ScalarMappable(plt.Normalize(*limits), cmap)
    bar = figure.colorbar(handle, ax=axes, fraction=0.046, label=label)
    return bar


def phase_bar(figure, axes):
    """The colour-to-phase key for the panels beside it."""
    bar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), PHASE),
        ax=axes,
        fraction=0.046,
        ticks=[-np.pi, 0.0, np.pi],
    )
    bar.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    bar.set_label("phase [rad]")


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


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

TR = 0.0041  # s
TE = 0.0021  # s
FLIP = 6.0  # degrees

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
# The phantom, the relaxation maps behind it and the coil sensitivities, built
# as :doc:`/auto_examples/01-basics/01-from-kspace-to-image` builds them.
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

# BrainWeb's volume is indexed (inferior-superior, posterior-anterior,
# left-right), so its first axis selects an axial slice; an image is drawn from
# its first row down, so flipping puts anterior at the top.
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# A square field of view around the head, with a margin for the aliasing of an
# undersampled acquisition to fold into.
occupied = np.nonzero(fractions.sum(-1) > 0.5)
middle = [int((axis.min() + axis.max()) / 2) for axis in occupied]
half = int(round((1 + MARGIN) * max(axis.max() - axis.min() for axis in occupied) / 2))
source = tuple(
    slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, fractions.shape[:2])
)
box = np.zeros((2 * half, 2 * half, fractions.shape[-1]), dtype=np.float32)
box[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))] = (
    fractions[source]
)
memberships = torch.nn.functional.interpolate(
    torch.as_tensor(box).permute(2, 0, 1)[None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0]

# Where each class sits in ``memberships``, by the name the table gives it.
CLASS = {entries[label]["Tissue"]: index for index, label in enumerate(TISSUES)}

weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

# A T1-weighted spin echo, at a repetition time of 600 ms and an echo time of
# 12 ms.
signal = (
    proton_density
    * (1 - torch.exp(-600.0 / T1.clamp(min=1e-3)))
    * torch.exp(-12.0 / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()

# A smooth quadratic phase, so that nothing depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
# One inversion-recovery curve per tissue class, from the same signal model the
# dictionary came from, combined by membership and proton density.
series = torch.zeros(FRAMES, SIZE, SIZE, dtype=torch.complex64)
occupancy = torch.zeros(SIZE, SIZE)
for index in range(len(TISSUES)):
    seconds = float(tissue_t1[index]) / 1000.0
    curve = bt.signal(
        F=True, I=True, r=TR, e=TE, f=FLIP, n=FRAMES, flag_1=(seconds, seconds, 1)
    ).squeeze()
    weighted = memberships[index] * float(tissue_pd[index])
    series += weighted[None].to(torch.complex64) * curve[:, None, None]
    occupancy += weighted
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
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
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
#
# The penalty is locally low rank: the coefficient maps are stacked into a
# matrix per block of voxels, and its nuclear norm is penalized.
# ``joint_axes`` is what makes the coefficients the columns of that matrix,
# which is what states the thing a subspace reconstruction knows and a
# wavelet penalty does not -- that neighbouring voxels follow the *same* few
# curves, not that each coefficient map is separately sparse. Penalizing the
# maps one at a time instead leaves the coefficients free to disagree with
# each other, and the recovered curves with them.

data = measured / optim.data_scaling(measured[..., None], A=A)
term = priors.LocallyLowRank(axes=(-1, -2), weight=0.005, joint_axes=(-3,), block=8)
coefficients = optim.ADMM(term, maxiter=30)(data, A)

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

support = occupancy > 0.2 * float(occupancy.max())
dominant = memberships.argmax(0)
pure = memberships.max(0).values > 0.7

for name, index in CLASS.items():
    selected = support & pure & (dominant == index)
    if int(selected.sum()) < 20:
        continue
    estimate = 1000.0 * float(t1_map[selected].median())
    print(
        f"{name:>13}  table {tissue_t1[index]:6.0f} ms"
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

figure, axes = panels(1, 3)
for axis, values, title in (
    (axes[0, 0], T1, "membership-weighted $T_1$"),
    (axes[0, 1], 1000.0 * t1_map, "fitted $T_1$"),
):
    parameter(axis, torch.where(support, values, torch.zeros(())), "T1", title)
scalebar(figure, axes[0, 1], name="T1")

voxel = torch.nonzero(support & pure & (dominant == CLASS["WM"]))
voxel = voxel[len(voxel) // 2]
axes[0, 2].remove()
axis = figure.add_subplot(1, 3, 3)
truth_curve = series[:, voxel[0], voxel[1]].real
fitted_curve = recovered[:, voxel[0], voxel[1]].real
# The reconstruction determines the curve up to a global scale, which is
# divided out here so the two can be read against each other.
fitted_curve = fitted_curve * float((truth_curve * fitted_curve).sum() / (fitted_curve**2).sum())
axis.plot(truth_curve.cpu().numpy(), lw=2.4, color="0.7", label="phantom")
axis.plot(fitted_curve.cpu().numpy(), lw=1.0, ls="--", color="C1", label="recovered")
axis.set_xlabel("frame")
axis.set_ylabel("signal [a.u.]")
axis.legend(fontsize=9)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The coefficient maps are not images of anything: each is the weight of one
# singular vector of the dictionary, and only the first carries the
# magnetization in a form an eye reads. The later ones carry what the earlier
# ones cannot represent, which is a difference between recovery curves rather
# than a tissue.
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

"""
================================
Dynamic golden-angle radial MRI
================================

A continuously acquired golden-angle radial scan reconstructed as a time
series, with a temporal regularizer compensating for the undersampling of each
frame.

The acquisition is one uninterrupted train of spokes, each rotated from the
last by the golden angle [#winkelmann]_. Frames are cut out of it afterwards: any block of
consecutive spokes covers k-space approximately uniformly, so the frame
duration is a reconstruction parameter rather than an acquisition parameter.
Thirteen spokes across a 128 matrix is fifteenfold undersampled, and no frame
is invertible on its own; the series is recoverable because the frames are not
independent, which a total variation penalty along time states.

This is the encoding of :doc:`../02-non-cartesian/02-radial-sense` with one
axis added: the image is ``(frames, y, x)``, the trajectory indexes frames as
well as shots, and the sensitivities are shared across all of them.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/01-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.
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
FRAMES = 16
SPOKES = 13  # per frame

# %%
#
# A contrast-enhanced series
# --------------------------
#
# The phantom is the BrainWeb segmentation again, with a bolus passing through
# it: a gamma-variate enhancement curve applied to each tissue class in
# proportion to its vascularity, strongest in grey matter, weaker in white
# matter and absent in cerebrospinal fluid. The series is therefore piecewise
# smooth in time with a spatial structure that is the same in every frame, which
# is the structure the temporal penalty uses.

CLASSES = {"grey matter": ("GM", 0.8), "white matter": ("WM", 0.25)}

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
# A gamma-variate bolus, and the static image enhanced by it where each class
# is vascular.
moment = torch.linspace(0.0, 1.0, FRAMES)
bolus = (moment / 0.25) ** 3 * torch.exp(3.0 - 3.0 * moment / 0.25)
bolus = bolus / bolus.max()

series = image[None].repeat(FRAMES, 1, 1)
for label, weight in CLASSES.values():
    enhancing = (memberships[CLASS[label]] * image).to(torch.complex64)
    series = series + weight * enhancing[None] * bolus[:, None, None]
# sphinx_gallery_end_ignore

# %%
#
# Acquisition
# -----------
#
# ``FRAMES * SPOKES`` spokes are generated as one golden-angle trajectory and
# then reshaped, so that the first axis indexes frames and the second the shots
# within a frame. The image varies along the frame axis, so each frame has its
# own non-uniform FFT and normal kernel inside the one operator, applied under
# the same coil loop.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES * SPOKES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, SPOKES, SIZE, 3)

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

A = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(A(series), n=1e-6, s=3)

print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.items`` is the number of frames the encoding carries, each with its
# own transform.
#
# Reconstruction
# --------------
#
# Two reconstructions of the same data. The first treats the frames as
# independent: the adjoint of the encoding applied to density-compensated
# samples, which is the gridding reconstruction of thirteen spokes per frame.
# The second solves the whole series at once under a total variation penalty
# along the frame axis, which states that the signal is constant in time except
# at a few instants -- the reconstruction GRASP performs [#feng]_.

weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1).clamp(min=0.25)
gridded = A.H(measured * weights.to(torch.complex64))

data = measured / optim.data_scaling(measured[..., None], A=A)
temporal = optim.ADMM(priors.TotalVariation(axes=(-3,), weight=0.02), maxiter=30)(data, A)

# %%
#
# The frame axis is ``-3``, the axis in front of the two spatial ones; a term
# given ``(-1, -2)`` instead would penalize the spatial gradient, and one given
# all three penalizes both. Which axes a term acts on is the whole difference
# between a spatial and a temporal regularizer.

# %%

# sphinx_gallery_start_ignore
chosen = (0, FRAMES // 3, 2 * FRAMES // 3, FRAMES - 1)
figure, axes = panels(3, len(chosen))
top = float(series.abs().max())
for column, frame in enumerate(chosen):
    show(axes[0, column], series[frame], f"frame {frame}", vmax=top)
    show(axes[1, column], scaled(gridded[frame], series[frame]), vmax=top)
    show(axes[2, column], scaled(temporal[frame], series[frame]), vmax=top)
for row, label in enumerate(("phantom", "gridding", "temporal TV")):
    axes[row, 0].set_ylabel(label)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The quantity a perfusion study reports is the signal in a region as a
# function of time, so the reconstructions are compared on that curve. The
# region here is the grey matter, where the enhancement was applied.

region = memberships[CLASS["GM"]] > 0.6

curves = {
    "phantom": series.abs(),
    "gridding": scaled(gridded, series),
    "temporal TV": scaled(temporal, series),
}
truth = curves["phantom"][:, region].mean(-1)
for name, volume in curves.items():
    if name == "phantom":
        continue
    enhancement = volume[:, region].mean(-1)
    curve = float((enhancement - truth).norm() / truth.norm())
    frames = bt.nrmse(series.abs(), volume, scaled=True)
    print(f"{name:>12}  curve NRMSE {curve:.3f}   frame NRMSE {frames:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.0))
for name, volume in curves.items():
    axis.plot(range(FRAMES), volume[:, region].mean(-1).cpu().numpy(), marker="o", ms=3, label=name)
axis.set_xlabel("frame")
axis.set_ylabel("mean signal in grey matter [a.u.]")
axis.legend()
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Gridding recovers the shape of the enhancement curve, since the streaks of a
# radial acquisition are spread over the image rather than concentrated where
# the signal is, but carries the frame-to-frame variation of the streak pattern
# into it. The regularized reconstruction is smoother in time by construction,
# which reduces noise and also biases the curve: a change confined to one frame
# is attenuated by a temporal total variation penalty more than any other.

# %%
#
# References
# ----------
#
# .. [#winkelmann] Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal
#    radial profile order based on the Golden Ratio for time-resolved MRI.
#    *IEEE Trans Med Imaging* 26(1):68-76 (2007).
#    https://doi.org/10.1109/TMI.2006.885337
#
# .. [#feng] Feng L, Grimm R, Block KT, Chandarana H, Kim S, Xu J, Axel L, Sodickson DK,
#    Otazo R. Golden-angle radial sparse parallel MRI: combination of
#    compressed sensing, parallel imaging, and golden-angle radial sampling for
#    fast and flexible dynamic volumetric MRI. *Magn Reson Med* 72(3):707-717
#    (2014). https://doi.org/10.1002/mrm.24980

"""
================================
Dynamic golden-angle radial MRI
================================

A continuously acquired golden-angle radial scan reconstructed as a time
series, with a temporal regularizer in place of the temporal resolution the
undersampling destroys.

The acquisition is one uninterrupted train of spokes, each rotated from the
last by the golden angle. Frames are cut out of it afterwards, which is what
the golden angle buys: any block of consecutive spokes covers k-space
approximately uniformly, so the frame duration is a reconstruction parameter
rather than an acquisition parameter. Thirteen spokes across a 128 matrix is
fifteenfold undersampled, and no frame is invertible on its own; what makes the
series recoverable is that the frames are not independent, which a total
variation penalty along time states.

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


def show(axis, values, title=None, vmax=None, cmap="gray"):
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


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
SLICE = 90

# %%
#
# A contrast-enhanced series
# --------------------------
#
# The phantom is the BrainWeb segmentation again, with a bolus passing through
# it: a gamma-variate enhancement curve applied to each tissue class in
# proportion to its vascularity, strongest in grey matter, weaker in white
# matter and absent in cerebrospinal fluid. The series is therefore piecewise
# smooth in time and identical in space from frame to frame, which is the
# structure the reconstruction will exploit.

CLASSES = {"CSF": (1, 0.0), "grey matter": (2, 0.8), "white matter": (3, 0.25)}

# sphinx_gallery_start_ignore
table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
proton_density = np.array([float(row["PD (ms)"]) for row in rows], dtype=np.float32)

fractions = get_mri(sub_id=0, contrast="fuzzy")[:, :, SLICE]
fractions = np.flip(fractions.transpose(1, 0, 2), 0).copy()


def resampled(values):
    grid = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


time = torch.linspace(0.0, 1.0, FRAMES)
bolus = (time / 0.25) ** 3 * torch.exp(3.0 - 3.0 * time / 0.25)
bolus = bolus / bolus.max()

static = resampled(fractions @ proton_density)
peak = float(static.max())
series = (static / peak)[None].repeat(FRAMES, 1, 1)
for label, weight in CLASSES.values():
    occupancy = resampled(fractions[..., label]) * proton_density[label] / peak
    series = series + weight * occupancy[None] * bolus[:, None, None]
series = series.to(torch.complex64)
# sphinx_gallery_end_ignore

# %%
#
# Acquisition
# -----------
#
# ``FRAMES * SPOKES`` spokes are generated as one golden-angle trajectory and
# then reshaped, so that the first axis indexes frames and the second the shots
# within a frame. A trajectory with an encoding axis is one transform over all
# of its samples rather than one transform per frame, which is what lets a
# single plan serve the whole series.

trajectory = bt.traj(readout=SIZE, spokes=FRAMES * SPOKES, radial=True, golden=True)
trajectory = trajectory.reshape(FRAMES, SPOKES, SIZE, 3)

# sphinx_gallery_start_ignore
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

A = linop.NoncartesianSense(sensitivities, (FRAMES, SIZE, SIZE), traj=trajectory)
measured = bt.noise(A(series), n=1e-6, s=3)

print(f"{A.ishape} -> {A.oshape}")
print(A.plan)

# %%
#
# ``plan.items`` is the number of frames the encoding carries, and the
# transform is planned once for all of them.
#
# Reconstruction
# --------------
#
# Two reconstructions of the same data. The first treats the frames as
# independent: the adjoint of the encoding applied to density-compensated
# samples, which is the gridding reconstruction of thirteen spokes per frame.
# The second solves the whole series at once under a total variation penalty
# along the frame axis, which states that the signal is constant in time except
# at a few instants -- the reconstruction GRASP performs.

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
# What the reconstruction is for is the curve, not the frame: the quantity a
# perfusion study reports is the signal in a region as a function of time. The
# region here is the grey matter, where the enhancement was applied.

region = (
    torch.nn.functional.interpolate(
        torch.as_tensor(np.ascontiguousarray(fractions[..., 2], dtype=np.float32))[None, None],
        size=(SIZE, SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    > 0.6
)

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
    frames = bartorch.nrmse(series.abs(), volume, scaled=True)
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
# which is a bias as much as it is a denoiser: a change confined to one frame
# is what a total variation penalty along time is least likely to keep.

"""
=============================
Preprocessing and evaluation
=============================

The array operations around a reconstruction: prewhitening the channels,
aligning repetitions, changing the field of view, measuring the result, and
exchanging data with BART's file format.

None of these is a reconstruction, and each of them changes what a
reconstruction means. Least squares assumes white noise; a repetition that
moved is a different object; an error measured against a differently scaled
reference is not an error. The functions here are in the ``bartorch``
namespace, take axis indices rather than BART bitmasks, and return tensors.

The phantom and the coil sensitivities are built as in
:doc:`01-from-kspace-to-image`; the cell that does it is hidden on this page and
present in the script this page can be downloaded as.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
PHASE = Colormap("colorcet:CET_C6").to_matplotlib()

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
import tempfile
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch.io import readcfl, writecfl

SIZE = 128
COILS = 8

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
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# %%
#
# Noise prewhitening
# ------------------
#
# The noise of a receive array is correlated between channels: neighbouring
# coils share their thermal noise and their preamplifier coupling. Least
# squares is the maximum-likelihood estimator only for noise that is white, so
# the data is transformed by the inverse square root of the noise covariance
# first, and a noise scan -- an acquisition with no excitation -- is measured
# to estimate that covariance.
#
# The simulation below correlates the channels' noise by a factor falling with
# the distance between them, which is the structure a ring of coils produces,
# and measures a noise-only acquisition of the same channels.

generator = torch.Generator().manual_seed(2)
channels = torch.arange(COILS)
mixing = (0.6 ** (channels[:, None] - channels[None, :]).abs().float()).to(torch.complex64)


def channel_noise(scale=0.01):
    """Complex Gaussian noise, correlated across the channels by ``mixing``."""
    white = torch.randn(COILS, SIZE, SIZE, generator=generator) + 1j * torch.randn(
        COILS, SIZE, SIZE, generator=generator
    )
    return scale * torch.einsum("ij,jyx->iyx", mixing, white.to(torch.complex64))


noise_scan = bartorch.fft(channel_noise(), axes=(-2, -1), unitary=True)
kspace = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True) + noise_scan

whitened = bt.whiten(kspace[:, None], noise_scan[:, None])

# %%
#
# What the transform does is visible in the channel covariance, which is the
# identity for white noise. The measure below is the mean off-diagonal
# magnitude as a fraction of the mean diagonal: zero when the channels are
# uncorrelated.


def correlation(channels):
    """Mean off-diagonal magnitude of the channel covariance, over its diagonal."""
    flat = channels.reshape(COILS, -1)
    covariance = (flat @ flat.conj().T) / flat.shape[1]
    diagonal = torch.diagonal(covariance).abs()
    off = (covariance.abs() - torch.diag(diagonal)).abs().sum() / (COILS * (COILS - 1))
    return float(off / diagonal.mean()), covariance


before, covariance_before = correlation(noise_scan)
after, covariance_after = correlation(bt.whiten(noise_scan[:, None], noise_scan[:, None])[:, 0])

print(f"channel correlation: {before:.3f} measured, {after:.3f} after whitening")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 2)
for axis, values, title in (
    (axes[0, 0], covariance_before, "measured"),
    (axes[0, 1], covariance_after, "after whitening"),
):
    handle = axis.imshow(
        (values.abs() / values.abs().max()).cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0
    )
    axis.set_title(title)
    axis.set_xlabel("channel")
axes[0, 0].set_ylabel("channel")
figure.colorbar(handle, ax=axes[0, 1], fraction=0.046)
figure.suptitle("noise covariance, normalized magnitude")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Aligning repetitions
# --------------------
#
# Averaging repetitions that moved between them blurs the average.
# :func:`bartorch.estimate_shift` measures a translation from the phase of the
# cross-spectrum of two arrays, to a fraction of a voxel, and returns it in
# voxels along the axes it was given.

moved = bartorch.circshift(image, (3, -5), (-2, -1))
shift = bartorch.estimate_shift(image, moved, axes=(-2, -1))

print(f"shift estimated as {shift.tolist()} voxels")

# %%
#
# A whole-voxel shift is undone by :func:`bartorch.circshift`, and a
# sub-voxel one by resampling -- :func:`bartorch.interpolate` on a shifted
# grid, or :func:`bartorch.fovshift`, which applies the corresponding linear
# phase in k-space instead. Rotation and scaling need
# :func:`bartorch.register_affine`, which fits an affine transform by mutual
# information, and :func:`bartorch.affine_transform` or
# :func:`bartorch.warp` to apply what it returns.

realigned = bartorch.circshift(moved, (-3, 5), (-2, -1))
print(f"residual after realignment: {float((realigned - image).abs().max()):.1e}")

# %%
#
# Field of view
# -------------
#
# :func:`bartorch.resize` crops or zero-pads around the centre, which in
# k-space is a change of resolution and in image space a change of field of
# view. Zero-padding k-space to twice its size interpolates the image onto
# twice the grid without adding information to it, as a scanner's
# interpolation filter does.

interpolated = bartorch.ifft(
    bartorch.resize(bartorch.fft(image, axes=(-2, -1), unitary=True), (2 * SIZE, 2 * SIZE)),
    axes=(-2, -1),
    unitary=True,
)

print(f"{tuple(image.shape)} -> {tuple(interpolated.shape)}")

# %%
#
# Measuring a result
# ------------------
#
# :func:`bartorch.nrmse`, :func:`bartorch.psnr` and :func:`bartorch.ssim`
# compare an estimate against a reference. The comparison is between
# magnitudes, since a SENSE reconstruction leaves the phase of the
# sensitivities in the image.
#
# Only ``nrmse(..., scaled=True)`` divides out the global scale a
# reconstruction does not determine; the peak signal-to-noise ratio and the
# structural similarity are both defined against an absolute scale, so an
# estimate has to be brought onto the reference's before either means anything.
# The reconstruction below is scaled by whatever ``pics`` estimated from the
# data, which is enough to make the difference.

maps = bt.ecalib(kspace[:, None], maps=1, crop=0.8)
estimate = bt.pics(kspace[:, None], maps, l2=0.01, maxiter=20).abs()
common = estimate * float((image.abs() * estimate).sum() / (estimate**2).sum())

print(f"NRMSE {bartorch.nrmse(image.abs(), estimate, scaled=True):.3f}")
print(f"PSNR  {bartorch.psnr(image.abs(), common):.1f} dB")
print(f"SSIM  {bartorch.ssim(image.abs(), common):.3f}")

# %%
#
# :func:`bartorch.roi_stat` reports a statistic over a region rather than over
# the whole image, which is how a phantom measurement is reported.

region = (memberships[CLASS["WM"]] > 0.8).to(torch.complex64)

for name, values in (("phantom", image.abs()), ("reconstruction", common)):
    volume = values.to(torch.complex64)
    mean = float(bartorch.roi_stat(region, volume, "mean").real)
    deviation = float(bartorch.roi_stat(region, volume, "std").real)
    print(f"{name:>14}, white matter: {mean:.3f} +/- {deviation:.3f}")

# %%
#
# CFL files
# ---------
#
# BART's file format is a pair: a `.cfl` of complex floats and a `.hdr` naming
# the dimensions. It is written in BART's axis order, the reverse of a
# tensor's, so the axes are reversed at that boundary and nowhere else --
# ``.T`` on a NumPy array reverses all of them.

with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "image")
    writecfl(path, image.numpy().T)
    restored = torch.from_numpy(np.ascontiguousarray(readcfl(path).T))

difference = float((restored - image).abs().max())
print(f"round trip: {tuple(restored.shape)}, largest difference {difference:.1e}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 4)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
show(axes[0, 1], moved, "translated", vmax=peak)
show(axes[0, 2], realigned, "realigned", vmax=peak)
show(axes[0, 3], region.real, "white matter region", vmax=1.0)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The region statistic separates the two things an image-quality number mixes
# together: the reconstruction recovers the mean of the region, and what it
# adds is spread, which the standard deviation over the region reports and a
# single error number does not.

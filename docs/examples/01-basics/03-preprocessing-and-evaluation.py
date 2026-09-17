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
SLICE = 90

# sphinx_gallery_start_ignore
table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
proton_density = np.array([float(row["PD (ms)"]) for row in rows], dtype=np.float32)
labels = {row["Tissue"]: index for index, row in enumerate(rows)}

fractions = get_mri(sub_id=0, contrast="fuzzy")[:, :, SLICE]
fractions = np.flip(fractions.transpose(1, 0, 2), 0).copy()


def resampled(values):
    grid = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


magnitude = resampled(fractions @ proton_density)
image = (magnitude / magnitude.max()).to(torch.complex64)

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
# first, which is what a noise scan -- an acquisition with no excitation -- is
# measured for.
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
# twice the grid without adding information to it, which is what a scanner's
# reconstruction filter does.

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

region = (resampled(fractions[..., labels["WM"]]) > 0.8).to(torch.complex64)

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

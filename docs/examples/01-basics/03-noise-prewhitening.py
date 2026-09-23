"""
==================
Noise prewhitening
==================

The effect of channel-noise correlation on a SENSE reconstruction, and its
removal by prewhitening with a noise measurement.

The thermal noise of a receive array is correlated between channels and
differs in level from one channel to another.  Least squares is the
maximum-likelihood estimator only for white noise, so the data are first
transformed by a whitening matrix :math:`W` with :math:`W \\Psi W^H = I`,
:math:`\\Psi` being the channel noise covariance estimated from a
noise-only acquisition [#roemer]_ [#pruessmann]_.  ESPIRiT then calibrates
the sensitivities of the whitened channels, and the reconstruction proceeds
unchanged.

The signal-to-noise ratio of the two reconstructions is measured by the
pseudo-replica method [#robson]_: the same reconstruction is repeated on
independent noise realizations added to one noise-free acquisition, and the
standard deviation across repetitions is the noise of each voxel.

The phantom and the coil sensitivities are built as in
:doc:`01-from-kspace-to-image`; the cell that does it is hidden on this page
and present in the script this page can be downloaded as.
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
# Correlated channel noise
# ------------------------
#
# The noise covariance used for the simulation couples neighbouring channels
# by a factor falling as :math:`0.5^{|i-j|}` and gives the channels standard
# deviations between 0.8 and 1.25 of a common level.  A noise scan -- an
# acquisition with the RF transmitter off -- measures the same channels
# without signal; its covariance is the estimate of :math:`\Psi` that
# :func:`bartorch.tools.whiten` inverts.

SIGMA = 0.01  # noise level, relative to the image's peak

channels = torch.arange(COILS)
levels = torch.linspace(0.8, 1.25, COILS)
correlation = 0.5 ** (channels[:, None] - channels[None, :]).abs().float()
covariance = levels[:, None] * correlation * levels[None, :]
mixing = torch.linalg.cholesky(covariance).to(torch.complex64)
generator = torch.Generator().manual_seed(2)


def channel_noise(shape):
    """Complex Gaussian noise with the channel covariance above."""
    white = torch.randn(COILS, *shape, generator=generator)
    white = (white + 1j * torch.randn(COILS, *shape, generator=generator)) / 2**0.5
    return SIGMA * torch.einsum("ij,j...->i...", mixing, white.to(torch.complex64))


noise_scan = channel_noise((SIZE, SIZE))[:, None]

# %%
#
# The whitening matrix maps the measured covariance to the identity.  The
# measure below is the mean magnitude of the off-diagonal covariance entries,
# relative to the mean diagonal entry: zero for uncorrelated channels.


def channel_covariance(samples):
    """Sample covariance of the channels, over every other axis."""
    flat = samples.reshape(COILS, -1)
    return (flat @ flat.conj().T) / flat.shape[1]


def off_diagonal(matrix):
    diagonal = torch.diagonal(matrix).abs()
    return float((matrix.abs().sum() - diagonal.sum()) / (COILS * (COILS - 1)) / diagonal.mean())


measured = channel_covariance(noise_scan)
whitened = channel_covariance(bt.whiten(noise_scan, noise_scan))

print(f"off-diagonal covariance: {off_diagonal(measured):.3f} measured")
print(f"                         {off_diagonal(whitened):.3f} after whitening")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH * 0.72, PAGE_WIDTH * 0.36))
for axis, values, title in (
    (axes[0], measured, "measured"),
    (axes[1], whitened, "after whitening"),
):
    handle = axis.imshow(
        (values.abs() / values.abs().max()).numpy(), cmap="magma", vmin=0.0, vmax=1.0
    )
    axis.set_title(title)
    axis.set_xlabel("channel")
    axis.set_xticks(range(0, COILS, 2))
    axis.set_yticks(range(0, COILS, 2))
axes[0].set_ylabel("channel")
figure.colorbar(handle, ax=axes, fraction=0.046, label="|covariance| / max")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Acquisition and calibration
# ---------------------------
#
# Every second phase encode is acquired, with 24 central lines kept for
# calibration.  Each pseudo-replica adds a new noise realization to the same
# noise-free k-space.  The sensitivities are calibrated once per pipeline,
# from the first replica: from the channels as measured, and from the
# whitened channels.

CALIBRATION = 24
REPLICAS = 32

lines = torch.zeros(SIZE)
lines[::2] = 1.0
lines[SIZE // 2 - CALIBRATION // 2 : SIZE // 2 + CALIBRATION // 2] = 1.0
pattern = lines.reshape(SIZE, 1).to(torch.complex64)

noiseless = bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True)


def acquire():
    """One replica: the noise-free k-space plus new noise, sampled."""
    return ((noiseless + channel_noise((SIZE, SIZE))) * pattern)[:, None]


first = acquire()
maps_measured = bt.ecalib(first, maps=1, calib_size=CALIBRATION, crop=0.8)
maps_whitened = bt.ecalib(bt.whiten(first, noise_scan), maps=1, calib_size=CALIBRATION, crop=0.8)

# %%
#
# Pseudo-replicas
# ---------------
#
# Both pipelines use the same reconstruction, conjugate-gradient SENSE with a
# small Tikhonov weight and a fixed number of iterations; they differ only in
# whether the data are whitened first.

plain, prewhitened = [], []
for _ in range(REPLICAS):
    data = acquire()
    plain.append(bt.pics(data, maps_measured, l2=1e-3, maxiter=30))
    prewhitened.append(bt.pics(bt.whiten(data, noise_scan), maps_whitened, l2=1e-3, maxiter=30))

plain, prewhitened = torch.stack(plain), torch.stack(prewhitened)

# %%
#
# The signal-to-noise ratio of a voxel is the magnitude of the mean
# reconstruction over its standard deviation across replicas.  Both are
# reported over the white matter, where the phantom is homogeneous.

white_matter = memberships[CLASS["WM"]] > 0.8


def snr_map(replicas):
    return replicas.mean(0).abs() / replicas.std(0)


for name, replicas in (("as measured", plain), ("prewhitened", prewhitened)):
    values = snr_map(replicas)[white_matter]
    mean, median = float(values.mean()), float(values.median())
    print(f"{name:>12}  white-matter SNR {mean:6.1f} (median {median:.1f})")

gain = snr_map(prewhitened) / snr_map(plain)
print(f"SNR ratio, prewhitened over as measured: {float(gain[white_matter].median()):.2f} (median)")

# %%

# sphinx_gallery_start_ignore
support = (image.abs() > 0.05 * float(image.abs().max())).numpy()
figure, axes = panels(1, 3)
shared = max(
    float(snr_map(plain)[white_matter].max()), float(snr_map(prewhitened)[white_matter].max())
)
for axis, values, title in (
    (axes[0, 0], snr_map(plain), "SNR, as measured"),
    (axes[0, 1], snr_map(prewhitened), "SNR, prewhitened"),
):
    handle = axis.imshow(
        np.where(support, values.numpy(), np.nan), cmap="viridis", vmin=0.0, vmax=shared
    )
    axis.set_title(title)
figure.colorbar(handle, ax=axes[0, :2], fraction=0.046, label="SNR")
ratio = axes[0, 2].imshow(
    np.where(support, gain.numpy(), np.nan), cmap="RdBu_r", vmin=0.5, vmax=1.5
)
axes[0, 2].set_title("SNR ratio")
figure.colorbar(ratio, ax=axes[0, 2], fraction=0.046, label="prewhitened / as measured")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# With this covariance, prewhitening raises the white-matter SNR by the
# ratio printed above.  The size of the gain depends on the array's noise
# correlation and on the spread of the channel noise levels; it is measured
# here for one simulated covariance.
#
# References
# ----------
#
# .. [#roemer] Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM. The NMR
#    phased array. *Magn Reson Med* 16(2):192-225 (1990).
#    https://doi.org/10.1002/mrm.1910160203
#
# .. [#pruessmann] Pruessmann KP, Weiger M, Börnert P, Boesiger P. Advances in
#    sensitivity encoding with arbitrary k-space trajectories. *Magn Reson Med*
#    46(4):638-651 (2001). https://doi.org/10.1002/mrm.1241
#
# .. [#robson] Robson PM, Grant AK, Madhuranthakam AJ, Lattanzi R, Sodickson DK,
#    McKenzie CA. Comprehensive quantification of signal-to-noise ratio and
#    g-factor for image-based and k-space-based parallel imaging
#    reconstructions. *Magn Reson Med* 60(4):895-907 (2008).
#    https://doi.org/10.1002/mrm.21728

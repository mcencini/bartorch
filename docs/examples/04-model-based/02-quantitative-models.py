"""
=====================================
Parameter maps straight from k-space
=====================================

A multi-echo spin-echo acquisition fitted for :math:`T_2` in two ways: by
reconstructing the echo images and fitting them afterwards, and by putting the
signal model inside the forward operator and fitting the k-space directly.

The two-step route solves an ill-posed reconstruction eight times over, once
per echo, and then fits a model to the answers. Each reconstruction is
undersampled on its own and nothing in it knows that the eight images are
related. The model-based route puts that relation in the forward operator,

.. math::

   y_{c,e} = P_e F \\, (S_c \\cdot M_e(\\theta)),

where :math:`M` is the signal model and :math:`\\theta` the parameter maps, and
solves for :math:`\\theta` directly. The unknowns then number three maps rather
than eight images, and every echo constrains all of them.

The model here is :class:`bartorch.nlop.MultiEcho`, a TorchSim simulator as a
BART nonlinear operator; the solver is the Gauss-Newton loop of
:doc:`01-nonlinear-inversion`, over a different model.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/01-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.

Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. *Physics-based reconstruction
methods for magnetic resonance imaging.* Phil Trans R Soc A 379:20200196
(2021).
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
import time
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import linop, nlop, optim

SIZE = 96
COILS = 8
ECHOES = 8
ACCELERATION = 4

ECHO_TIMES = torch.tensor([12.5 * (echo + 1) for echo in range(ECHOES)])  # ms

# %%
#
# Phantom
# -------
#
# The :math:`T_2` of each tissue class from the BrainWeb table, combined by
# membership, and the echo images from the mono-exponential decay
# :math:`M_0 \exp(-\mathrm{TE}/T_2)` written out here rather than taken from
# the model that will be fitted.

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
support = memberships.sum(0) > 0.5
amplitude = torch.where(support, proton_density, torch.zeros(()))
t2 = torch.where(support, T2.clamp(20.0, 400.0), torch.tensor(50.0))
# sphinx_gallery_end_ignore

contrasts = (amplitude[None] * torch.exp(-ECHO_TIMES[:, None, None] / t2[None])).to(torch.complex64)

# %%
#
# Acquisition
# -----------
#
# Each echo is sampled at a quarter of the phase encodes, with its own draw, so
# no two echoes are missing the same part of k-space. The echoes are a batch of
# the encoding rather than an axis inside it: the sensitivities are shared, the
# transform is the same, and only the pattern differs, so the operator is the
# Cartesian SENSE encoding of :doc:`../01-basics/02-operators-and-solvers` with
# the per-echo pattern applied to its samples.

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
generator = torch.Generator().manual_seed(4)
lines = torch.zeros(ECHOES, 1, SIZE, 1)
for echo in range(ECHOES):
    lines[echo, 0, torch.randperm(SIZE, generator=generator)[: SIZE // ACCELERATION]] = 1.0
    lines[echo, 0, SIZE // 2 - 4 : SIZE // 2 + 4] = 1.0
# sphinx_gallery_end_ignore

encoding = linop.CartesianSense(sensitivities, (ECHOES, SIZE, SIZE), ndim=2)
E = linop.Diagonal(lines.to(torch.complex64), encoding.oshape) @ encoding

measured = bt.noise(E(contrasts), n=1e-6, s=9)
data = measured / float(E.H(measured).abs().max())

print(f"{E.ishape} -> {E.oshape}")

# %%
#
# The signal model
# ----------------
#
# :class:`bartorch.nlop.MultiEcho` maps parameter maps to one image per echo.
# Its unknowns are :math:`T_2` and a complex amplitude, carried as three real
# maps in a bounded parameterisation rather than in their own units, so
# :meth:`~bartorch.nlop.SignalModel.initial` builds a starting point from
# values and :meth:`~bartorch.nlop.SignalModel.split` reads the fit back.

M = nlop.MultiEcho([float(te) for te in ECHO_TIMES], (SIZE, SIZE))
start = M.initial(T2=80.0)

print(f"unknowns {M.names}: {M.ishapes[0]} -> {M.oshapes[0]}")

# %%
#
# Two routes
# ----------
#
# The first reconstructs the echo images by conjugate gradients and fits the
# model to them. The second composes the model with the encoding and fits the
# k-space. Both are the same Gauss-Newton loop with the same number of steps,
# and they differ only in what stands between the unknowns and the data.

STEPS = 20

start_time = time.perf_counter()
images = optim.CG(maxiter=40)(data, E)
two_step = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)(images, M, x0=start)
print(f"reconstruct, then fit:  {time.perf_counter() - start_time:5.1f} s")

start_time = time.perf_counter()
model_based = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)(data, E @ M, x0=start)
print(f"model inside the operator: {time.perf_counter() - start_time:5.1f} s")

# %%
#
# ``E @ M`` composes a linear operator with a nonlinear one; the derivative of
# the composition at a point is the encoding applied to the derivative of the
# model -- exactly what a Gauss-Newton step asks of the composition.

estimates = {
    name: M.split(fit)["T2"]
    for name, fit in (("reconstruct, then fit", two_step), ("model-based", model_based))
}

for name, estimate in estimates.items():
    error = float((estimate[support] - t2[support]).norm() / t2[support].norm())
    median = float(estimate[support].median())
    print(f"{name:>22}  median {median:5.1f} ms   relative error {error:.3f}")

print(f"{'phantom':>22}  median {float(t2[support].median()):5.1f} ms")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
for axis, values, title in (
    (axes[0, 0], t2, "phantom"),
    (axes[0, 1], estimates["reconstruct, then fit"], "reconstruct, then fit"),
    (axes[0, 2], estimates["model-based"], "model-based"),
):
    parameter(axis, torch.where(support, values, torch.zeros(())).detach(), "T2")
    axis.set_title(title, fontsize=10)
scalebar(figure, axes[0, 2], name="T2")

figure, axes = panels(1, 4)
for column, echo in enumerate((0, 2, 4, 7)):
    axes[0, column].imshow(images[echo].abs().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, column].set_title(f"TE = {float(ECHO_TIMES[echo]):.0f} ms")
figure.suptitle("echo images from the two-step route")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The echo images carry the aliasing each echo's own sampling leaves, and the
# fit that follows has no way to tell that apart from decay: the two-step
# :math:`T_2` map is the noisier of the two and biased upward on this data.
# Fitting the k-space constrains the three maps with all eight echoes at once,
# which is the whole of the difference -- the model, the solver and the number
# of steps are the same.
#
# What this route also makes available is regularization of the maps rather
# than of the images, since the maps are what the solver holds; BART's own
# ``moba`` is :func:`bartorch.tools.moba`, and applies its penalties there.

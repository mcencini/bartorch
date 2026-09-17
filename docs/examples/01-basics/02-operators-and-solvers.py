"""
=====================
Operators and solvers
=====================

The same reconstruction written as an encoding operator and a solver rather
than as a call to a BART application.

:func:`bartorch.tools.pics` assembles three things and hands them to BART's
iteration: the encoding operator, the regularization terms, and the algorithm.
:mod:`bartorch.linop` and :mod:`bartorch.optim` expose those three separately,
for the reconstructions BART has no application for: an encoding with an extra
factor in it, a solver reached from an outer loop, an operator defined in
Python.

This example builds the encoding of :doc:`01-from-kspace-to-image`, checks it
against the definition of an adjoint, solves with it, and compares the result
with the application. The phantom, the coil sensitivities and the sampling are
that example's; the cell that builds them is hidden on this page and present in
the script this page can be downloaded as.
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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 192
COILS = 8
ACCELERATION = 3
CALIBRATION = 24

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

# sphinx_gallery_start_ignore
# The acquisition of :doc:`/auto_examples/01-basics/01-from-kspace-to-image`:
# the k-space of the coil images, and a variable-density random set of phase
# encodes with a fully sampled centre.
kspace = bt.noise(bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True), n=1e-5, s=42)

encodes = torch.arange(SIZE) - SIZE // 2
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0 * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
kspace = kspace[:, None] * lines.reshape(SIZE, 1).to(torch.complex64)
maps = bt.ecalib(kspace, maps=1, calib_size=CALIBRATION, crop=0.8)
# sphinx_gallery_end_ignore

# %%
#
# The encoding operator
# ---------------------
#
# :func:`bartorch.linop.CartesianSense` is :math:`A = P F S` as one operator.
# It takes the sensitivities, the shape of the image it maps from, and the
# sampling pattern; the shape of the k-space it maps to follows from those.
# :func:`bartorch.tools.pattern` reads the pattern off the measured data, which
# is where a prospectively undersampled acquisition gets it from.
#
# ``modulated=True`` selects BART's uncentred sample convention, which its
# applications iterate in; the default is the centred convention that
# :func:`bartorch.fft` produces. The two differ by a modulation of the samples
# and give the same image, so the choice matters only when the operator has to
# meet data that is already in one of them, as it does below.

pattern = bt.pattern(kspace)
A = linop.CartesianSense(maps.squeeze(1), (SIZE, SIZE), pattern.squeeze(), modulated=True)

print(f"{A.ishape} -> {A.oshape}")
print(A.plan)
print(f"fused: {A.plan.fused}")

# %%
#
# ``A.plan`` reports the form the operator was lowered into: which transform,
# what multiplies the image and the samples, and how the normal operator
# :math:`A^H A` is applied. It is a property of the built operator rather than
# a prediction, and ``plan.fused`` is false where the composition could not be
# expressed as one encoding and fell back to a chain, which computes the same
# numbers more slowly.
#
# Applying the adjoint is not the same as applying the transpose, and a
# reconstruction built on the wrong one converges to the wrong image. The
# definition :math:`\langle Ax, y\rangle = \langle x, A^H y\rangle` holds for
# any pair of vectors, and holds for random vectors as readily as for real
# data, so it is a usable check on an operator.

generator = torch.Generator().manual_seed(0)
probe = torch.randn(A.ishape, dtype=torch.complex64, generator=generator)
samples = torch.randn(A.oshape, dtype=torch.complex64, generator=generator)

forward = (A(probe).conj() * samples).sum()
adjoint = (probe.conj() * A.H(samples)).sum()
print(f"relative difference {abs(forward - adjoint) / abs(forward):.2e}")

# %%
#
# Solving
# -------
#
# A solver is called as ``solver(y, A)``. What it is given is not the array
# the scanner wrote but what ``pics`` iterates on: the sampling pattern
# applied, the modulation into the uncentred convention, and the data divided
# by the scaling :func:`bartorch.optim.data_scaling` estimates from the adjoint
# reconstruction, which is the step that makes a regularization weight
# transferable from one dataset to the next.

measured = bartorch.fftmod(kspace * pattern, axes=(-1, -2, -3), inverse=True)
scale = optim.data_scaling(measured)
data = (measured / scale).squeeze(1)

term = priors.Wavelet(axes=(-1, -2), weight=0.002)
assembled = optim.FISTA(term, maxiter=100)(data, A)

# %%
#
# With the same preprocessing the assembled solve and the application are not
# merely close: they are the same iteration over the same operator, and return
# the same bits.

tool = bt.pics(kspace, maps, regularizers=term, solver="fista", maxiter=100)
print(f"identical to pics: {torch.equal(assembled.squeeze(), tool.squeeze())}")

# %%
#
# Operator algebra
# ----------------
#
# ``@`` composes, ``+`` adds, ``A.H`` is the adjoint and ``A.gram()`` the
# normal operator :math:`A^H A`. A composition builds a single BART operator
# rather than a Python chain, so a solver iterating on it does not return to
# Python between applications. :func:`bartorch.optim.maxeigen` runs the power
# iteration on an operator, which is how a gradient step size is chosen: the
# Lipschitz constant of the least-squares gradient is the largest eigenvalue of
# :math:`A^H A`.

print(f"largest eigenvalue of A^H A: {optim.maxeigen(A.gram()):.3f}")

# %%
#
# An operator defined in Python joins the algebra through
# :meth:`~bartorch.linop.LinearOperator.from_callbacks`, which BART applies as
# a callback. Here it is a spatially varying phase, as an off-resonance or an
# eddy-current phase would be, placed between the image and the encoding.

field = torch.exp(1j * 0.4 * torch.pi * grid_x).to(torch.complex64)
phase = linop.LinearOperator.from_callbacks(
    (SIZE, SIZE), (SIZE, SIZE), lambda u: field * u, lambda u: field.conj() * u
)
composed = A @ phase
print(f"{composed.ishape} -> {composed.oshape}, fused: {composed.plan.fused}")

# %%
#
# Differentiation
# ---------------
#
# Applying an operator to a tensor that requires a gradient records the
# application for autograd. The gradient torch propagates back through
# :math:`y = Ax` is :math:`A^H g` rather than :math:`A^T g`, which is the
# convention torch uses for complex tensors and the reason a real-valued check
# would not distinguish the two.

variable = data.new_zeros(A.ishape).requires_grad_(True)
residual = A(variable) - data
(residual.abs() ** 2).sum().backward()

expected = 2 * A.H(-data)
difference = float((variable.grad - expected).abs().max() / expected.abs().max())
print(f"relative difference from 2 A^H (Ax - y): {difference:.2e}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
show(axes[0, 1], scaled(A.H(data), image), "adjoint reconstruction", vmax=peak)
show(axes[0, 2], scaled(assembled, image), "FISTA, wavelet penalty", vmax=peak)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The adjoint of the encoding is not its inverse: :math:`A^H y` is the coil
# combination of the zero-filled k-space, and carries the aliasing the sampling
# operator left. What the solver adds is the inversion.
#
# The regularization terms are the subject of :mod:`bartorch.priors`, and the
# iterations of :mod:`bartorch.optim`;
# :doc:`../../explanation/inverse-problems` states which algorithm applies to
# which problem.

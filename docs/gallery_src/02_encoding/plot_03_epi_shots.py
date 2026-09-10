"""
An idealized multi-shot EPI encoding
====================================

Represent known shot phases and interleaved phase-encode masks with existing
BART operators. Requires the base CPU installation. Readout polarity and ghost
correction are assumed already handled; off-resonance is omitted. This is a
synthetic encoding exercise, not an implementation of a published EPI method.
See :doc:`/gallery/research` for the literature and measured-data requirements.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt
from bartorch import linop

bartorch.set_num_threads(1)
torch.manual_seed(3)
n, nshots, ncoils = 24, 2, 4
grid = torch.linspace(-1, 1, n)
yy, xx = torch.meshgrid(grid, grid, indexing="ij")
maps = torch.stack([torch.exp(1j * j * yy) for j in range(ncoils)])[None] / ncoils**0.5
phases = torch.stack([torch.ones_like(xx), torch.exp(1j * (1.2 * xx + yy))])[:, None]
shape = (1, 1, n, n)
data_shape = (nshots, ncoils, n, n)
SD = linop.MultiplySum(maps * phases, shape, data_shape)
F = linop.FFT(data_shape, axes=(-2, -1))
mask = torch.zeros(nshots, 1, n, n, dtype=torch.complex64)
for shot in range(nshots):
    mask[shot, :, shot::nshots, :] = 1
A = linop.Sampling(mask, data_shape) @ F @ SD
truth = bt.phantom([n, n]).reshape(shape)
data = (
    torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(maps * phases * truth, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    * mask
)
torch.testing.assert_close(A(truth), data, atol=3e-6, rtol=3e-5)
probe = torch.randn(*data_shape, dtype=torch.complex64)
torch.testing.assert_close(
    torch.vdot(data.flatten(), probe.flatten()),
    torch.vdot(truth.flatten(), A.adjoint(probe).flatten()),
    atol=3e-5,
    rtol=3e-5,
)
estimate = A.lstsq(data, lambda_=1e-4, maxiter=80, tol=1e-7)
residual = (A(estimate) - data).norm() / data.norm()
assert residual < 0.01
print(f"Relative data residual: {residual.item():.2e}")

# %%
# Unknown shot phase is a different inverse problem. Estimating it or imposing
# local/structured low rank across shot images requires additional modeling.
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, value, title in zip(
    axes,
    [truth.abs().squeeze(), phases[1, 0].angle(), estimate.abs().squeeze()],
    ["Truth magnitude", "Known shot 2 phase (rad)", "Joint reconstruction"],
):
    im = ax.imshow(value.numpy(), cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.show()

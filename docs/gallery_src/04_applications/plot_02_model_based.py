"""
Model-based multi-echo reconstruction
=====================================

Fit amplitude and decay-rate maps directly from Fourier data using a PyTorch
signal model and BART's iteratively regularized Gauss-Newton solver. Requires
the base CPU installation. Rates are in inverse seconds; echo times are seconds.
This small example omits coil encoding and physiological model complications.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt
from bartorch.ops import LinearOperator, NonlinearOperator

bartorch.set_num_threads(1)
n, nechoes = 8, 6
times = torch.linspace(0, 1.0, nechoes)
amplitude = bt.phantom([n, n]).real.reshape(n, n)
truth = torch.stack([amplitude, 0.5 + 0.5 * amplitude]).to(torch.complex64)


def signal(parameters):
    return parameters[0][None] * torch.exp(-parameters[1][None] * times[:, None, None])


M = NonlinearOperator.from_torch(signal, (2, n, n), (nechoes, n, n))
F = LinearOperator.fft((nechoes, n, n), axes=(-2, -1))
A = F @ M
# Simulate using the analytic signal and an independent Fourier implementation.
data = torch.fft.fftshift(torch.fft.fft2(
    torch.fft.ifftshift(signal(truth), dim=(-2, -1)), norm="ortho"
), dim=(-2, -1))
x0 = torch.stack([torch.ones(n, n), 0.5 * torch.ones(n, n)]).to(torch.complex64)
estimate = A.irgnm(data, x0, iterations=12, alpha=1, alpha_min=1e-6, redu=3, cgiter=60)
foreground = amplitude > 0.1
assert (estimate[0][foreground] - truth[0][foreground]).abs().max() < 0.05
print("Relative signal residual:", ((signal(estimate) - signal(truth)).norm() / signal(truth).norm()).item())

# %%
# A decay rate is not identifiable where amplitude is zero. Mask background
# before assessing parameter error. No gradient through the full BART solve is
# provided here: autograd supplies the local model derivatives to Gauss-Newton.
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, value, title in zip(
    axes,
    [estimate[0].real, truth[1].real * foreground, estimate[1].real * foreground],
    ["Estimated amplitude", "True rate (1/s)", "Estimated rate (1/s)"],
):
    im = ax.imshow(value.numpy(), cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.show()

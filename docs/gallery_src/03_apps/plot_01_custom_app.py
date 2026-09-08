"""
Build a reconstruction app with a Python callback
================================================

A fixed phase field is a simple custom encoding component. Wrap it as a BART
linear operator, compose it with BART's FFT, and solve a regularized inverse
problem. Requires the base CPU installation.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt
from bartorch.ops import LinearOperator

bartorch.set_num_threads(1)


def make_encoding(shape, phase):
    """Construct a fixed phase modulation followed by Fourier encoding."""
    modulation = torch.exp(1j * phase).to(torch.complex64)
    W = LinearOperator.from_callbacks(
        shape, shape,
        forward=lambda image: modulation * image,
        adjoint=lambda data: modulation.conj() * data,
    )
    return LinearOperator.fft(shape, axes=(-2, -1)) @ W


n = 32
truth = bt.phantom([n, n])
grid = torch.linspace(-1, 1, n)
phase = 0.8 * grid[:, None] + 0.3 * grid[None, :].square()
A = make_encoding(tuple(truth.shape), phase)
# Simulate data independently in PyTorch.
data = torch.fft.fftshift(torch.fft.fft2(
    torch.fft.ifftshift(torch.exp(1j * phase) * truth), norm="ortho"
))
lambda_ = 0.01
reconstruction = A.lstsq(data, lambda_=lambda_, maxiter=30, tol=1e-7)
# This fully sampled unitary model has a closed-form ridge solution.
torch.testing.assert_close(reconstruction, truth / (1 + lambda_), atol=2e-5, rtol=2e-4)
print("Relative data residual:", ((A(reconstruction) - data).norm() / data.norm()).item())

# %%
# Keep callbacks side-effect free: inputs are views of BART-owned working
# buffers. For multiple datasets, make shape/device/phase choices explicit in
# the application and retain the composed operator for repeated solves.
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, value, title in zip(
    axes, [truth.abs(), phase, reconstruction.abs()],
    ["Truth magnitude", "Known phase (rad)", "Reconstruction magnitude"],
):
    im = ax.imshow(value.numpy(), cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.show()

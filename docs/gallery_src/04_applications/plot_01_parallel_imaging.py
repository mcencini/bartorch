"""
Parallel imaging with ESPIRiT and PICS
======================================

Calibrate from a fully sampled central region, retrospectively undersample
phase-encode lines, and reconstruct with a quadratic penalty. Requires the base
CPU installation. This teaching phantom does not estimate an in-vivo g-factor.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt

bartorch.set_num_threads(1)
n = 32
full = bt.phantom([n, n], kspace=True, ncoils=4)
mask = torch.zeros_like(full)
mask[..., ::2, :] = 1
mask[..., n // 2 - 8 : n // 2 + 8, :] = 1
sampled = full * mask
maps = bt.ecalib(sampled, calib_size=16, maps=1)
reconstruction = bt.pics(sampled, maps, l=2, lambda_=0.001, iter_=60)
print(f"Actual acceleration including calibration: {mask.numel() / mask.real.sum().item():.2f}")

# %%
# A fully sampled coil-combined reference computed using PyTorch's FFT avoids
# comparing two PICS invocations. Compare magnitudes after a global scale fit:
# PICS applies data scaling, and sensitivity maps have a phase gauge.
coil_images = torch.fft.fftshift(
    torch.fft.ifft2(torch.fft.ifftshift(full, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
).reshape(4, n, n)
reference = (coil_images * maps.reshape(4, n, n).conj()).sum(0).abs()
result = reconstruction.abs().squeeze()
scale = (result * reference).sum() / result.square().sum()
result = result * scale
relative_error = (result - reference).norm() / reference.norm()
assert relative_error < 0.25
print(f"Scale-aligned magnitude error: {relative_error.item():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, value, title in zip(
    axes,
    [reference, result, (result - reference).abs()],
    ["Fully sampled reference", "Undersampled PICS", "Absolute magnitude error"],
):
    im = ax.imshow(value.numpy(), cmap="gray", vmin=0, vmax=reference.max().item())
    ax.set_title(title)
    ax.axis("off")
plt.show()

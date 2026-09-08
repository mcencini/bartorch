"""
Phantom, Fourier transform, and calibration
==========================================

Use prepackaged tools on tensors, inspect their shapes, and verify a unitary
Fourier round trip. Requires bartorch, PyTorch, and Matplotlib on CPU.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt

bartorch.set_num_threads(1)
image = bt.phantom([32, 32])
kspace = bt.fft(image, axes=(-2, -1), unitary=True)
back = bt.ifft(kspace, axes=(-2, -1), unitary=True)
torch.testing.assert_close(back, image, atol=2e-6, rtol=2e-5)
print("image:", image.shape, image.dtype, image.device)

# %%
# Keep the coil and slice axes for BART tools. Calibration estimates spatial
# sensitivity maps from fully sampled central k-space; it does not create a
# universal set of maps independent of the acquisition.
coil_kspace = bt.phantom([32, 32], kspace=True, ncoils=4)
maps = bt.ecalib(coil_kspace, calib_size=16, maps=1)
print("coil k-space:", coil_kspace.shape, "maps:", maps.shape)

fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
for ax, data, title in zip(
    axes,
    [image.abs(), torch.log1p(kspace.abs()), maps.reshape(4, 32, 32)[0].abs()],
    ["Phantom magnitude", "log(1 + |k-space|)", "Coil 1 sensitivity magnitude"],
):
    ax.imshow(data.numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.show()

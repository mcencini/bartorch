"""
Prewhitening, coil compression, and sensitivity maps
====================================================

Estimate a whitening transform from noise-only samples, compress the whitened
coil space, and calibrate in that same space. Requires the base CPU installation.
The noise sample matrix and imaging data share the same simulated coil basis.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt

bartorch.set_num_threads(1)
torch.manual_seed(4)
ncoils, nsamples, n = 4, 4096, 32
mixing = torch.eye(ncoils, dtype=torch.complex64)
mixing[1, 0] = 0.6 + 0.2j
mixing[2, 1] = 0.4j
noise = mixing @ torch.randn(ncoils, nsamples, dtype=torch.complex64)
# Coil dimension is BART dimension 3: (coils, z, y, x) in Python.
noise_data = noise.reshape(ncoils, 1, nsamples, 1)
white_noise = bt.whiten(noise_data, noise_data).reshape(ncoils, nsamples)


def covariance(samples):
    return samples @ samples.mH / samples.shape[-1]


torch.testing.assert_close(
    covariance(white_noise),
    torch.eye(ncoils, dtype=torch.complex64),
    atol=0.08,
    rtol=0.08,
)

# %%
# Apply the same coil mixing to synthetic signal data. In a real acquisition,
# receiver noise and coil signals are already expressed in a common coil basis.
# Estimate calibration maps after whitening/compression so that the encoding
# model matches the transformed k-space.
original = bt.phantom([n, n], kspace=True, ncoils=ncoils)
kspace = (mixing @ original.reshape(ncoils, -1)).reshape(ncoils, 1, n, n)
white_kspace = bt.whiten(kspace, noise_data)
compression = bt.cc(white_kspace, M=True)
compressed = bt.ccapply(white_kspace, compression, p=2)
maps = bt.ecalib(compressed, calib_size=16, maps=1)
print("k-space:", kspace.shape, "compressed:", compressed.shape, "maps:", maps.shape)
assert compressed.numel() == 2 * n * n
retained = (compressed.abs().square().sum() / white_kspace.abs().square().sum()).item()
print(f"Retained signal energy: {retained:.3f}")

# %%
# Compression trades coil information for lower memory/compute cost. Choose the
# retained rank using signal energy and reconstruction quality, not just speed.
fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
for ax, data, title in zip(
    axes,
    [covariance(noise).abs(), covariance(white_noise).abs(), maps.reshape(2, n, n)[0].abs()],
    ["Noise covariance magnitude", "Whitened covariance magnitude", "Virtual coil 1 map"],
):
    im = ax.imshow(data.numpy(), cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.show()

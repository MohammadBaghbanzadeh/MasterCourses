# Exported notebook code; may contain IPython commands.
# %%
from __future__ import print_function
import numpy as np
from PIL import Image
from sporco import util, array, plot
import sporco.metric as sm
from sporco.admm import bpdn
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Load Gen-M-Safa.jpg
img = np.array(Image.open('Gen-M-Safa.jpg').convert('L'), dtype=np.float64) / 255.0
print(f"Gen-M-Safa.jpg - Shape: {img.shape}")

# Pad image to make dimensions compatible
blksz = (8, 8)
stpsz = (2, 2)
h, w = img.shape
pad_h = (blksz[0] - h % stpsz[0]) % stpsz[0]
pad_w = (blksz[1] - w % stpsz[1]) % stpsz[1]
img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

# Add Gaussian noise
np.random.seed(12345)
sigma = 0.1
imgn = img_padded + np.random.normal(0.0, sigma, img_padded.shape)

# Extract blocks
blocks = array.extract_blocks(imgn, blksz, stpsz)
blockmeans = np.mean(blocks, axis=(0, 1))
blocks -= blockmeans
blocks = blocks.reshape(np.prod(blksz), -1)
print(f"Number of blocks: {blocks.shape[1]}")

# Load dictionary
D = util.convdicts()['G:8x8x64'].reshape(np.prod(blksz), -1)
print(f"Dictionary shape: {D.shape}")

# Set solver options
lmbda = 1e-1
opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 250,
                         'RelStopTol': 3e-3, 'AuxVarObj': False,
                         'AutoRho': {'Enabled': False}, 'rho': 1e1*lmbda})

# Solve
print("Running BPDN solver...")
b = bpdn.BPDN(D, blocks, lmbda, opt)
X = b.solve()

# Reconstruct denoised image
reconstructed = np.dot(D, X).reshape(blksz + (-1,)) + blockmeans
imgd = array.average_blocks(reconstructed, img_padded.shape, stpsz)
imgd = imgd[:h, :w]  # Crop back to original size

# Calculate PSNR
psnr_noisy = sm.psnr(img, imgn[:h, :w])
psnr_denoised = sm.psnr(img, imgd)

print(f"Noisy image PSNR: {psnr_noisy:.2f} dB")
print(f"Denoised image PSNR: {psnr_denoised:.2f} dB")
print(f"BPDN solve time: {b.timer.elapsed('solve'):.2f} s")

# Save denoised image
imgd_uint8 = np.clip(imgd * 255, 0, 255).astype(np.uint8)
Image.fromarray(imgd_uint8).save('Gen-M-Safa_denoised.png')
print("Saved: Gen-M-Safa_denoised.png")

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(imgn[:h, :w], cmap='gray')
axes[1].set_title(f'Noisy Image (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB')
axes[1].axis('off')
axes[2].imshow(imgd, cmap='gray')
axes[2].set_title(f'Denoised Image\nPSNR: {psnr_denoised:.2f} dB')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('Gen-M-Safa_comparison.png', dpi=150)
plt.show()

# %%
# Load M-PilAfkan.jpg
img = np.array(Image.open('M-PilAfkan.jpg').convert('L'), dtype=np.float64) / 255.0
print(f"M-PilAfkan.jpg - Shape: {img.shape}")

# Pad image to make dimensions compatible
blksz = (8, 8)
stpsz = (2, 2)
h, w = img.shape
pad_h = (blksz[0] - h % stpsz[0]) % stpsz[0]
pad_w = (blksz[1] - w % stpsz[1]) % stpsz[1]
img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

# Add Gaussian noise
np.random.seed(12345)
sigma = 0.1
imgn = img_padded + np.random.normal(0.0, sigma, img_padded.shape)

# Extract blocks
blocks = array.extract_blocks(imgn, blksz, stpsz)
blockmeans = np.mean(blocks, axis=(0, 1))
blocks -= blockmeans
blocks = blocks.reshape(np.prod(blksz), -1)
print(f"Number of blocks: {blocks.shape[1]}")

# Load dictionary
D = util.convdicts()['G:8x8x64'].reshape(np.prod(blksz), -1)
print(f"Dictionary shape: {D.shape}")

# Set solver options
lmbda = 1e-1
opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 250,
                         'RelStopTol': 3e-3, 'AuxVarObj': False,
                         'AutoRho': {'Enabled': False}, 'rho': 1e1*lmbda})

# Solve
print("Running BPDN solver...")
b = bpdn.BPDN(D, blocks, lmbda, opt)
X = b.solve()

# Reconstruct denoised image
reconstructed = np.dot(D, X).reshape(blksz + (-1,)) + blockmeans
imgd = array.average_blocks(reconstructed, img_padded.shape, stpsz)
imgd = imgd[:h, :w]  # Crop back to original size

# Calculate PSNR
psnr_noisy = sm.psnr(img, imgn[:h, :w])
psnr_denoised = sm.psnr(img, imgd)

print(f"Noisy image PSNR: {psnr_noisy:.2f} dB")
print(f"Denoised image PSNR: {psnr_denoised:.2f} dB")
print(f"BPDN solve time: {b.timer.elapsed('solve'):.2f} s")

# Save denoised image
imgd_uint8 = np.clip(imgd * 255, 0, 255).astype(np.uint8)
Image.fromarray(imgd_uint8).save('M-PilAfkan_denoised.png')
print("Saved: M-PilAfkan_denoised.png")

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(imgn[:h, :w], cmap='gray')
axes[1].set_title(f'Noisy Image (σ={sigma})\nPSNR: {psnr_noisy:.2f} dB')
axes[1].axis('off')
axes[2].imshow(imgd, cmap='gray')
axes[2].set_title(f'Denoised Image\nPSNR: {psnr_denoised:.2f} dB')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('M-PilAfkan_comparison.png', dpi=150)
plt.show()

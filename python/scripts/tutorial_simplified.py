# %% [markdown]
# ### Overview
# In this notebook we will generate a dataset $\mathcal X$ according to an underlying probability distribution $p_{X}(x)$, and then apply a transformation $\mathcal A(\cdot) :\mathcal X \rightarrow \mathcal Y $ to the data to map it to a transformed space $\mathcal Y$. The final goal is to train a neural network that learns the inverse mapping from $\mathcal Y$ back to $\mathcal X$.

# %% [markdown]
# #### Data Generation
# Each data point in $\mathcal X$ is a tuple $(f,\sigma)$ where both $f$ and $\sigma$ are M by M arrays. 

# %% [markdown]
# 1. Generate a dataset $\mathcal X =\{f_i,\sigma_i\}_{i=1}^{|\mathcal X|}$

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_blur_map(M=64, vmin=0.1, vmax=1, num_maps=1):
    # 1. Generate uncorrelated noise for all maps at once
    white_noise = np.random.randn(num_maps, M, M)

    # 2. & 3. Apply a Gaussian filter to each image separately
    # scipy.ndimage.gaussian_filter can process the first axis as batch if you set the sigma for each axis
    blur_maps = gaussian_filter(white_noise, sigma=(0, 4, 4))

    # 4. Normalize and scale each map
    blur_maps = (np.clip(blur_maps, -0.5, 0.5) + 0.5) * (vmax - vmin) + vmin

    return blur_maps

M=64; vmin=0.1; vmax=1; num_maps=10
blur_map = generate_blur_map(M, vmin, vmax, num_maps=num_maps)

import time
from IPython.display import display, clear_output

fig, ax = plt.subplots(figsize=[6,4])
im = ax.imshow(blur_map[0], cmap='inferno')
ax.set_title(f'Blur Map ({M}x{M}), vmin={vmin}, vmax={vmax}, scene: 0')
ax.axis('off')
cbar = plt.colorbar(im, ax=ax)
plt.show()

for i in range(num_maps):
    im.set_data(blur_map[i])
    ax.set_title(f'Blur Map ({M}x{M}), vmin={vmin}, vmax={vmax}, scene: {i}')
    # Update colorbar to match new data
    im.set_clim(vmin=blur_map[i].min(), vmax=blur_map[i].max())
    cbar.update_normal(im)
    clear_output(wait=True)
    display(fig)
    time.sleep(0.1)

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_scene(M=512, num_cells=25, num_scenes=1):
    """Generates 2D Worley (cellular) noise for multiple scenes."""
    x, y = np.mgrid[0:M, 0:M]
    grid_coords = np.stack((x, y), axis=-1)  # (M, M, 2)
    # Generate all points for all scenes: (num_scenes, num_cells, 2)
    points = np.random.rand(num_scenes, num_cells, 2) * M
    # Expand grid and points for broadcasting
    grid_coords_exp = grid_coords[None, :, :, None, :]  # (1, M, M, 1, 2)
    points_exp = points[:, None, None, :, :]            # (num_scenes, 1, 1, num_cells, 2)
    # Compute differences and squared distances
    diffs = grid_coords_exp - points_exp                # (num_scenes, M, M, num_cells, 2)
    dists_sq = np.sum(diffs**2, axis=-1)                # (num_scenes, M, M, num_cells)
    # Sort distances and get f1, f2
    sorted_dists = np.sort(dists_sq, axis=-1)           # (num_scenes, M, M, num_cells)
    f1 = np.sqrt(sorted_dists[..., 0])                  # (num_scenes, M, M)
    f2 = np.sqrt(sorted_dists[..., 1])                  # (num_scenes, M, M)
    worley_textures = f2 - f1                           # (num_scenes, M, M)

    # normalize the max value between 0.7 to 1
    worley_textures /= worley_textures.max(axis=(1,2))[:,None,None]
    worley_textures *= np.random.uniform(0.7,1,num_scenes)[:,None,None]
    
    # if num_scenes == 1:
    #     return worley_textures[0]
    return worley_textures

M=64; num_cells=100; num_scenes=10
f = generate_scene(M=M, num_cells=num_cells, num_scenes=num_scenes)

fig, ax = plt.subplots()
im = ax.imshow(f[0], cmap='magma')
ax.set_title(f'Scene ({M}x{M}), num_cells: {num_cells}, scene: 0')
ax.axis('off')
cbar = plt.colorbar(im, ax=ax)
plt.show()

# if num_scenes > 1: 
for i in range(num_scenes):
    im.set_data(f[i])
    ax.set_title(f'Scene ({M}x{M}), num_cells: {num_cells}, scene: {i}')
    # Update colorbar to match new data
    im.set_clim(vmin=f[i].min(), vmax=f[i].max())
    cbar.update_normal(im)
    clear_output(wait=True)
    display(fig)
    time.sleep(0.1)


# %% [markdown]
# #### Applying the Transformation
# 
# Now that we have the generation routines for both $f$ and $\sigma$, we can now
# implement our transformation to simulate measurements as $$y = \mathcal A_\sigma (f) + n$$
# 
# $f$ is our scene and $\sigma$ is the blur map which indicates the amount of blur 
# that each pixel in $f$ will experience as part of the transformation.

# %%
import torch
def gauss_func(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * torch.exp(-0.5*((x-mean)/sigma)**2)

def forward_op_torch(
    scene=None,
    blur_width=None,
    device=None):
    """
    Given 2d (or 3d where the first dimension is batch dim) arrays of intensity,
    doppler, and linewidth, calculate the noise free measurements at the
    specified spectral orders.

    Args:
        true_intensity (Tensor): 2d or 3d array of true intensities.
        blur_width (Tensor): 2d or 3d array of blur widths in the units
            of pixels.
        pixelated (bool): if True, take the integral of Gaussian along a pixel
            instead of impulse sampling at the midpoint.

    Returns:
        measurements (Tensor): 2d or 3d array of measurements. The first
        dimension is the batch dim (optional in case input arrays are 2d)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == None:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = scene.device
    reduce = False
    if len(scene.shape) == 2:
        reduce = True
        scene = scene[None]
        blur_width = blur_width[None]

    k, aa, bb = scene.shape
    out = torch.zeros((k,aa,bb))
    out = out.to(device=device, dtype=torch.float)
    diffrange = torch.arange(aa)[None,None,:,None]-torch.arange(aa)[None,None,None,:]
    diffrange = diffrange.to(device=device, dtype=torch.float)
    # assume columns of detector are independent
    out = torch.einsum(
        'lkij,lkj->lki',
        gauss_func(
            diffrange, 
            # a*true_doppler.permute(0,2,1)[:,:,None,:], 
            0,
            blur_width.permute(0,2,1)[:,:,None,:]
        ), 
        scene.permute(0,2,1)
    ).permute(0,2,1)
    if reduce:
        return out[0]
    else:
        return out

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Generate a scene
f = torch.from_numpy(generate_scene(M=64, num_cells=100)).to(device=device)

# Generate a blur field
blur_map = torch.from_numpy(generate_blur_map(M=64, vmin=0.1, vmax=3)).to(device=device)

# Apply the transformation
y = forward_op_torch(scene=f, blur_width=blur_map)

fig, ax = plt.subplots(1, 3, figsize=[15, 8])

im0 = ax[0].imshow(f[0].cpu().numpy(), cmap='magma')
ax[0].set_title('Scene')
cbar0 = plt.colorbar(im0, ax=ax[0], orientation='horizontal', pad=0.05)
cbar0.ax.set_xlabel('Value')

im1 = ax[1].imshow(blur_map[0].cpu().numpy(), cmap='inferno')
ax[1].set_title('Blur Width Map')
cbar1 = plt.colorbar(im1, ax=ax[1], orientation='horizontal', pad=0.05)
cbar1.ax.set_xlabel('Blur Width')

im2 = ax[2].imshow(y[0].cpu().numpy(), cmap='magma')
ax[2].set_title('Measurements')
cbar2 = plt.colorbar(im2, ax=ax[2], orientation='horizontal', pad=0.05)
cbar2.ax.set_xlabel('Measurement')

plt.show()

# %%
import deepinv as dinv
import torch.nn as nn
import torch.optim as optim

# %% [markdown]
# #### Inverse Problem: Reconstructing Blur Map from Measurements
# 
# Now we'll use DeepInv to solve the inverse problem: given measurements $y$ and scenes $f$, 
# reconstruct the blur map $\sigma$ that was used in the forward operation.
# 
# The problem is: $\min_{\sigma} \|y - \mathcal{A}_\sigma(f)\|_2^2$

# %%
# Setup the inverse problem
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data
M = 64
num_scenes = 1
f = torch.from_numpy(generate_scene(M=M, num_cells=300, num_scenes=num_scenes)).to(device=device)
true_blur_map = torch.from_numpy(generate_blur_map(M=M, vmin=0.1, vmax=3, num_maps=num_scenes)).to(device=device)

# Apply forward operator to get measurements
y = forward_op_torch(scene=f, blur_width=true_blur_map)

print(f"Data shapes:")
print(f"Scenes f: {f.shape}")
print(f"True blur map: {true_blur_map.shape}")
print(f"Measurements y: {y.shape}")

# %%
# Initialize the reconstruction (random initialization)
reconstructed_blur = torch.ones_like(true_blur_map, device=device)
reconstructed_blur.requires_grad_(True)  # Enable gradients AFTER scaling

# Create a simple prior (regularization)
def prior(x):
    """Simple prior: encourage smoothness"""
    # Total variation-like regularization
    dx = torch.diff(x, dim=1)
    dy = torch.diff(x, dim=2)
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

# Define the loss function
def reconstruction_loss(reconstructed_blur, f, y):
    """Loss function: ||y - A(f, reconstructed_blur)||_2^2"""
    predicted_y = forward_op_torch(scene=f, blur_width=reconstructed_blur)
    return torch.mean((y - predicted_y)**2) + 0.0000001*prior(reconstructed_blur)

# %%
# Optimization setup
optimizer = optim.Adam([reconstructed_blur], lr=0.05)
num_iterations = 10000

# Training loop
losses = []
for i in range(num_iterations):
    optimizer.zero_grad()
    
    loss = reconstruction_loss(reconstructed_blur, f, y)
    loss.backward()
    
    optimizer.step()
    
    # Apply bounds using a differentiable approach
    reconstructed_blur.data = torch.clamp(reconstructed_blur.data, 0.05, 5.0)
    
    losses.append(loss.item())
    
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# %%
# Visualize results
fig, axes = plt.subplots(2, 3, figsize=[18, 12])

# First row: Original data
im0 = axes[0,0].imshow(f[0].cpu().numpy(), cmap='magma')
axes[0,0].set_title('Scene (first)')
cbar0 = plt.colorbar(im0, ax=axes[0,0], orientation='horizontal', pad=0.05)

im1 = axes[0,1].imshow(true_blur_map[0].cpu().numpy(), cmap='inferno')
axes[0,1].set_title('True Blur Map (first)')
cbar1 = plt.colorbar(im1, ax=axes[0,1], orientation='horizontal', pad=0.05)

im2 = axes[0,2].imshow(y[0].cpu().numpy(), cmap='magma')
axes[0,2].set_title('Measurements (first)')
cbar2 = plt.colorbar(im2, ax=axes[0,2], orientation='horizontal', pad=0.05)

# Second row: Reconstruction results
im3 = axes[1,0].imshow(f[0].cpu().numpy(), cmap='magma')
axes[1,0].set_title('Scene (first)')
cbar3 = plt.colorbar(im3, ax=axes[1,0], orientation='horizontal', pad=0.05)

im4 = axes[1,1].imshow(reconstructed_blur[0].detach().cpu().numpy(), cmap='inferno')
axes[1,1].set_title('Reconstructed Blur Map (first)')
cbar4 = plt.colorbar(im4, ax=axes[1,1], orientation='horizontal', pad=0.05)

im5 = axes[1,2].imshow(forward_op_torch(scene=f[0:1], blur_width=reconstructed_blur[0:1])[0].detach().cpu().numpy(), cmap='magma')
axes[1,2].set_title('Reconstructed Measurements (first)')
cbar5 = plt.colorbar(im5, ax=axes[1,2], orientation='horizontal', pad=0.05)

plt.tight_layout()
plt.show()

# Plot loss convergence
plt.figure(figsize=[10, 6])
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Reconstruction Loss Convergence')
plt.yscale('log')
plt.grid(True)
plt.show()

# %%
# Quantitative evaluation
with torch.no_grad():
    mse = torch.mean((reconstructed_blur - true_blur_map)**2)
    mae = torch.mean(torch.abs(reconstructed_blur - true_blur_map))
    
    print(f"Reconstruction Quality:")
    print(f"MSE: {mse.item():.6f}")
    print(f"MAE: {mae.item():.6f}")
    print(f"Relative MSE: {mse.item() / torch.mean(true_blur_map**2):.6f}")

# %%

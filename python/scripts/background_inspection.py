from math import e

import numpy as np
import matplotlib.pyplot as plt

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
pathdir='/home/kamo/resources/slitless/data/eis_data/datasets/dset_v4/smallset/train/',
data_='eis_train_5_dsetv4.npy' # 64x64 EIS images

data = np.load(path_data+data_, allow_pickle=True).item()
param4dar, meas4dar, cube4dar = data['param3d'], data['meas'], data['datacube']

bkg = meas4dar[:,0] - param4dar[:,0]
i = 1

plt.figure()
plt.imshow(param4dar[i][0], cmap='hot')
plt.title('Param3d Int')
plt.colorbar()
plt.show()

# plt.figure()
# plt.imshow(cube4dar[0].sum(axis=-1), cmap='hot')
# plt.title('Cube Sum')
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(meas4dar[i][0], cmap='hot')
plt.title('Meas 0')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(bkg[i], cmap='hot')
plt.title('Bkg')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(meas4dar[i][0] / 1.171 - param4dar[i,0], cmap='hot')
plt.title('Diff')
plt.colorbar()
plt.show()

alphas = 0.3*np.arange(100)/100
betas = 5*np.arange(100)
errors = np.zeros((len(alphas), len(betas)))

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        error_ar = meas4dar[:,0] - (alpha+1)*param4dar[:,0] - beta
        errors[i,j] = np.sum(error_ar**2)

plt.figure()
plt.imshow(errors)
plt.colorbar()
plt.title('Errors')
plt.xlabel('Beta')
plt.ylabel('Alpha')
plt.grid(which='both', axis='both')
plt.show()


# Find the best alpha and beta
best_idx = np.unravel_index(np.argmin(errors), errors.shape)
best_alpha = alphas[best_idx[0]]
best_beta = betas[best_idx[1]]

print(f'Best Alpha: {best_alpha}')
print(f'Best Beta: {best_beta}')

# Visualize the best fit
i = 1
corrected_diff = meas4dar[i, 0] - (best_alpha + 1) * param4dar[i, 0] - best_beta

plt.figure()
plt.imshow(corrected_diff, cmap='hot')
plt.title(f'Corrected Diff (Alpha={best_alpha:.3f}, Beta={best_beta:.3f})')
plt.colorbar()
plt.show()
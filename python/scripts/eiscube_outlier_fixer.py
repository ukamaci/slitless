# 2026-01-17
# Ulas Kamaci
# This script intends to analyze, understand, and develop an automated detection
# and correction algorithm for the large negative values in the EIS cubes, which
# mess up our simulated slitless projections, hence the dataset, hence the
# training.

# Planning:
# First look up a couple of examples to understand the problem, then implement 
# the first simple algorithm that comes to your mind and iteratively refine if 
# needed. Developing the right visualization tools is key here for speedy and 
# accurate development.

# Development:
# What information do i need?
# - Would a very simple fix like masking all the values less than -10 and
# interpolating them with their nearest neighbor solve this problem? If not, 
# then what is needed?

# Any algorithm I'll develop will have two parts: Detection and Interpolation.
# Detection will have a threshold parameter. And I should choose the type of 
# interpolation.

import matplotlib.pyplot as plt
import numpy as np
import glob, eispac
from scipy.interpolate import interp1d
from slitless.eistools import fit_spectra_joblib, eis_to_ssi_interpolator
import matplotlib
# matplotlib.use('Agg')

DISP_SCALE = 13.5*1.65/1000
# Implement the simple threshold/mask/interpolate algorithm to correct the data

# Masking

# Idea: For each spectral slice of the datacube, check if there is a mixture of
# zero and nonzero pixels. Threshold could be 100 (mark them as zero if less than
# 100, nonzero otherwise). Then, mask the zero valued pixels of the slices which 
# include a mixture of zeros and nonzeros (don't mask anything if it's all zeros).

def zero_masker(cube):
    zeros = cube == 0
    zero_counts = zeros.sum(axis=(0,1))
    maskable_slices = (zero_counts>50) & (zero_counts < 64**2-50)
    mask = zeros * maskable_slices[np.newaxis, np.newaxis]
    return mask

def negative_masker(cube):
    return cube < -1000

def interpolator(cube, mask):
    interpolated_cube = cube.copy()
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            if mask[i,j].sum() == 0:
                continue
            x_axis = np.arange(cube.shape[-1])[~mask[i,j]]
            y_axis = cube[i,j][~mask[i,j]]
            interp_fn = interp1d(x_axis, y_axis, kind='cubic',
                                    bounds_error=False, fill_value=0)
            interpolated_cube[i, j, :] = interp_fn(np.arange(cube.shape[-1]))
    return interpolated_cube

# Load data from dataloader
train_dir = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v3/data/train/'
files = glob.glob(train_dir + '*.npy')
# files.sort()

# Visualize loaded data to find out sample indices with the outlier problem.
# Further visual inverstigation of the anomaly cases
# for j in (np.arange(64)):
for j in [10]:#,10,11]:
    if j==8:
        r1,r2,c = 9,10,20
    elif j==10:
        r1,r2,c = 5,6,20
    elif j==11:
        r1,r2,c = 19,20,20
    data = np.load(files[j], allow_pickle=True).item()
    meas = data['meas_0']
    inten = data['int']
    cube = data['datacube']
    mask = zero_masker(cube)
    mask += negative_masker(cube)
    maskind = np.where(mask.sum(axis=(0,1))>0)[0]
    cube_cor = interpolator(cube, mask)
    meas_cor = cube_cor.sum(axis=-1)*DISP_SCALE

    # plt.figure(figsize=(10,5))
    # plt.title(f'{j}: {data['filename']}')
    # plt.plot(np.median(cube[r1],axis=0), '-o', label=f'[{r1},{c}]')
    # plt.plot(np.median(cube[r2],axis=0), '-o', label=f'[{r2},{c}]')
    # plt.legend()
    # plt.savefig('/home/kamo/resources/slitless/figures/eiscube_correction/analysis/fig_{}.png'.format(j))

    # fig, ax = plt.subplots(1,3,figsize=(15,5))
    # plt.suptitle(f'{j}: {data['filename']}')
    # im = ax[0].imshow(meas, cmap='hot', vmin=inten.min(), vmax=inten.max())
    # plt.colorbar(im, ax=ax[0], shrink=0.8)
    # ax[0].set_title('Meas 0')
    # im = ax[1].imshow(inten, cmap='hot', vmin=inten.min(), vmax=inten.max())
    # plt.colorbar(im, ax=ax[1], shrink=0.8)
    # ax[1].set_title('Inten')
    # im = ax[2].imshow(meas_cor, cmap='hot', vmin=inten.min(), vmax=inten.max())
    # plt.colorbar(im, ax=ax[2], shrink=0.8)
    # if len(maskind)>0:
    #     ax[2].set_title('Corrected Meas')
    # else:
    #     ax[2].set_title('Uncorrected Meas')
    # plt.savefig('/home/kamo/resources/slitless/figures/eiscube_correction/results/fig_{}.png'.format(j))

    # plt.figure()
    # plt.title('Diff (inten-meas)')
    # plt.imshow(inten-meas, cmap='hot')
    # plt.colorbar()
    # plt.show()

    # for i in maskind:
    #     fig, ax = plt.subplots(1,2,figsize=(10,5))
    #     plt.suptitle('Spectral Slices')
    #     im = ax[0].imshow(cube[:,:,i], cmap='hot')
    #     plt.colorbar(im, ax=ax[0], shrink=0.8)
    #     ax[0].set_title(f'Slice {i}')
    #     im = ax[1].imshow(mask[:,:,i], cmap='hot')
    #     plt.colorbar(im, ax=ax[1], shrink=0.8)
    #     ax[1].set_title('Mask')
    #     plt.show()


pathdir = '/home/kamo/resources/slitless/data/eis_data/'
date = data['filename']
eispac.db.download_hdf5_data(
    # filename=f'eis_l0_{date}.fits.gz', 
    filename=f'eis_{date}', 
    local_top=pathdir+'l2/'
)

# Select local files (relative paths are fine)
eis_filepath = pathdir + f'l2/eis_{date}.data.h5'
template_filepath = pathdir + 'templates/fe_12_195_119.2c.template.h5'

# Load the data and fit template
# Note: read_cube() performs basic pointing corrections and
#       applies the pre-flight radiometric calibration
data_cube = eispac.read_cube(eis_filepath, window=195.119)
tmplt = eispac.read_template(template_filepath)
param3d, err3d, status3d, cube_cor = fit_spectra_joblib(
    data_cube, tmplt, n_jobs=48, component=0)

r0,c0 = data['patch_coords']
dc_cor = interpolator(data_cube.data, data_cube.mask)
dc_ssi = eis_to_ssi_interpolator(cube_cor, lamdim=21)

plt.imshow(data_cube.mask[r0:r0+64,c0:c0+64].sum(axis=2), cmap='hot')

fig, ax = plt.subplots(1,4,figsize=(20,5))
plt.suptitle(f'{j}: {data['filename']}')
im = ax[0].imshow(data_cube.data[r0:r0+64,c0:c0+64].sum(axis=2), cmap='hot')
plt.colorbar(im, ax=ax[0], shrink=0.8)
ax[0].set_title('Meas 0')
im = ax[1].imshow(data_cube.mask[r0:r0+64,c0:c0+64].sum(axis=2), cmap='hot')
plt.colorbar(im, ax=ax[1], shrink=0.8)
ax[1].set_title('Mask')
im = ax[2].imshow(dc_cor[r0:r0+64,c0:c0+64].sum(axis=2), cmap='hot')
plt.colorbar(im, ax=ax[2], shrink=0.8)
ax[2].set_title('Corrected Meas 0')
im = ax[3].imshow(cube_cor[r0:r0+64,c0:c0+64].sum(axis=2), cmap='hot')
plt.colorbar(im, ax=ax[3], shrink=0.8)
ax[3].set_title('Fit-Corrected Meas 0')
plt.show()
# plt.savefig('/home/kamo/resources/slitless/figures/eiscube_correction/analysis/tmp.png')

# Visualize the result to verify it works. If not, find out why, and fix it.
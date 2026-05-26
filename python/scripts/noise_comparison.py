import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/eis_train_5_dsetv6.npy'
IDX = 0

data = np.load(path_data, allow_pickle=True).item()
param3d = data['param3d'][IDX]

spectral_orders = [0, -1, 1]
Sr = Source(param3d=param3d, pix=False)
Imgr = Imager(
    pixelated=True,
    spectral_orders=spectral_orders,
    dispersion_scale=0.022275,
    mid_wavelength=195.119,
)

gaussian_dbsnrs = [25, 50, 100, None]
poisson_dbsnrs  = [10, 20, 30, None]

def simulate(dbsnr, noise_model):
    meas = Imgr.get_measurements(sources=Sr, dbsnr=dbsnr, noise_model=noise_model,
                                 no_noise=dbsnr is None)
    return meas[0]  # zeroth order

for noise_model, dbsnrs in [('gaussian', gaussian_dbsnrs), ('poisson', poisson_dbsnrs)]:
    fig, axs = plt.subplots(1, len(gaussian_dbsnrs), figsize=(15, 4))
    fig.suptitle(f'{noise_model.capitalize()} noise — order 0')
    for ax, dbsnr in zip(axs, dbsnrs):
        im = ax.imshow(simulate(dbsnr, noise_model), cmap='hot')
        ax.set_title(f'dbsnr={dbsnr}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()

plt.show()

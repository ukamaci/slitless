import matplotlib.pyplot as plt
import eispac
import numpy as np
from tqdm import tqdm
import os
from slitless.forward import datacube_generator, Source, Imager

pathdir = '/home/kamo/resources/slitless/data/eis_data/'

dates = [
   '20070124_181113',
]

if __name__ == '__main__':
    for num, date in enumerate(tqdm(dates)):
        print('{}/{}: {}'.format(num+1,len(dates), date))
        # download file if it doesn't already exist
        eispac.db.download_hdf5_data(
            # filename=f'eis_l0_{date}.fits.gz', 
            filename=f'eis_{date}', 
            local_top=pathdir+'l2/')

        # Select local files (relative paths are fine)
        eis_filepath = pathdir + f'l2/eis_{date}.data.h5'
        template_filepath = pathdir + 'templates/fe_12_195_119.2c.template.h5'

        # Load the data and fit template
        # Note: read_cube() performs basic pointing corrections and
        #       applies the pre-flight radiometric calibration
        data_cube = eispac.read_cube(eis_filepath, window=195.119)
        tmplt = eispac.read_template(template_filepath)

        fit_path = '/home/kamo/resources/slitless/data/eis_data/fits/eis_20070124_181113.fe_12_195_119.2c-0.fit.h5'
        if os.path.exists(fit_path):
            fit_res = eispac.read_fit(fit_path)
        else:
            # Fit the spectra
            fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max', unsafe_mp=True)

        inten = fit_res.get_map(0, 'int').data
        vel = fit_res.get_map(0, 'vel').data
        width = fit_res.get_map(0, 'width').data
    
        sr = Source(inten=inten, vel=vel, width=width, pix=False)
        im = Imager()
        srpix = im.topix(sr)

        lamdim = 21
        dc_gauss = datacube_generator(srpix.param3d, lamdim=lamdim, pixelated=False)
        dc_gauss /= im.dispersion_scale/1000
        
        wavelength_grid_target = (
            sr.wavelength + 
            im.dispersion_scale/1000 * (np.arange(lamdim)-lamdim//2)
        )

        wavelength_grid_source = data_cube.wavelength

        # Interpolate data_cube.data (shape: [y, x, lambda_src]) from wavelength_grid_source (shape: [y, x, lambda_src])
        # into wavelength_grid_target (shape: [lambda_tgt])

        from scipy.interpolate import interp1d

        # Prepare output array with new wavelength dimension
        y_sz, x_sz, _ = data_cube.data.shape
        lamdim = wavelength_grid_target.shape[0]
        data_cube_interpolated = np.zeros((y_sz, x_sz, lamdim), dtype=data_cube.data.dtype)

        for iy in range(y_sz):
            for ix in range(x_sz):
                wl_src = wavelength_grid_source[iy, ix, :]
                flux_src = data_cube.data[iy, ix, :]

                # Check for monotonicity & valid values
                valid = np.isfinite(wl_src) & np.isfinite(flux_src)
                if valid.sum() < 2:
                    data_cube_interpolated[iy, ix, :] = np.nan
                    continue

                try:
                    interp_fn = interp1d(wl_src[valid], flux_src[valid], kind='cubic',
                                         bounds_error=False, fill_value=0)
                    data_cube_interpolated[iy, ix, :] = interp_fn(wavelength_grid_target)
                except Exception as e:
                    data_cube_interpolated[iy, ix, :] = np.nan

        # Now data_cube_interpolated is aligned to wavelength_grid_target at each pixel
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        im0 = ax[0].imshow(dc_gauss.sum(axis=0), aspect='auto', cmap='jet')
        ax[0].set_title('Gauss Fitted')
        cbar0 = plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        im1 = ax[1].imshow(data_cube_interpolated.sum(axis=2), aspect='auto', cmap='jet')
        ax[1].set_title('Interpolated to SSI')
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        im2 = ax[2].imshow(data_cube.data.sum(axis=2), aspect='auto', cmap='jet')
        ax[2].set_title('EIS Grid')
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        plt.show()

        # Now data_cube_interpolated is aligned to wavelength_grid_target at each pixel
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        im0 = ax[0].imshow(dc_gauss[:,165,:].T, aspect='auto', cmap='jet')
        cbar0 = plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        ax[0].set_title('Gauss Fitted')
        im1 = ax[1].imshow(data_cube_interpolated[165], aspect='auto', cmap='jet')
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_title('Interpolated to SSI')
        im2 = ax[2].imshow(data_cube.data[165], aspect='auto', cmap='jet')
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        ax[2].set_title('EIS Grid')
        plt.show()

        # Now data_cube_interpolated is aligned to wavelength_grid_target at each pixel
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        im0 = ax[0].imshow(dc_gauss[:,:,116].T, aspect='auto', cmap='jet')
        cbar0 = plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        ax[0].set_title('Gauss Fitted')
        im1 = ax[1].imshow(data_cube_interpolated[:,116], aspect='auto', cmap='jet')
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_title('Interpolated to SSI')
        im2 = ax[2].imshow(data_cube.data[:,116], aspect='auto', cmap='jet')
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        ax[2].set_title('EIS Grid')
        plt.show()

        for i, j in zip(np.arange(0,y_sz,15),np.arange(0,x_sz,15)):
            c0_x, c0_y = fit_res.get_fit_profile(0, coords=[i,j], num_wavelengths=100)
            c_x, c_y = fit_res.get_fit_profile(coords=[i,j], num_wavelengths=100)
            plt.figure(figsize=(15,5))
            plt.plot(wavelength_grid_target, dc_gauss[:,i,j], '-o', label='Gauss')
            # plt.plot(c0_x, c0_y, '-', color='purple', label='Interpolated Gauss')
            plt.plot(c_x, c_y, '-', color='black', label='Combined Fit')
            plt.plot(wavelength_grid_target, data_cube_interpolated[i,j], '-o', label='Interpolated')
            plt.plot(wavelength_grid_source[i,j], data_cube.data[i,j], '-o', label='EIS Grid')
            plt.grid(which='both', axis='both')
            plt.legend()
            plt.show()
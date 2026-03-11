# 2025-12-24
# Ulas Kamaci
# Post-APJ revision EIS fitting & interpolation routine 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eispac, os
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from slitless.forward import forward_op_tomo_3d, Source
from joblib import Parallel, delayed

from slitless.eistools import eis_to_ssi_interpolator, eis_random_cropper, quickplot
from joblib import Parallel, delayed
from eispac.extern.mpfit import mpfit
import eispac.core.fitting_functions as fit_fns
from eispac.core.scale_guess import scale_guess
from eispac.instr import calc_velocity # <--- We use the official function now
import copy

pathdir = '/home/kamo/resources/slitless/data/eis_data/'
savedir = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v3/'
WAVELENGTH = 195.119
DISPERSION_SCALE = 22.275

def _worker_fit_chunk(args):
    """
    Worker returns RAW fitted parameters only.
    Velocity calculation is strictly forbidden here to avoid 'dirty reference' errors.
    """
    wave_row, int_row, err_row, template, parinfo, min_points = args
    n_pixels = int_row.shape[0] 
    n_wave = int_row.shape[1]
    
    # We need to know how many parameters (gaussians * 3 + poly)
    n_params = len(parinfo) 
    n_gauss = template['n_gauss']
    n_poly = template['n_poly']
    min_w = template['wmin']
    max_w = template['wmax']
    oldguess = template['fit']
    
    # Allocate output for RAW PARAMETERS
    # Shape: (256 pixels, n_params)
    out_params = np.zeros((n_pixels, n_params))
    out_status = np.zeros(n_pixels)
    out_chi2   = np.zeros(n_pixels)

    # Re-usable masking array
    mask = np.ones(n_wave)

    for i in range(n_pixels):
        wave_i = wave_row[i, :]
        int_i = int_row[i, :]
        err_i = err_row[i, :]

        loc_good = np.where((err_i > 0) & (wave_i >= min_w) & (wave_i <= max_w))[0]
        if len(loc_good) < min_points: 
            out_status[i] = -1
            continue

        w_g = wave_i[loc_good]
        i_g = int_i[loc_good]
        e_g = err_i[loc_good]

        # Update guess (This DIRDIES the parinfo object, which is why we can't use it for Ref Wavelength)
        newguess = scale_guess(w_g, i_g, oldguess, n_gauss, n_poly)
        for k in range(len(newguess)): parinfo[k]['value'] = newguess[k]

        fa = {'x': w_g, 'y': i_g, 'error': e_g, 'n_gauss': n_gauss, 'n_poly': n_poly}
        out = mpfit(fit_fns.multigaussian_deviates, parinfo=parinfo,
                    functkw=fa, xtol=1e-6, ftol=1e-6, gtol=1e-6,
                    maxiter=2000, quiet=1)
        
        out_status[i] = out.status
        # Normalized Chi2
        out_chi2[i] = out.fnorm / max(1, out.dof)
        
        if out.status > 0:
            out_params[i, :] = out.params

    return out_params, out_status, out_chi2



dates = [
   '20070124_181113',
   '20070127_034020',
   '20070128_061012',
   '20070326_183342',
   '20070512_094534',
   '20070702_120742',
#    '20070708_105820',
#    '20070929_140227',
#    '20071004_140220',
#    '20071211_002416',
#    '20080106_105243',
#    '20080203_073210',
#    '20091211_195014',
#    '20090114_233904',
#    '20150807_022045'
]

#example_line: 'https://sdc.uio.no/vol/fits/eis/level0/2006/12/01/eis_l0_20061201_151432.fits.gz\n'
with open(savedir+'/meta/file_names.txt') as f:
    dates = [line.split('/')[-1].split('.')[0][7:] for line in f.read().splitlines()]

dates = dates[522:]
if __name__ == '__main__':
    # NB: The "name guard" above is important for running safe, parallel fitting
    #     in a stand-alone script. If running in an interactive shell, eispac will
    #     default to a single core. If you _really_ know what you are doing,
    #     you can override it. Just be careful.

    for num, date in enumerate(tqdm(dates)):
        try:
            print('{}/{}: {}'.format(num+1,len(dates), date))
            # download file if it doesn't already exist

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

            if min(data_cube.data.shape[:2]) < 64:
                print(f'{date} FOV dimensions {data_cube.data.shape[:2]} too small (<64), skipping')
                continue

            #remove the downloaded data and head files and the fit file
            os.remove(eis_filepath)
            os.remove(pathdir + f'l2/eis_{date}.head.h5')

            # Fit the spectra
            fit_path = f'/home/kamo/resources/slitless/data/eis_data/fits/eis_{date}.fe_12_195_119.2c-0.fit.h5'
            # if os.path.exists(fit_path):
            #     fit_res = eispac.read_fit(fit_path)
            # else:
            #     fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu=1, unsafe_mp=False)
            #     # eispac.save_fit(fit_res, save_dir=f'/home/kamo/resources/slitless/data/eis_data/fits/')

            # # Get the first (dominant) Gaussian component from the fit
            # param3d = np.stack(( # shape: (3,256,256)?
            #     fit_res.get_map(0, 'int').data, 
            #     fit_res.get_map(0, 'vel').data, 
            #     fit_res.get_map(0, 'width').data
            # ))
    # 3. FIT OR LOAD FIT
            if os.path.exists(fit_path):
                print(f"Loading cached fit for {date}...")
                fit_res = eispac.read_fit(fit_path)
                param3d = np.stack(( 
                    fit_res.get_map(0, 'int').data, 
                    fit_res.get_map(0, 'vel').data, 
                    fit_res.get_map(0, 'width').data
                ))
            else:
                print(f"Computing fit for {date} (Joblib)...")
                tmplt = eispac.read_template(template_filepath)
                
                # Data to RAM
                safe_data = np.array(data_cube.data).astype(np.float64)
                safe_wave = np.array(data_cube.wavelength).astype(np.float64)
                safe_errs = np.array(data_cube.uncertainty.array).astype(np.float64)
                n_rows = safe_data.shape[0]
                
                # Prepare Tasks
                # We use the template's parinfo as the base
                p_base = tmplt.parinfo
                t_base = tmplt.template
                tasks = []
                
                for y in range(n_rows):
                    tasks.append((
                        safe_wave[y, :, :], 
                        safe_data[y, :, :], 
                        safe_errs[y, :, :], 
                        t_base, 
                        copy.deepcopy(p_base), # Worker gets its own copy to dirty up
                        7
                    ))

                # Run Parallel Fit
                results = Parallel(n_jobs=48, backend="loky", verbose=0, batch_size='auto')(
                    delayed(_worker_fit_chunk)(t) for t in tasks
                )
                
                # Unpack Results
                # results[y] = (out_params, out_status)
                full_params = np.stack([r[0] for r in results]) # (Y, X, Params)
                full_status = np.stack([r[1] for r in results])
                
                # --- POST-PROCESSING (The "Clean" Phase) ---
                # 1. Identify Component Indices
                # Primary line is Gaussian 0.
                # Index Logic: 0=Peak, 1=Centroid, 2=Width
                idx_peak = 0
                idx_cent = 1
                idx_width = 2
                
                raw_peak = full_params[:, :, idx_peak]
                raw_cent = full_params[:, :, idx_cent]
                raw_width = full_params[:, :, idx_width]
                
                # 2. Calculate Intensity (Area)
                # Area = sqrt(2pi) * Peak * Width
                intensity = np.sqrt(2 * np.pi) * raw_peak * raw_width
                
                # 3. Calculate Velocity using OFFICIAL Function
                # Use 'tmplt.parinfo' which was never sent to the workers
                # This ensures 'rest_wave' is the original Laboratory Wavelength.
                rest_wave = tmplt.parinfo[idx_cent]['value'] 
                
                # Use eispac's own function to handle the math
                velocity = calc_velocity(raw_cent, rest_wave)
                
                # 4. Filter Bad Fits
                bad_mask = (full_status <= 0)
                intensity[bad_mask] = 0
                velocity[bad_mask] = 0
                raw_width[bad_mask] = 0
                
                # 5. Pack
                param3d = np.stack((intensity, velocity, raw_width))
                
                # Cache it
                # np.save(os.path.join(fit_savedir, f'param3d_{date}.npy'), param3d)
                
            # Interpolate the data-cube into SSI detector coordinates
            data_cube_i = eis_to_ssi_interpolator(data_cube, lamdim=21)

            # Crop the data-cube and Gaussian params prior to simulating measurements
            # param3d_c, dc_c = eis_data_cropper(param3d=param3d, data_cube=data_cube_i)
            param3d_c, dc_c, patch_coords = eis_random_cropper(param3d=param3d, data_cube=data_cube_i, numcopies=15)
            fig, ax = quickplot(array=param3d, patch_coords=patch_coords, title=f'{date} Full')
            fig.savefig(savedir+f'figs/data_{date}_full.png')
            plt.close(fig)

            # Feed the data-cubes through SSI forward model
            for i in range(len(dc_c)):
                meas = forward_op_tomo_3d(dc_c[i].transpose(2,0,1), orders=[0,-1,1,-2,2])
                out = {
                    'meas_0': meas[0],
                    'meas_-1': meas[1],
                    'meas_1': meas[2],
                    'meas_-2': meas[3],
                    'meas_2': meas[4],
                    'int': param3d_c[i][0],
                    'vel': param3d_c[i][1],
                    'width': param3d_c[i][2],
                    'datacube': dc_c[i],
                    'filename': f'{date}',
                    'patch_coords': patch_coords[i]
                }
                np.save(savedir+f'data/data_{date}_{i}.npy', out)
        except:
            continue


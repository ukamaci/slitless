import numpy as np
import matplotlib.pyplot as plt
import eispac
import sys, os
from slitless.forward import Source, Imager, datacube_generator
from slitless.recon import gauss_pmf_fitter2, smart_fit_spectra_joblib

def main():
    # 1. Load Data
    path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
    data_file = 'eis_train_5_dsetv4.npy'
    
    print(f"Loading data from {path_data + data_file}...")
    try:
        data = np.load(path_data + data_file, allow_pickle=True).item()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Using the 5th patch as an example
    true_param_phy = data['param3d'][4] 
    
    # 2. Setup Imager and Source
    # Note: Param3d from dsetv4 is in physical units, so pix=False
    imager = Imager(pixelated=True, spectral_orders=[0, -1, 1])
    source = Source(param3d=true_param_phy, pix=False)
    
    # Scale intensity for the pixel conversions
    imager.intenscale = true_param_phy[0].max()
    imager.topix(source)
    true_param_pix = imager.srpix.param3d
    
    # 3. Generate the data cube (in pixel space, matching what 'smart' produces internally)
    lamdim = 21
    print("Generating true datacube sandbox...")
    cube = datacube_generator(true_param_pix, pixelated=True, lamdim=lamdim)
    
    # --- PMF Fitter ---
    print("Fitting using PMF Fitter...")
    recon_pmf_pix = gauss_pmf_fitter2(cube)
    
    # --- MPFit Joblib Fitter ---
    print("Fitting using MPFit Joblib Fitter...")
    template_filepath = '/home/kamo/resources/slitless/data/eis_data/templates/fe_12_195_119.2c.template.h5'
    tmplt = eispac.read_template(template_filepath)
    
    # Construct wave array and scale cube physically as done in smart()
    wave = imager.mid_wavelength + imager.dispersion_scale * (np.arange(lamdim) - lamdim // 2)
    cube_scaled = cube / imager.dispersion_scale * imager.intenscale
    
    recon_mpfit_phy = smart_fit_spectra_joblib(cube_scaled, tmplt, wave=wave, n_jobs=-1)
    
    # Scale MPFit output back to pixel units to compare fairly against PMF
    SPEED_OF_LIGHT = 299792.458
    recon_mpfit_pix = recon_mpfit_phy.copy()
    recon_mpfit_pix[0] /= imager.intenscale
    rest_wave = tmplt.parinfo[1]['value']
    actual_wave = rest_wave * (1 + recon_mpfit_pix[1] / SPEED_OF_LIGHT)
    recon_mpfit_pix[1] = (actual_wave - imager.mid_wavelength) / imager.dispersion_scale
    recon_mpfit_pix[2] = recon_mpfit_pix[2] / imager.dispersion_scale
    
    # 4. Evaluate Metrics
    def compute_rmse(true, est):
        return np.sqrt(np.mean((true - est)**2, axis=(-1, -2)))
        
    rmse_pmf = compute_rmse(true_param_pix, recon_pmf_pix)
    rmse_mpf = compute_rmse(true_param_pix, recon_mpfit_pix)
    
    print("\n--- RMSE (Pixel Units) ---")
    print(f"PMF Fitter -> Int: {rmse_pmf[0]:.4f} | Vel: {rmse_pmf[1]:.4f} | Wid: {rmse_pmf[2]:.4f}")
    print(f"MPFit      -> Int: {rmse_mpf[0]:.4f} | Vel: {rmse_mpf[1]:.4f} | Wid: {rmse_mpf[2]:.4f}")
    
    # 5. Visualizations
    
    # Select a bright pixel to plot the 1D spectrum
    iy, ix = np.unravel_index(np.argmax(true_param_pix[0]), true_param_pix[0].shape)
    print(f"\nPlotting 1D spectra at brightest pixel: (y={iy}, x={ix})")
    
    # Generate fitted cubes from extracted params to overlay
    cube_pmf_pix = datacube_generator(recon_pmf_pix, pixelated=True, lamdim=lamdim)
    cube_mpf_pix = datacube_generator(recon_mpfit_pix, pixelated=True, lamdim=lamdim)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Pixel Space Overlay
    x_pix = np.arange(lamdim)
    axes[0].plot(x_pix, cube[:, iy, ix], 'ko', label='True Cube', markersize=6)
    axes[0].plot(x_pix, cube_pmf_pix[:, iy, ix], 'b-', label='PMF Fit', linewidth=2)
    axes[0].plot(x_pix, cube_mpf_pix[:, iy, ix], 'r--', label='MPFit Joblib', linewidth=2)
    axes[0].set_title(f'1D Spectrum at ({ix}, {iy}) - Pixel Space')
    axes[0].set_xlabel('Pixel Index (lambda dimension)')
    axes[0].set_ylabel('Intensity [Pixel Units]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Physical Space Overlay
    axes[1].plot(wave, cube_scaled[:, iy, ix], 'ko', label='True Cube (Scaled)', markersize=6)
    
    pmf_phy_curve = cube_pmf_pix[:, iy, ix] / imager.dispersion_scale * imager.intenscale
    mpf_phy_curve = cube_mpf_pix[:, iy, ix] / imager.dispersion_scale * imager.intenscale
    
    axes[1].plot(wave, pmf_phy_curve, 'b-', label='PMF Fit', linewidth=2)
    axes[1].plot(wave, mpf_phy_curve, 'r--', label='MPFit Joblib', linewidth=2)
    axes[1].set_title(f'1D Spectrum at ({ix}, {iy}) - Physical Space')
    axes[1].set_xlabel('Wavelength [Angstroms]')
    axes[1].set_ylabel('Intensity [Physical Units]')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Full 2D Parameter Maps
    source.plot(title="True Parameters")
    Source(param3d=recon_pmf_pix, pix=True).plot(title="PMF Extracted Parameters")
    Source(param3d=recon_mpfit_pix, pix=True).plot(title="MPFit Extracted Parameters")

if __name__ == '__main__':
    main()

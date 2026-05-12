import os
import numpy as np
import matplotlib.pyplot as plt
import eispac
from copy import deepcopy
from slitless.recon import smart_fit_spectra_joblib

# Constants
DISP_SCALE_A = 0.022275
WAVELENGTH = 195.119
SPEED_OF_LIGHT = 299792.458

def gaussian_eval(wave_arr, intensity, velocity, width, rest_wave):
    center = rest_wave * (1 + velocity / SPEED_OF_LIGHT)
    peak = intensity / (np.sqrt(2 * np.pi) * width)
    return peak * np.exp(-0.5 * ((wave_arr - center) / width)**2)

def main():
    data_file_v4 = '/home/kamo/resources/slitless/data/datasets/baseline/eis_train_5_dsetv4.npy'
    tmplt_file = '/home/kamo/resources/slitless/data/eis_data/templates/fe_12_195_119.2c.template.h5'
    
    data_v4 = np.load(data_file_v4, allow_pickle=True).item()
    
    datacubes = data_v4['datacube']  # The physical scaled cube is identical in both
    param3ds_v4 = data_v4['param3d']
    
    sample_image = 0
    lamdim = datacubes.shape[-1]
    sample_y, sample_x = np.unravel_index(np.argmax(param3ds_v4[sample_image, 0]), param3ds_v4[sample_image, 0].shape)
    
    cube_phys = (datacubes[sample_image] / DISP_SCALE_A).transpose(2, 0, 1)
    spectrum = cube_phys[:, sample_y, sample_x]
    
    wave = WAVELENGTH + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)
    tmplt = eispac.read_template(tmplt_file)
    rest_wave = tmplt.parinfo[1]['value']
    
    print("\n=== SIMULATING DSET_V5 ON FULL 64x64 PATCH ===")
    param3ds_v5 = smart_fit_spectra_joblib(cube_phys, tmplt, wave=wave, n_jobs=8, component=0)

    int_v4, vel_v4, wid_v4 = param3ds_v4[sample_image, :, sample_y, sample_x]
    int_v5, vel_v5, wid_v5 = param3ds_v5[:, sample_y, sample_x]
    
    print("\n=== PIXEL GROUND TRUTHS ===")
    print(f"dset_v4 (Relative) - Int: {int_v4:7.3f}, Vel: {vel_v4:7.3f} km/s, Wid: {wid_v4 * SPEED_OF_LIGHT / WAVELENGTH:7.3f} km/s")
    print(f"Simulated v5 (Absolute) - Int: {int_v5:7.3f}, Vel: {vel_v5:7.3f} km/s, Wid: {wid_v5 * SPEED_OF_LIGHT / WAVELENGTH:7.3f} km/s")

    print("\n=== RUNNING SMART_FIT_SPECTRA_JOBLIB ON 1x1 CROP ===")
    crop = cube_phys[:, sample_y:sample_y+1, sample_x:sample_x+1]
    recon_smart = smart_fit_spectra_joblib(crop, tmplt, wave=wave, n_jobs=1, component=0)
    int_smart, vel_smart, wid_smart = recon_smart[:, 0, 0]
    print(f"smart_fit 1x1    - Int: {int_smart:7.3f}, Vel: {vel_smart:7.3f} km/s, Wid: {wid_smart * SPEED_OF_LIGHT / WAVELENGTH:7.3f} km/s\n")

    # --- PLOTTING SECTION ---
    wmin, wmax = tmplt.template['wmin'], tmplt.template['wmax']
    loc_good = np.where((wave >= wmin) & (wave <= wmax))[0]
    w_g, s_g = wave[loc_good], spectrum[loc_good]
    w_dense = np.linspace(w_g.min(), w_g.max(), 300)
    
    fig = plt.figure(figsize=(14, 10))

    # Top Left: Absolute Velocity Map
    vmax = np.max(np.abs(param3ds_v5[1]))
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(param3ds_v5[1], cmap='seismic', vmin=-vmax, vmax=vmax)
    ax1.set_title(f'Absolute Velocity Map (Simulated v5)\nMedian: {np.median(param3ds_v5[1]):.2f} km/s')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Top Right: Relative Velocity Map
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(param3ds_v4[sample_image, 1], cmap='seismic', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'Relative Velocity Map (dset_v4)\nMedian: {np.median(param3ds_v4[sample_image, 1]):.2f} km/s')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Bottom: 1D Spectrum Fit
    ax_spec = plt.subplot(2, 1, 2)
    ax_spec.plot(w_g, s_g, 'ko', label='Raw Data (SSI)')
    ax_spec.plot(w_dense, gaussian_eval(w_dense, int_smart, vel_smart, wid_smart, rest_wave), 'g:', linewidth=3, 
            label=f"smart_fit 1x1")
    ax_spec.plot(w_dense, gaussian_eval(w_dense, int_v5, vel_v5, wid_v5, rest_wave), 'b-', linewidth=2, 
            label=f"Absolute Vel (Simulated v5)")
    ax_spec.plot(w_dense, gaussian_eval(w_dense, int_v4, vel_v4, wid_v4, rest_wave), 'r--', linewidth=2, 
            label=f"Relative Vel (dset_v4)")
    ax_spec.set_title(f'Spectrum Fit at Pixel ({sample_y}, {sample_x})')

    # Add text box
    text_str = (
        f"Absolute Vel (Simulated v5):\n"
        f" Int: {int_v5:7.1f}\n"
        f" Vel: {vel_v5:7.2f} km/s\n"
        f" Wid: {wid_v5 * SPEED_OF_LIGHT / WAVELENGTH:7.2f} km/s\n\n"
        f"Relative Vel (dset_v4):\n"
        f" Int: {int_v4:7.1f}\n"
        f" Vel: {vel_v4:7.2f} km/s\n"
        f" Wid: {wid_v4 * SPEED_OF_LIGHT / WAVELENGTH:7.2f} km/s"
        f"\n\nsmart_fit 1x1:\n"
        f" Int: {int_smart:7.1f}\n"
        f" Vel: {vel_smart:7.2f} km/s\n"
        f" Wid: {wid_smart * SPEED_OF_LIGHT / WAVELENGTH:7.2f} km/s"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax_spec.text(0.02, 0.95, text_str, transform=ax_spec.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)
            
    ax_spec.legend(loc='lower center')
    ax_spec.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sandbox_verification.png', dpi=200, bbox_inches='tight')
    print("Saved definitive proof figure to 'sandbox_verification.png'")

if __name__ == '__main__':
    main()
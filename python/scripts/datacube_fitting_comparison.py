import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import eispac
from slitless.recon import gauss_pmf_fitter, smart_fit_spectra_joblib

# Constants for EIS / SMART unit conversions
DISP_SCALE_A = 0.022275  # Angstroms per pixel
WAVELENGTH = 195.119     # Angstroms
SPEED_OF_LIGHT = 299792.458  # km/s


def get_metrics(true, est):
    """Calculate RMSE and bias over the two spatial dimensions."""
    rmse = np.sqrt(np.mean((true - est) ** 2, axis=(-1, -2)))
    bias = np.mean(est - true, axis=(-1, -2))
    return rmse, bias


def make_background_free_template(template):
    tmplt = copy.deepcopy(template)
    n_gauss = tmplt.template.get('n_gauss', 1)
    # Explicitly remove the background polynomial from the model
    tmplt.template['n_poly'] = 0
    # Truncate parinfo and the initial fit guess to only include Gaussian parameters
    num_gauss_params = 3 * n_gauss
    tmplt.parinfo = tmplt.parinfo[:num_gauss_params]
    if 'fit' in tmplt.template:
        tmplt.template['fit'] = tmplt.template['fit'][:num_gauss_params]
    return tmplt


def pmf_to_physical(pmf_out):
    recon = np.zeros_like(pmf_out)
    recon[0] = pmf_out[0] * DISP_SCALE_A
    recon[1] = pmf_out[1] * DISP_SCALE_A * SPEED_OF_LIGHT / WAVELENGTH
    recon[2] = pmf_out[2] * DISP_SCALE_A
    return recon


def fit_pmf(cube_phys):
    cube_safe = np.clip(cube_phys, 0.0, None)
    pmf_out = gauss_pmf_fitter(cube_safe)
    return pmf_to_physical(pmf_out)


def fit_pixel_spectrum(wave, spectrum, template):
    from copy import deepcopy
    from eispac.core.fitting_functions import multigaussian_deviates, multigaussian
    from eispac.core.scale_guess import scale_guess
    from eispac.extern.mpfit import mpfit

    p_base = deepcopy(template.parinfo)
    t_base = template.template
    n_gauss = t_base['n_gauss']
    n_poly = t_base['n_poly']

    guess = scale_guess(wave, spectrum, t_base['fit'], n_gauss, n_poly)
    for k in range(len(guess)):
        p_base[k]['value'] = guess[k]

    functkw = {
        'x': wave,
        'y': spectrum,
        'error': np.ones_like(spectrum),
        'n_gauss': n_gauss,
        'n_poly': n_poly
    }

    fit = mpfit(
        multigaussian_deviates,
        parinfo=p_base,
        functkw=functkw,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        maxiter=2000,
        quiet=1
    )

    if fit.status <= 0:
        raise RuntimeError('Spectrum fit failed for selected pixel')

    params = fit.params
    wave_dense = np.linspace(wave.min(), wave.max(), 400)
    combined_dense = multigaussian(params, wave_dense, n_gauss, n_poly)

    gauss_components = []
    for g in range(n_gauss):
        p_comp = params.copy()
        for j in range(n_gauss):
            if j != g:
                p_comp[3*j] = 0.0
        p_comp[3*n_gauss:] = 0.0
        gauss_components.append(multigaussian(p_comp, wave_dense, n_gauss, n_poly))

    bg_params = params.copy()
    bg_params[:3*n_gauss] = 0.0
    background_dense = multigaussian(bg_params, wave_dense, n_gauss, n_poly)

    return params, wave_dense, combined_dense, gauss_components, background_dense


def ideal_param3d_curve(wave, intensity, velocity, width):
    center = WAVELENGTH * (1 + velocity / SPEED_OF_LIGHT)
    # Convert total intensity to peak flux density to match eispac's point-evaluation
    peak = intensity / (np.sqrt(2 * np.pi) * width)
    return peak * np.exp(-0.5 * ((wave - center) / width)**2)


def plot_spectrum_fit(ax, wave, spectrum, wave_dense, combined, components, background, ideal_curve, coords, method_name="", text_str=""):
    ax.plot(wave, spectrum, 'ko', label='Data')
    ax.plot(wave_dense, combined, 'r-', label='Combined fit', linewidth=2)
    colors = ['b', 'g', 'm']
    for i, comp in enumerate(components):
        ax.plot(wave_dense, comp, colors[i % len(colors)] + '--', label=f'Gaussian {i+1}')
    ax.plot(wave_dense, background, 'c-.', label='Background', linewidth=1.5)
    ax.plot(wave_dense, ideal_curve, 'k:', label='Ideal param3d', linewidth=2)
    ax.set_xlabel('Wavelength [Å]')
    ax.set_ylabel('Flux density')
    title_str = f'Spectrum fit at pixel (y={coords[0]}, x={coords[1]})'
    if method_name:
        title_str += f' [{method_name}]'
    ax.set_title(title_str)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)
    if text_str:
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace', bbox=props)


def main():
    path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
    data_file = 'eis_train_5_dsetv4.npy'
    path_templates = '/home/kamo/resources/slitless/data/eis_data/templates/'

    print(f"Loading dataset: {data_file}...")
    data = np.load(os.path.join(path_data, data_file), allow_pickle=True).item()

    datacubes = data['datacube']
    param3ds = data['param3d']
    num_images = datacubes.shape[0]
    lamdim = datacubes.shape[-1]

    print("Preparing EIS fitting templates...")
    template_2c = eispac.read_template(os.path.join(path_templates, 'fe_12_195_119.2c.template.h5'))
    template_1c = eispac.read_template(os.path.join(path_templates, 'fe_12_195_119.1c.template.h5'))
    template_1c_nobkg = make_background_free_template(template_1c)

    settings = {
        '2 Gauss + Bkg (MPFit)': {'tmplt': template_2c, 'method': 'mpfit'},
        '1 Gauss + Bkg (MPFit)': {'tmplt': template_1c, 'method': 'mpfit'},
        '1 Gauss (MPFit)': {'tmplt': template_1c_nobkg, 'method': 'mpfit'},
        '1 Gauss (PMF)': {'method': 'pmf'}
    }

    results_rmse = {name: np.zeros((num_images, 3)) for name in settings}
    results_bias = {name: np.zeros((num_images, 3)) for name in settings}
    new_param3ds = np.zeros_like(param3ds)

    wave = WAVELENGTH + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)

    sample_image = 0
    sample_y, sample_x = np.unravel_index(np.argmax(np.sum(datacubes[sample_image], axis=-1)), datacubes[sample_image].shape[:2])
    cube_phys_sample = (datacubes[sample_image] / DISP_SCALE_A).transpose(2, 0, 1)
    spectrum_sample = cube_phys_sample[:, sample_y, sample_x]
    param_sample = param3ds[sample_image][:, sample_y, sample_x]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, config) in enumerate(settings.items()):
        ax = axes[idx]
        try:
            if config['method'] == 'mpfit':
                params_sample, wave_dense_sample, combined_sample, gauss_comps_sample, background_sample = fit_pixel_spectrum(
                    wave, spectrum_sample, config['tmplt']
                )
                int_fit = np.sqrt(2 * np.pi) * params_sample[0] * params_sample[2]
                vel_fit = SPEED_OF_LIGHT * (params_sample[1] - WAVELENGTH) / WAVELENGTH
                wid_fit_kms = params_sample[2] * SPEED_OF_LIGHT / WAVELENGTH
            else:
                recon_pmf = fit_pmf(cube_phys_sample)
                pmf_params = recon_pmf[:, sample_y, sample_x]
                wave_dense_sample = np.linspace(wave.min(), wave.max(), 400)
                combined_sample = ideal_param3d_curve(wave_dense_sample, pmf_params[0], pmf_params[1], pmf_params[2])
                gauss_comps_sample = [combined_sample]
                background_sample = np.zeros_like(wave_dense_sample)
                
                int_fit = pmf_params[0]
                vel_fit = pmf_params[1]
                wid_fit_kms = pmf_params[2] * SPEED_OF_LIGHT / WAVELENGTH

            text_str = (
                f"Ideal param3d:\n"
                f" Int: {param_sample[0]:7.1f}\n"
                f" Vel: {param_sample[1]:7.1f} km/s\n"
                f" Wid: {param_sample[2]*SPEED_OF_LIGHT/WAVELENGTH:7.1f} km/s\n"
                f"Gaussian 1:\n"
                f" Int: {int_fit:7.1f}\n"
                f" Vel: {vel_fit:7.1f} km/s\n"
                f" Wid: {wid_fit_kms:7.1f} km/s"
            )

            ideal_curve_sample = ideal_param3d_curve(
                wave_dense_sample, param_sample[0], param_sample[1], param_sample[2]
            )
            plot_spectrum_fit(
                ax, wave, spectrum_sample, wave_dense_sample, 
                combined_sample, gauss_comps_sample, background_sample, 
                ideal_curve_sample, coords=(sample_y, sample_x),
                method_name=name, text_str=text_str
            )
        except Exception as e:
            print(f"Warning: failed to produce sample fit plot for {name}: {e}")

    plt.tight_layout()
    plt.savefig('spectrum_fit_summary_2x2.png', dpi=200, bbox_inches='tight')
    print("Saved 2x2 summary figure to spectrum_fit_summary_2x2.png")
    plt.show()

    for i in range(num_images):
        print(f"\n--- Processing image {i+1}/{num_images} ---")
        cube = datacubes[i]
        true_param = param3ds[i]
        
        # Stored cube is scaled: divide by DISP_SCALE to get flux density for fitting.
        cube_phys = (cube / DISP_SCALE_A).transpose(2, 0, 1)

        for name, config in settings.items():
            print(f"  Fitting with {name}...")
            if config['method'] == 'mpfit':
                recon = smart_fit_spectra_joblib(
                    cube_phys,
                    config['tmplt'],
                    wave=wave,
                    n_jobs=-1,
                    component=0
                )
            else:
                recon = fit_pmf(cube_phys)

            if name == '2 Gauss + Bkg (MPFit)':
                new_param3ds[i] = recon

            rmse, bias = get_metrics(true_param, recon)
            results_rmse[name][i] = rmse
            results_bias[name][i] = bias

    print("\n" + "=" * 73)
    print("                         EVALUATION SUMMARY")
    print("=" * 73)

    for name in settings:
        print(f"\nMethod: {name}")
        print("-" * 73)
        for i in range(num_images):
            rmse = results_rmse[name][i]
            bias = results_bias[name][i]
            # Convert width from Angstroms to km/s
            rmse_width_kms = rmse[2] * SPEED_OF_LIGHT / WAVELENGTH
            bias_width_kms = bias[2] * SPEED_OF_LIGHT / WAVELENGTH
            print(
                f" Img {i+1}: Int RMSE={rmse[0]:8.2f}, Bias={bias[0]:8.2f} | "
                f"Vel RMSE={rmse[1]:6.3f} km/s, Bias={bias[1]:7.3f} km/s | "
                f"Wid RMSE={rmse_width_kms:6.3f} km/s, Bias={bias_width_kms:7.3f} km/s"
            )

        avg_rmse = np.mean(results_rmse[name], axis=0)
        avg_bias = np.mean(results_bias[name], axis=0)
        avg_rmse_width_kms = avg_rmse[2] * SPEED_OF_LIGHT / WAVELENGTH
        avg_bias_width_kms = avg_bias[2] * SPEED_OF_LIGHT / WAVELENGTH
        print("-" * 73)
        print(
            f" AVG  : Int RMSE={avg_rmse[0]:8.2f}, Bias={avg_bias[0]:8.2f} | "
            f"Vel RMSE={avg_rmse[1]:6.3f} km/s, Bias={avg_bias[1]:7.3f} km/s | "
            f"Wid RMSE={avg_rmse_width_kms:6.3f} km/s, Bias={avg_bias_width_kms:7.3f} km/s"
        )

    print("\n" + "=" * 73)
    out_path = os.path.join(path_data, 'eis_train_5_dsetv5.npy')
    print(f"Saving updated ground truth dataset to {out_path}...")
    data_v5 = data.copy()
    data_v5['param3d'] = new_param3ds
    np.save(out_path, data_v5)
    print("Done!")


if __name__ == '__main__':
    main()

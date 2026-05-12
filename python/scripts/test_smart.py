import numpy as np
from slitless.forward import Source, Imager, gauss_pix, datacube_generator, tomomtx_gen
from scipy.ndimage import convolve
from slitless.recon import gauss_pmf_fitter, smart_fit_spectra_joblib
import matplotlib.pyplot as plt
import eispac

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data='eis_train_5_dsetv4.npy' # 64x64 EIS images
data = np.load(path_data+data, allow_pickle=True).item()
# param4dar, meas4dar = data['param3d'], data['meas']
param3dar, meas3dar = data['param3d'][4], data['meas'][4]
source_pix = False
intenscale = meas3dar.max()

numdetectors = 3
dbsnr = 10
noise_model=None # Noise-free measurements
# noise_model='poisson'
# noise_model='gaussian'

Imgr = Imager(pixelated=True, dbsnr=dbsnr, avg_count=dbsnr**2, noise_model=noise_model, 
# Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9, noise_model=noise_model, 
    spectral_orders=[0,-1,1,-2,2][:numdetectors])

Imgr.meas3dar_nn = meas3dar[:numdetectors] / intenscale
Imgr.meas3dar = meas3dar[:numdetectors] / intenscale

imager=Imgr
psi=0.2
maxouter=5
maxinner=20
inf_prior_width=1.38
fitter='pmf'
tmplt=None
n_jobs=-1

if imager is not None:
    meas = imager.meas3dar.copy()
    NK, M, N = meas.shape
    L = 21

    orders = imager.spectral_orders
    inf = True if inf_prior_width is not None else False
    orders_list = orders + ['inf'] if inf else orders

    meas_list = [meas[k] for k in range(NK)]
    if inf:
        infprior = gauss_pix(np.outer(np.arange(L), np.ones(N)), L//2, inf_prior_width)
        infprior = infprior / infprior.sum(axis=0) * meas[0].sum(axis=0)
        meas_list.append(infprior)

    num_projs = len(meas_list)

    mtx_list = []
    for order in orders_list:
        mtx_list.append(tomomtx_gen((L, M), orders=[order]))

    mtx_s_list = []
    for k in range(num_projs):
        mapped = mtx_list[k].T @ np.ones((mtx_list[k].shape[0], 1))
        mtx_s_list.append((mapped < 0.01).flatten())

    int0 = meas[0].copy()
    
    if imager is not None:
        rest_wave = imager.srpix.rest_wavelength if hasattr(imager, 'srpix') else 195.117937907451
        mid_wave = imager.mid_wavelength
        disp_scale = imager.dispersion_scale
    else:
        rest_wave = 195.117937907451
        mid_wave = 195.119
        disp_scale = 0.022275
        
    vel_pix_0 = (rest_wave - mid_wave) / disp_scale
    width_pix_0 = 0.02888811 / disp_scale
    
    vel0 = vel_pix_0 * np.ones_like(int0)
    width0 = width_pix_0 * np.ones_like(int0)
    cube = datacube_generator(np.stack((int0,vel0,width0),axis=0), lamdim=L)
    
    k0 = np.array([0.25, 0.5, 0.25])
    k1 = np.outer(k0,k0)
    kernel = k0[None,None] * k1[:,:,None]

    for i in range(maxouter):
        print('Outer Iter: {}/{}'.format(i+1,maxouter))
        cube = (cube + cube**(1+psi))*np.sum(cube)/np.sum(cube + cube**(1+psi))
        cube = convolve(cube, kernel)
        for j in range(maxinner):
            cube_flat = cube.reshape(L*M, N)
            
            meas2_list = []
            for k in range(num_projs):
                meas2_list.append(mtx_list[k] @ cube_flat)
                
            chi_list = []
            for k in range(num_projs):
                chi_list.append(np.mean(((meas_list[k]-meas2_list[k])**2)/(meas_list[k]+1e-7)))
            chi = np.array(chi_list)
            unconverged = chi > 1e-10
            if np.sum(unconverged) == 0:
                continue
            
            Cor_list = []
            for k in range(num_projs):
                cor_k = (meas_list[k]/(meas2_list[k]+1e-5))**(2/3)
                Cor_k_flat = mtx_list[k].T @ cor_k
                Cor_k_flat[mtx_s_list[k], :] = 1.0
                Cor_list.append(Cor_k_flat.reshape(L, M, N))
            
            Cor = np.stack(Cor_list, axis=0)
            Corr = np.prod(Cor[unconverged], axis=0)**(1/np.sum(unconverged))
            cube *= Corr
        print(f'chi:{np.mean(chi)}')
    # cors.append(Cor)
    # cubes.append(cube)

# --- DIAGNOSTIC EVALUATION ---
print("\n" + "="*60)
print("            DIAGNOSTIC EVALUATION")
print("="*60)

def get_metrics(true, est):
    rmse = np.sqrt(np.mean((true - est) ** 2, axis=(-1, -2)))
    bias = np.mean(est - true, axis=(-1, -2))
    return rmse, bias

lamdim = 21
DISP_SCALE_A = 0.022275
WAVELENGTH_CENTER = 195.119
SPEED_OF_LIGHT = 299792.458
template_filepath = '/home/kamo/resources/slitless/data/eis_data/templates/fe_12_195_119.2c.template.h5'
REST_WAVELENGTH = eispac.read_template(template_filepath).parinfo[1]['value']

wave = WAVELENGTH_CENTER + DISP_SCALE_A * (np.arange(lamdim) - lamdim // 2)

# Convert tomographic cube to physical flux density
cube_phys = cube * intenscale / DISP_SCALE_A
# Load true datacube and convert to physical flux density
true_cube_phys = data['datacube'][4].transpose(2,0,1) / DISP_SCALE_A

# 1. MPFit
print("Fitting reconstructed cube with MPFit (2 Gauss + Bkg)...")
tmplt = eispac.read_template(template_filepath)
recon_mpfit = smart_fit_spectra_joblib(cube_phys, tmplt, wave=wave, n_jobs=n_jobs, component=0)

# 2. PMF
print("Fitting reconstructed cube with PMF (with background subtraction)...")
bg = np.min(cube_phys, axis=0, keepdims=True)
cube_safe = np.clip(cube_phys - bg, 0.0, None)
recon_pmf_pix = gauss_pmf_fitter(cube_safe)

recon_pmf = np.zeros_like(recon_pmf_pix)
recon_pmf[0] = recon_pmf_pix[0] * DISP_SCALE_A
recon_pmf[1] = recon_pmf_pix[1] * DISP_SCALE_A * SPEED_OF_LIGHT / REST_WAVELENGTH
recon_pmf[2] = recon_pmf_pix[2] * DISP_SCALE_A

# Print Metrics
def print_metrics(name, est):
    rmse, bias = get_metrics(param3dar, est)
    rmse_w_kms = rmse[2] * SPEED_OF_LIGHT / REST_WAVELENGTH
    bias_w_kms = bias[2] * SPEED_OF_LIGHT / REST_WAVELENGTH
    print(f"--- {name} ---")
    print(f" Int RMSE={rmse[0]:8.2f}, Bias={bias[0]:8.2f}")
    print(f" Vel RMSE={rmse[1]:8.3f} km/s, Bias={bias[1]:8.3f} km/s")
    print(f" Wid RMSE={rmse_w_kms:8.3f} km/s, Bias={bias_w_kms:8.3f} km/s")

print_metrics("MPFit", recon_mpfit)
print_metrics("PMF", recon_pmf)

# Plotting 1: 1D Spectrum Comparison
y_max, x_max = np.unravel_index(np.argmax(param3dar[0]), param3dar[0].shape)

def gaussian_eval(wave_arr, param, rest_wave):
    center = rest_wave * (1 + param[1] / SPEED_OF_LIGHT)
    peak = param[0] / (np.sqrt(2 * np.pi) * param[2])
    return peak * np.exp(-0.5 * ((wave_arr - center) / param[2])**2)
    
plt.figure(figsize=(10, 6))
plt.plot(wave, true_cube_phys[:, y_max, x_max], 'k.-', label='True Datacube', linewidth=2)
plt.plot(wave, cube_phys[:, y_max, x_max], 'g.-', label='Reconstructed Datacube (Tomography)', linewidth=2)

plt.plot(wave, gaussian_eval(wave, recon_mpfit[:, y_max, x_max], rest_wave=REST_WAVELENGTH), 'r--', label='MPFit Primary Gaussian', linewidth=2)
plt.plot(wave, gaussian_eval(wave, recon_pmf[:, y_max, x_max], rest_wave=REST_WAVELENGTH), 'b:', label='PMF Extracted Gaussian', linewidth=2)

plt.title(f'1D Spectrum Comparison at Brightest Pixel (y={y_max}, x={x_max})')
plt.xlabel('Wavelength [Å]')
plt.ylabel('Flux Density')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('test_smart_1d_spectrum.png', dpi=200, bbox_inches='tight')
plt.show()

# Plotting 2: 2D Parameter Maps
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
titles = ['Intensity', 'Velocity', 'Line Width']
cmaps = ['hot', 'seismic', 'plasma']

vmaxs = [param3dar[0].max(), 10, param3dar[2].max()]
vmins = [0, -10, param3dar[2].min()]

for i in range(3):
    # Truth
    im = axes[0, i].imshow(param3dar[i], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
    axes[0, i].set_title(f'True {titles[i]}')
    fig.colorbar(im, ax=axes[0, i], fraction=0.8)
    
    # MPFit
    im = axes[1, i].imshow(recon_mpfit[i], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
    axes[1, i].set_title(f'MPFit {titles[i]}')
    fig.colorbar(im, ax=axes[1, i], fraction=0.8)
    
    # PMF
    im = axes[2, i].imshow(recon_pmf[i], cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
    axes[2, i].set_title(f'PMF {titles[i]}')
    fig.colorbar(im, ax=axes[2, i], fraction=0.8)
    
plt.tight_layout()
plt.savefig('test_smart_2d_maps.png', dpi=200, bbox_inches='tight')
plt.show()
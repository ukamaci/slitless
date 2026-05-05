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
intenscale = param3dar[0].max()

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
if inf_prior_width is not None:
    infprior = gauss_pix(np.outer(np.arange(M),np.ones(M)), M//2, inf_prior_width)
    meas = np.concatenate((meas, infprior[None]), axis=0)
    meas[-1]*=meas[0].mean(axis=0)[None]/meas[-1].mean(axis=0)[None]
    NK += 1 
    
orders = imager.spectral_orders
inf=True if inf_prior_width is not None else False
orders = orders+['inf'] if inf else orders
#initialize
cubes = []
cors = []
int0 = meas[0].copy()
vel0 = np.zeros_like(int0)
width0 = 1.38*np.ones_like(vel0)
cube = datacube_generator(np.stack((int0,vel0,width0),axis=0))
cubes.append(cube)
k0 = np.array([0.25, 0.5, 0.25])
k1 = np.outer(k0,k0)
kernel = k0[None,None] * k1[:,:,None]

mtx = tomomtx_gen((M,M), orders=orders)
mtx_t = np.einsum('ijk->ikj', mtx.reshape(-1,M,M*M))
mtx_s = (np.sum(mtx_t,axis=2)<0.01).astype(int).reshape(-1,M,M)[:,:,:,None]
mtx_s = np.repeat(mtx_s, N, axis=3)

for i in range(maxouter):
    print('Outer Iter: {}/{}'.format(i+1,maxouter))
    # contrast enhancement
    cube = (cube + cube**(1+psi))*np.sum(cube)/np.sum(cube + cube**(1+psi))
    # kernel smoothing
    cube = convolve(cube, kernel)
    for j in range(maxinner):
        # print('Inner Iter: {}/{}'.format(j+1,maxinner))
        # meas2 = forward_op_tomo_3d(cube, orders=orders, inf=inf)
        meas2 = (mtx @ cube.reshape(-1,N)).reshape(meas.shape)
        # chi-square
        chi = np.mean(((meas-meas2)**2)/(meas+1e-7), axis=(1,2))
        unconverged = chi>0.0000000001
        # if all are converged, go to the next iter
        if np.sum(unconverged)==0:
            continue
        # cor = (meas/(meas2+1e-5))**(2/(3+1*(inf==True)))
        cor = (meas/(meas2+1e-5))**(2/(3))
        # cor = (meas2/meas)**2/3 # Warning!: reversed order in ESIS2022
        # Cor = forward_op_tomo_3d_transpose(cor, orders=orders, inf=inf)
        Cor = np.einsum('ijk,ikm->ijm',mtx_t, cor).reshape(NK,M,M,N)
        # Cor[mtx_s==1]=1

        mtx_s_u, mtx_s_l = mtx_s.copy(), mtx_s.copy()
        mtx_s_u[:,M//2:]=0
        mtx_s_l[:,:M//2]=0
        tails = Cor[:,M//2,[0,-1],:] #(NK,2,N)
        ind_ul = mtx_s[:,0,M//2-1][:,None,None] #(NK,1,1,N) 
        mtx_s_ul = mtx_s_u*ind_ul
        mtx_s_ur = mtx_s_u*(1-ind_ul)
        mtx_s_ll = mtx_s_l*(1-ind_ul)
        mtx_s_lr = mtx_s_l*ind_ul
        mtx_c_ul = mtx_s_ul * tails[:,[0],None]
        mtx_c_ur = mtx_s_ur * tails[:,[1],None]
        mtx_c_ll = mtx_s_ll * tails[:,[0],None]
        mtx_c_lr = mtx_s_lr * tails[:,[1],None]
        Cor[mtx_s_ul==1] = mtx_c_ul[mtx_s_ul==1]
        Cor[mtx_s_ur==1] = mtx_c_ur[mtx_s_ur==1]
        Cor[mtx_s_ll==1] = mtx_c_ll[mtx_s_ll==1]
        Cor[mtx_s_lr==1] = mtx_c_lr[mtx_s_lr==1]

        Corr = np.prod(Cor[unconverged],axis=0)**(1/np.sum(unconverged))
        cube *= Corr
    print(f'chi:{chi}')
    # cors.append(Cor)
    # cubes.append(cube)

if fitter=='mpfit':
    if tmplt is None:
        template_filepath = '/home/kamo/resources/slitless/data/eis_data/templates/fe_12_195_119.2c.template.h5'
        tmplt = eispac.read_template(template_filepath)
    lamdim = cube.shape[0]
    # Construct the wavelength grid assuming pixel units center around lamdim//2
    wave = imager.srpix.wavelength + (imager.dispersion_scale / 1000) * (np.arange(lamdim) - lamdim // 2)
    cube = cube / (imager.dispersion_scale / 1000) * imager.intenscale
    
    recon = smart_fit_spectra_joblib(cube, tmplt, wave=wave, n_jobs=n_jobs)
    
    # convert physical units (km/s, Angstroms) back to pixel units to match original format
    recon[0] /= imager.intenscale
    recon[1] = recon[1] * (imager.srpix.wavelength / 300 / imager.dispersion_scale)
    recon[2] = recon[2] / (imager.dispersion_scale / 1000)
elif fitter=='pmf':
    recon = gauss_pmf_fitter(cube)

Source(param3d=recon).plot()

plt.figure()
plt.imshow(cube.mean(axis=2), cmap='hot')
plt.colorbar()
plt.show()
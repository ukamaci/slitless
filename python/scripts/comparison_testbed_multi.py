from slitless.forward import Source, Imager, forward_op, datacube_generator
from slitless.plotting import uiuc_im
from slitless.recon import (smart, grad_descent_solver, scipy_solver, tomoinv,
    Reconstructor, Reconstructor_Multi, nn_solver, diffusion_solver)
import numpy as np
import datetime, pickle, shutil, os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
# data='apj19_20x20.npy' # 20x20 APJ2019 image
# data='eis_5_64x64.npy' # 64x64 EIS images
data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images

# param4dar = np.load(path_data+data)
param4dar = uiuc_im()
if len(param4dar.shape)<4:
    param4dar = param4dar[np.newaxis]

savepath = '/home/kamo/resources/slitless/python/results/recons/'
save = False
M = param4dar.shape[-1]
numdetectors = 3
dbsnr = 50
noise_model='poisson'
# noise_model='gaussian'

def inf_priorer(param4dar):
    means = []
    for i in range(len(param4dar)):
        means.append(datacube_generator(param4dar[i]).mean(axis=(1,2)))
    return np.array(means).mean(axis=0)
    
mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
mask= np.ones_like(mask)
# print('Mask1')
Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9, noise_model=noise_model, 
# Imgr = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, noise_model='gaussian', 
    spectral_orders=[0,-1,1,-2,2][:numdetectors])

# sr = Source(param3d=param4dar[0],pix=True)
# m=Imgr.get_measurements(sources=sr)
# Imgr.plot('poisson_np: {}'.format(poisson_sn))

# # SCIPY
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=scipy_solver,
#     DATA_FIDELITY='L2',
#     OPTIMIZER='L-BFGS-B',
#     # OPTIMIZER='Nelder-Mead',
#     maxiter=10000,
#     lam_i=1e0,
#     lam_v=0.5,
#     lam_w=30
# )

# Tomoinv
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=tomoinv,
#     lam=1e-2,
#     stepsize=3e-2,
#     numiter=100,
#     proj='positivity',
#     data_step='grad',
#     positivity=True
# )

# # 2D GD
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=grad_descent_solver,
#     maxiter=1000,
#     lam_i=1e-8,
#     # lam_v=0.0001,
#     # lam_w=4.6e-3,
#     lam_v=0.000278,
#     lam_w=6e-4,
#     LR=5e-2
# )

# # U-Net
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=nn_solver,
#     model_path='dbsnr_50_poisson_K_3_dssize_full'
#     # model_path='2024_08_25__10_49_50_NF_64_BS_4_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_15_poisson_K_3_dssize_full'
# )

# Diffusion DPS
Rec = Reconstructor_Multi(
    imager=Imgr,
    param4dar=param4dar,
    pix=True,
    solver=diffusion_solver,
    model_path='model-10.pt',
    grad_scale=[1,1,1],
    num_samples=10
)

# # SMART
# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=smart,
#     psi=0.2,
#     maxouter=5,
#     maxinner=20,
#     inf_prior_width=1.38
#     # inf_prior_width=None
# )

recons = Rec.solve(num_realizations=1)
# Rec.recons[0].plot_loss()
fig, ax = recons[0].plot(compare=True, title=f'{Rec.solver.__name__}')
print('mask: {}'.format(mask[:2,:2]))
print('Solver: {}'.format(Rec.solver.__name__))
print('Solver Params: {}'.format(Rec.solver_params))
print('Recon Time Avg: {:.2f} s'.format(Rec.times.mean()))
print('RMSE_phy Avg (per Img): {}'.format(Rec.rmse_phy.mean(axis=1)))
print('RMSE_phy Avg: {}'.format(Rec.rmse_phy.mean(axis=(0,1))))
print('MAE_phy Avg: {}'.format(Rec.mae_phy.mean(axis=(0,1))))
print('Bias_phy Avg (per Img): {}'.format(Rec.bias_phy.mean(axis=1)))
print('Bias_phy Avg: {}'.format(Rec.bias_phy.mean(axis=(0,1))))
print('RMSE_pix Avg (per Img): {}'.format(Rec.rmse_pix.mean(axis=1)))
print('RMSE_pix Avg: {}'.format(Rec.rmse_pix.mean(axis=(0,1))))
# print('SSIMs: {}'.format(recons.ssim))
# print('SSIM Avg: {}'.format(recons.ssim.mean(axis=0)))

if save==True:
    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = (f'{now}_{Rec.solver.__name__}_{data[:-4]}_K_{numdetectors}_{noise_model}_dbsnr_{dbsnr}')
    savedir = savepath+name
    os.mkdir(savedir)
    recon_summary = [
    '############## Recon Parameters ############## \n',
    'mask: {} \n'.format(mask[:2,:2]),
    'Solver: {} \n'.format(Rec.solver.__name__),
    'Num Detectors: {} \n'.format(numdetectors),
    'Noise Model / dbsnr: {} / {} \n'.format(noise_model, dbsnr),
    'Num Realizations: {} \n'.format(Rec.num_realizations),
    'Solver Params: {} \n'.format(Rec.solver_params),
    'Recon Time Avg: {:.2f} s \n'.format(Rec.times.mean()),
    'RMSE_phy Avg (per Img): {} \n'.format(Rec.rmse_phy.mean(axis=1)),
    'RMSE_phy Avg: {} \n'.format(Rec.rmse_phy.mean(axis=(0,1))),
    'MAE_phy Avg: {} \n'.format(Rec.mae_phy.mean(axis=(0,1))),
    'Bias_phy Avg (per Img): {} \n'.format(Rec.bias_phy.mean(axis=1)),
    'Bias_phy Avg: {} \n'.format(Rec.bias_phy.mean(axis=(0,1))),
    'RMSE_pix Avg (per Img): {} \n'.format(Rec.rmse_pix.mean(axis=1)),
    'RMSE_pix Avg: {} \n'.format(Rec.rmse_pix.mean(axis=(0,1)))
    ]

    rec_array_pix = []
    rec_array_phy = []
    truth_array_pix = []
    truth_array_phy = []
    for i in range(len(Rec.recons)):
        fig, ax = recons[i].plot(compare=True, title=f'{Rec.solver.__name__}')
        fig.savefig(savedir+f'/recon_{i}.png')
        rec_array_pix.append(Rec.recons[i].recon)
        truth_array_pix.append(Rec.sources[i].param3d[None])
    rec_array_pix = np.array(rec_array_pix)
    truth_array_pix = np.array(truth_array_pix)
    rec_array_phy = Rec.imager.frompix(rec_array_pix, width_unit='km/s', array=True)
    truth_array_phy = Rec.imager.frompix(truth_array_pix, width_unit='km/s', array=True)
    diff_pix = rec_array_pix - truth_array_pix
    diff_phy = rec_array_phy - truth_array_phy

    def hist_plotter(diff, unit='phy'):
        unitstr = 'km/s' if unit=='phy' else 'pixels'
        fig, ax = plt.subplots(1,3, figsize=(13.8,4.8))
        ax[0].hist(diff[:,:,0].flatten(), bins=20, edgecolor='Black', color='sandybrown')
        ax[0].set_title(('Intensity RMS Error = {:.4f} \n '.format(np.sqrt(np.mean(diff[:,:,0]**2))) +
        r'Intensity Bias = {:.4f}'.format(np.mean(diff[:,:,0]))))
        ax[0].set_xlabel('Error')
        ax[0].set_ylabel('Number of Occurances')
        ax[1].hist(diff[:,:,1].flatten(), bins=20, edgecolor='Black', color='sandybrown')
        ax[1].set_title(('Doppler Velocity RMS Error = {:.3f} {} \n'.format(np.sqrt(np.mean(diff[:,:,1]**2)),unitstr)+
        'Doppler Velocity Bias = {:.3f} {}'.format(np.mean(diff[:,:,1]),unitstr)))
        ax[1].set_xlabel(f'Error [{unitstr}]')
        ax[1].set_ylabel('Number of Occurances')
        ax[2].hist(diff[:,:,2].flatten(), bins=20, edgecolor='Black', color='sandybrown')
        ax[2].set_title(('Line Width RMS Error = {:.3f} {} \n'.format(np.sqrt(np.mean(diff[:,:,2]**2)), unitstr)+
        'Line Width Bias = {:.3f} {}'.format(np.mean(diff[:,:,2]), unitstr)))
        ax[2].set_xlabel(f'Error [{unitstr}]')
        ax[2].set_ylabel('Number of Occurances')
        plt.tight_layout()
        plt.show()
        return fig, ax
    fig, ax = hist_plotter(diff_phy, 'phy')
    fig.savefig(savedir+'/error_hist_phy.png')
    fig, ax = hist_plotter(diff_pix, 'pix')
    fig.savefig(savedir+'/error_hist_pix.png')

    with open(f'{savedir}/summary.txt', 'w') as file:
        for line in recon_summary:
            file.write(line)

    with open(f'{savedir}/Rec.pickle', 'wb') as file:
        pickle.dump(Rec, file)

# from slitless.recon import comparison_test_multi
# path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
# data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images
# savepath = '/home/kamo/resources/slitless/python/results/recons/'
# Rec_result = comparison_test_multi(path_data, data, savepath, single_param4dar=True, save=False, numdetectors=3, dbsnr=50, 
#                           noise_model='poisson', solver='diffusion', model_path='model-10.pt', grad_scale=[1,1,1], num_samples=10)
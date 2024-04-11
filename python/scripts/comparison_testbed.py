from slitless.forward import Source, Imager, forward_op
from slitless.recon import grad_descent_solver, scipy_solver, Reconstructor
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

path_data = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v1/meta/selected_scans_train/'
date='20071211_002416' #APJ2019 image
# Rotate the params so that the effective dispersion direction is horizontal
inten=np.rot90(np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202])
inten /= inten.max()
Sr = Source(
    inten=inten,
    # inten=np.rot90(np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202]),
    vel=np.rot90(np.load(path_data+'vel_{}.npy'.format(date))[149:169, 182:202]),
    width=np.rot90(np.load(path_data+'width_{}.npy'.format(date))[149:169, 182:202]),
    pix=False
)
mask = np.array([[(i + j) % 2 for j in range(20)] for i in range(20)])
# mask= np.ones_like(mask)
# print('Mask1')
Imgr = Imager(pixelated=True, mask=mask)
Imgr.topix(Sr)
inten, vel, width = Imgr.srpix.param3d

meas = Imgr.get_measurements(max_count=50**2/0.9, noise_model='poisson',no_noise=False)

# Rec_gd = Reconstructor(
#     imager=Imgr,
#     solver=grad_descent_solver,
#     maxiter=1000,
#     lam_i=1e-8,
#     lam_v=10**-3.25,
#     lam_w=10**-3.25,
#     LR=5e-2
# )

# recons_gd = Rec_gd.solve(num_realizations=5)
# recons_gd.plot(compare=True, title='GD')
# print('Solver: {}'.format(Rec_gd.solver.__name__))
# print('Solver Params: {}'.format(Rec_gd.solver_params))
# print('mask: {}'.format(mask[:2,:2]))
# print('RMSEs (GD): {}'.format(recons_gd.rmse_phy))
# print('RMSE Avg (GD): {}'.format(recons_gd.rmse_phy.mean(axis=0)))

Rec_sci = Reconstructor(
    imager=Imgr,
    solver=scipy_solver,
    DATA_FIDELITY='L2',
    OPTIMIZER='L-BFGS-B',
    maxiter=10000,
    lam_i=5e-3,
    lam_v=5e-3,
    lam_w=5e-2
)

recons_lbfgs = Rec_sci.solve(num_realizations=1)
recons_lbfgs.plot(compare=True, title='LBFGS')
print('mask: {}'.format(mask[:2,:2]))
print('Solver: {}'.format(Rec_sci.solver.__name__))
print('Solver Params: {}'.format(Rec_sci.solver_params))
print('RMSEs (L-BFGS): {}'.format(recons_lbfgs.rmse_phy))
print('RMSE Avg (L-BFGS): {}'.format(recons_lbfgs.rmse_phy.mean(axis=0)))
# print('SSIMs: {}'.format(recons_lbfgs.ssim))
# print('SSIM Avg: {}'.format(recons_lbfgs.ssim.mean(axis=0)))

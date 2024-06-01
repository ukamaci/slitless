from slitless.forward import Source, Imager, forward_op
from slitless.recon import grad_descent_solver, scipy_solver, Reconstructor, Reconstructor_Multi, nn_solver
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data='apj19_20x20.npy' # 20x20 APJ2019 image
# data='eis_5_64x64.npy' # 20x20 APJ2019 image

param4dar = np.load(path_data+data)
if len(param4dar.shape)<4:
    param4dar = param4dar[np.newaxis]

M = param4dar.shape[-1]

mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
mask= np.ones_like(mask)
# print('Mask1')
Imgr = Imager(pixelated=True, mask=mask, max_count=50**2/0.9, noise_model='poisson')

# Rec = Reconstructor_Multi(
#     imager=Imgr,
#     param4dar=param4dar,
#     pix=True,
#     solver=scipy_solver,
#     DATA_FIDELITY='L2',
#     # OPTIMIZER='L-BFGS-B',
#     OPTIMIZER='Nelder-Mead',
#     maxiter=10000,
#     lam_i=1e-4,
#     lam_v=0.01,
#     lam_w=0.1
# )

Rec = Reconstructor_Multi(
    imager=Imgr,
    param4dar=param4dar,
    pix=True,
    solver=grad_descent_solver,
    maxiter=1000,
    lam_i=1e-8,
    lam_v=0.001,
    lam_w=4.6e-3,
    # lam_v=0.000278,
    # lam_w=6e-4,
    LR=5e-2
)

recons_lbfgs = Rec.solve(num_realizations=2)
recons_lbfgs[0].plot(compare=True, title='GD')
print('mask: {}'.format(mask[:2,:2]))
print('Solver: {}'.format(Rec.solver.__name__))
print('Solver Params: {}'.format(Rec.solver_params))
print('Recon Time Avg: {:.2f} s'.format(Rec.times.mean()))
print('RMSE_phy Avg (per Img): {}'.format(Rec.rmse_phy.mean(axis=1)))
print('RMSE_phy Avg: {}'.format(Rec.rmse_phy.mean(axis=(0,1))))
print('Bias_phy Avg (per Img): {}'.format(Rec.bias_phy.mean(axis=1)))
print('Bias_phy Avg: {}'.format(Rec.bias_phy.mean(axis=(0,1))))
print('RMSE_pix Avg (per Img): {}'.format(Rec.rmse_pix.mean(axis=1)))
print('RMSE_pix Avg: {}'.format(Rec.rmse_pix.mean(axis=(0,1))))
# print('SSIMs: {}'.format(recons_lbfgs.ssim))
# print('SSIM Avg: {}'.format(recons_lbfgs.ssim.mean(axis=0)))

# Rec = Reconstructor(
#     imager=Imgr,
#     solver=nn_solver
# )

# recons_nn = Rec.solve(num_realizations=10)
# recons_nn.plot(compare=True, title='NN')
# print('mask: {}'.format(mask[:2,:2]))
# print('Solver: {}'.format(Rec.solver.__name__))
# print('Solver Params: {}'.format(Rec.solver_params))
# print('RMSEs: {}'.format(recons_nn.rmse_phy))
# print('RMSE Avg: {}'.format(recons_nn.rmse_phy.mean(axis=0)))
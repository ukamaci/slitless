# Ulas Kamaci
# 2022-10-20
import torch, glob, expsweep
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from slitless.data_loader import BasicDataset, OntheflyDataset
from slitless.forward import Source, Imager, forward_op_torch, add_noise
from slitless.measure import cycle_loss, compare_ssim, compare_psnr, nrmse, tv_loss, grad_res_loss
from tqdm.auto import tqdm
from slitless.recon import grad_descent_solver

numim = 5
dataset_path = glob.glob('../../data/datasets/dset6*')[0]
temp_path = glob.glob('../../data/temp')[0]
trainset = OntheflyDataset(data_dir=dataset_path, fold='train', dbsnr=None)
trainloader = DataLoader(trainset, batch_size=numim, shuffle=True, num_workers=8)
data = next(iter(trainloader))
data = torch.stack((data[0], data[1]), axis=1).cpu().numpy()
np.save(temp_path+'data.npy', data)

def exp(*, dbsnr, lam_i, lam_v, lam_w, ind):
    data = np.load(temp_path+'data.npy')

    meas, truth = data[ind]
    meas = add_noise(meas, dbsnr=dbsnr, no_noise=dbsnr==None)

    x_int, x_vel, x_width, _,_,_ = grad_descent_solver(
        meas=meas, lam_i=lam_i, lam_v=lam_v, lam_w=lam_w,
        OPTIMIZER='ADAM', LR=1e-2, maxiters=10000, DATA_FIDELITY='L2')
    
    est = np.stack( ( x_int,
        x_vel,
        x_width ) )

    return{
        'ssim': compare_ssim(truth=truth, estimate=est),
        'rmse': nrmse(truth=truth, estimate=est)
    }


mc = expsweep.experiment(
    exp,
    lam_i=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    lam_v=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    lam_w=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    dbsnr=[15, 25, 35, None],
    ind=range(numim),
    cpu_count=1
)

if __name__ == '__main__':
# 2022-08-09 Ulas Kamaci
# Gradient descent for slitless using PyTorch Autograd
import torch, glob, copy, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from slitless.data_loader import BasicDataset
from slitless.forward import Source, Imager, forward_op_torch, add_noise
from slitless.measure import cycle_loss, compare_ssim, compare_psnr, nrmse, tv_loss, grad_res_loss
from slitless.evaluate import plot_recons_gd, stat_plotter, joint_plotter
from slitless.recon import grad_descent_solver
from tqdm.auto import tqdm

numim=10
# dataset_path = glob.glob('/home/kamo/resources/slitless/data/datasets/dset6*')[0]
dataset_path = glob.glob('/home/kamo/resources/slitless/data/eis_data/eistest64*')[0]
dataset = BasicDataset(data_dir = dataset_path, fold='test')
# numim=len(dataset)
dataloader = DataLoader(dataset, batch_size=numim, shuffle=False)
_, x = next(iter(dataloader))
pixelated = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = x.to(device=device, dtype=torch.float)
meas0 = forward_op_torch(x[:,0],x[:,1],x[:,2], pixelated=pixelated)

save = False
OPTIMIZER = 'Adam'
USE_TV_LOSS = True
DATA_FIDELITY = 'L2' # 'L1' or 'L2'
maxiters = 1000
dbsnr = [15]
mu_vel = 1e1 # GD step size for vel
mu_width = 1e1 # GD step size for width
lam_i = [10**-2] #, 10**-5, 10**-2] # TV norm regularization parameter for intensity
lam_v = [10**-3] #, 10**-3.5, 10**-3] # TV norm regularization parameter for velocity
lam_w = [10**-2] #, 10**-2.5, 10**-2] # TV norm regularization parameter for width
regparamdec = 0 # TV regu. param. exp.tial decay (set to 0 for no decay)
# beta = 10**(-regparamdec/maxiters) # TV reg. param. multiplier
# lammin = 1e-3
LR = 10**-2

for i in range(len(lam_v)):
    now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    name = (
        f'{now}_dbsnr_{dbsnr[i]}_BS_{numim}_EP_{maxiters}_LR_{LR}_' +
        f'{DATA_FIDELITY}_LOSS_TV_{OPTIMIZER}_lam_ivw_' +
        f'{np.log10(lam_i[i])}_{np.log10(lam_v[i])}_{np.log10(lam_w[i])}/'
    )
    if save:
        savedir = '../results/grad_descent/'+name
        os.mkdir(savedir)
    meas = add_noise(meas0, dbsnr=dbsnr[i], model='Gaussian')

    xh, losses, xhs, diffs_vel, diffs_width = grad_descent_solver(
        meas=meas,
        truth=x,
        OPTIMIZER='ADAM',
        USE_TV_LOSS=USE_TV_LOSS,
        DATA_FIDELITY=DATA_FIDELITY, # 'L1' or 'L2'
        lam_i=lam_i[i],
        lam_v=lam_v[i],
        lam_w=lam_w[i],
        LR=LR,
        maxiters=maxiters,
        return_arrays=True,
        savepath='../results/grad_descent/'+name
    )

    xx = x.cpu().numpy()
    ssims = compare_ssim(truth=xx, estimate=xh)
    # assume maxval for (int,vel,width)=(1,2,2.5)
    psnrs = compare_psnr(truth=xx, estimate=xh)
    rmses = nrmse(truth=xx, estimate=xh)

    ssims_m = ssims.mean(axis=0)
    rmses_m = rmses.mean(axis=0)

    ssims_s = ssims.std(axis=0)
    rmses_s = rmses.std(axis=0)

    print('SSIMs:  v: {:.2f}+/-{:.2f}  w: {:.2f}+/-{:.2f}'.format(
        ssims_m[1], ssims_s[1], ssims_m[2], ssims_s[2]))
    print('RMSEs:  v: {:.2f}+/-{:.2f}  w: {:.2f}+/-{:.2f}'.format(
        rmses_m[1], rmses_s[1], rmses_m[2], rmses_s[2]))

    # %% save
    if save:
        np.save(savedir+'recons.npy', xh)
        np.save(savedir+'truth.npy', xx)

        summary = [
        f'Dataset Path = {dataset_path} \n',
        f'Number of Images: {numim} \n',
        'SSIMs: i: {:.2f}+/-{:.2f}  v: {:.2f}+/-{:.2f}  w: {:.2f}+/-{:.2f} \n'.format(
        ssims_m[0], ssims_s[0], ssims_m[1], ssims_s[1], ssims_m[2], ssims_s[2]),
        'RMSEs: i: {:.2f}+/-{:.2f}  v: {:.2f}+/-{:.2f}  w: {:.2f}+/-{:.2f} \n'.format(
        rmses_m[0], rmses_s[0], rmses_m[1], rmses_s[1], rmses_m[2], rmses_s[2])
        ]

        with open(savedir+'summary.txt', 'w') as file:
            for line in summary:
                file.write(line)

        plot_recons_gd(meas=meas.cpu().numpy(), truth=xx, recon=xh, savedir=savedir+'recons/')

        stat_plotter(ssims, rmses, savedir)
        joint_plotter(truth=xx.transpose(1,0,2,3).reshape(3,-1), 
            recon=xh.transpose(1,0,2,3).reshape(3,-1), savedir=savedir)

    # %% loss
    plt.figure()
    plt.semilogy(losses)
    plt.title('Loss vs Iter')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.tight_layout()
    if save:
        plt.savefig(savedir+'loss.png')
    plt.show()

    plt.figure()
    plt.semilogy(diffs_vel)
    plt.title('||x_vel-xh_vel|| vs Iter')
    plt.xlabel('Iter')
    plt.ylabel('||x_vel-xh_vel||')
    plt.tight_layout()
    if save:
        plt.savefig(savedir+'x_vel_loss.png')
    plt.show()

    plt.figure()
    plt.semilogy(diffs_width)
    plt.title('||x_width-xh_width|| vs Iter')
    plt.xlabel('Iter')
    plt.ylabel('||x_width-xh_width||')
    plt.tight_layout()
    if save:
        plt.savefig(savedir+'x_width_loss.png')
    plt.show()

# %% recon
k=4
sr = Source(
    inten=x[k,0],
    vel=x[k,1],
    width=x[k,2],
    pix=True
)
sr2 = Source(
    inten=xh[k,0],
    vel=xh[k,1],
    width=xh[k,2],
    pix=True
)

sr.plot('Original')
plt.tight_layout()
sr2.plot('Estimated', ssims=ssims[k], rmses=rmses[k])
plt.tight_layout()
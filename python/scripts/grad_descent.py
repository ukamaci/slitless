# 2022-08-09 Ulas Kamaci
# Gradient descent for slitless using PyTorch Autograd
import torch, glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from slitless.data_loader import BasicDataset
from slitless.forward import Source, Imager, forward_op_torch
from slitless.measure import cycle_loss, compare_ssim, compare_psnr, tv_loss, grad_res_loss
from tqdm.auto import tqdm

dataset_path = glob.glob('/home/kamo/resources/slitless/data/datasets/dset3*')[0]
dataset = BasicDataset(data_dir = dataset_path, fold='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
_, x = next(iter(dataloader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sr = Source(
    inten=np.array(x[0,0]),
    vel=np.array(x[0,1]),
    width=np.array(x[0,2]),
    pix=True # the input arrays are given in pixel units
)

pixelated = True

imgr = Imager(pixelated=pixelated)
imgr.get_measurements(sr)

USE_OPTIMIZER = True
USE_TV_LOSS = True
GRAD_RES_LOSS = False
DATA_FIDELITY = 'L2' # 'L1' or 'L2'
maxiters = 20000
mu_vel = 1e1 # GD step size for vel
mu_width = 1e1 # GD step size for width
lam_v = 1e-2 # TV norm regularization parameter for velocity
lam_w = 1e-1 # TV norm regularization parameter for width
regparamdec = 6 # TV regu. param. exp.tial decay (set to 0 for no decay)
beta = 10**(-regparamdec/maxiters) # TV reg. param. multiplier
LR = 1e-2
losses = []
diffs_vel = []
diffs_width = []

meas = torch.from_numpy(imgr.meas3dar).to(device=device, dtype=torch.float)
x = x.to(device=device, dtype=torch.float)

xh_int = x[0,0]
xh_vel = torch.zeros(
    (meas.shape[1],meas.shape[2]), device=device, dtype=torch.float, requires_grad=True)
# xh_vel = x[0,1]
xh_width = 1*torch.ones((meas.shape[1],meas.shape[2]), device=device, dtype=torch.float)
xh_width = xh_width.requires_grad_()
# xh_width = x[0,2]
optimizer = optim.Adam([xh_vel, xh_width], lr=LR)
# optimizer = optim.SGD([xh_vel, xh_width], lr=LR, momentum=0.5)
xwidths = []
xvels = []

for i in tqdm(range(maxiters)):
    if USE_OPTIMIZER:
        optimizer.zero_grad()
    # compute the residual
    res = meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=pixelated)
    if DATA_FIDELITY == 'L1':
        loss = torch.mean(abs(res))
    elif DATA_FIDELITY == 'L2': 
        loss = torch.mean(res**2)
    if GRAD_RES_LOSS:
        loss += grad_res_loss(res, loss=DATA_FIDELITY)
    if USE_TV_LOSS:
        loss += beta**i*(lam_v*tv_loss(xh_vel) + lam_w*tv_loss(xh_width))
    loss.backward()

    if USE_OPTIMIZER:
        optimizer.step()
    else:
        with torch.no_grad():
            xh_vel -= mu_vel * xh_vel.grad
            xh_vel.grad.zero_()
            xh_width -= mu_width * xh_width.grad
            xh_width.grad.zero_()

    losses.append(loss.detach().cpu().numpy())
    diff_vel = torch.sum((x[0,1]-xh_vel)**2)/torch.sum(x[0,1]**2)
    diffs_vel.append(diff_vel.detach().cpu().numpy())

    # loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel.detach(),xh_width, pixelated=pixelated))**2)
    # loss.backward()

    # with torch.no_grad():
        # xh_width -= mu_width * xh_width.grad
        # xh_width.grad.zero_()

    #     losses.append(loss.detach().cpu().numpy())
    diff_width = torch.sum((x[0,2]-xh_width)**2)/torch.sum(x[0,2]**2)
    diffs_width.append(diff_width.detach().cpu().numpy())
    if i%1000==0:
        xwidths.append(xh_width.detach().cpu().numpy())
        xvels.append(xh_vel.detach().cpu().numpy())

# %% plots
plt.figure()
plt.plot(losses)
plt.title('Loss vs Iter')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(diffs_vel)
plt.title('||x_vel-xh_vel|| vs Iter')
plt.xlabel('Iter')
plt.ylabel('||x_vel-xh_vel||')
plt.show()

plt.figure()
plt.plot(diffs_width)
plt.title('||x_width-xh_width|| vs Iter')
plt.xlabel('Iter')
plt.ylabel('||x_width-xh_width||')
plt.show()

sr2 = Source(
    inten=xh_int.detach().cpu().numpy(),
    vel=xh_vel.detach().cpu().numpy(),
    width=xh_width.detach().cpu().numpy(),
    pix=True
)
ssims = compare_ssim(truth=sr.param3d, estimate=sr2.param3d)
# assume maxval for (int,vel,width)=(1,2,2.5)
psnrs = compare_psnr(truth=sr.param3d, estimate=sr2.param3d, maxval=(1,2,2.5))

sr.plot('Original')
plt.tight_layout()
sr2.plot('Estimated', ssims=ssims, psnrs=psnrs)
plt.tight_layout()
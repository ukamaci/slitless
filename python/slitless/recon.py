import torch, copy
import numpy as np
from torch import optim
from slitless.forward import Source, Imager, forward_op_torch
from slitless.measure import cycle_loss, compare_ssim, tv_loss
from tqdm.auto import tqdm

def grad_descent_solver(
    meas,
    truth=None,
    OPTIMIZER = 'ADAM',
    USE_TV_LOSS = True,
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    mu_vel = 1e1, # GD step size for vel
    mu_width = 1e1, # GD step size for width
    lam_i = 1e-7, # TV norm regularization parameter for intensity
    lam_v = 1e-2, # TV norm regularization parameter for velocity
    lam_w = 1e-2, # TV norm regularization parameter for width
    LR = 1e-2, # learning rate of the optimizer
    maxiters = 10000,
    return_arrays=False, # if true, return every 1000th vel&width recon
    savepath=None
):
    """
    It solves for the intensity, velocity and width using gradient descent.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    if type(meas) is not torch.Tensor:
        meas = torch.from_numpy(meas).to(device=device, dtype=torch.float)

    xh_int = copy.deepcopy(meas[...,0,:,:])
    xh_int = xh_int.requires_grad_()
    xh_vel = torch.zeros_like(
        xh_int, device=device, dtype=torch.float, requires_grad=True)
    xh_width = 1.25*torch.ones_like(xh_int, device=device, dtype=torch.float)
    xh_width = xh_width.requires_grad_()

    if OPTIMIZER is not None:
        if OPTIMIZER.upper() == 'ADAM':
            optimizer = optim.Adam([xh_int, xh_vel, xh_width], lr=LR)
        if OPTIMIZER.upper() == 'SGD':
            optimizer = optim.SGD([xh_int, xh_vel, xh_width], lr=LR)
    xhs = []
    if truth is not None:
        diffs_vel = []
        diffs_width = []

    for i in tqdm(range(maxiters)):
        if OPTIMIZER is not None:
            optimizer.zero_grad()
        if DATA_FIDELITY == 'L1':
            loss = torch.mean(abs(meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=True)))
        elif DATA_FIDELITY == 'L2': 
            loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=True))**2)
        if USE_TV_LOSS:
            loss += lam_i*tv_loss(xh_int) + lam_v*tv_loss(xh_vel) + lam_w*tv_loss(xh_width)
        loss.backward()

        if OPTIMIZER is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                xh_int -= mu_int * xh_int.grad
                xh_int.grad.zero_()
                xh_vel -= mu_vel * xh_vel.grad
                xh_vel.grad.zero_()
                xh_width -= mu_width * xh_width.grad
                xh_width.grad.zero_()

        losses.append(loss.detach().cpu().numpy())
        if truth is not None:
            diff_vel = torch.sum((truth[:,1]-xh_vel)**2)/torch.sum(truth[:,1]**2)
            diffs_vel.append(diff_vel.detach().cpu().numpy())

            diff_width = torch.sum((truth[:,2]-xh_width)**2)/torch.sum(truth[:,2]**2)
            diffs_width.append(diff_width.detach().cpu().numpy())

        if (savepath is not None) & (i>0) & (i%10000==0):
            xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
            np.save(savepath+f'recons_{i}.npy', xhs0)

        if return_arrays & (i>0) & (i%1000==0):
            xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
            xhs.append(xhs0)

    xhs = np.array(xhs)
    recon = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
    
    if truth is not None:
        return recon, losses, xhs, diffs_vel, diffs_width
    else:
        return recon, losses, xhs
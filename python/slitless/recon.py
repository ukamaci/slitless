import torch
import numpy as np
from torch import optim
from slitless.forward import Source, Imager, forward_op_torch
from slitless.measure import cycle_loss, compare_ssim, tv_loss
from tqdm.auto import tqdm

def grad_descent_solver(
    meas,
    OPTIMIZER = 'ADAM',
    USE_TV_LOSS = True,
    DATA_FIDELITY = 'L1', # 'L1' or 'L2'
    mu_vel = 1e1, # GD step size for vel
    mu_width = 1e1, # GD step size for width
    lam_v = 1e-2, # TV norm regularization parameter for velocity
    lam_w = 1e-2, # TV norm regularization parameter for width
    LR = 1e-3, # learning rate of the optimizer
    maxiters = 10000
):
    """
    It solves for the velocity and width using gradient descent.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    diffs_vel = []
    diffs_width = []

    meas = torch.from_numpy(meas).to(device=device, dtype=torch.float)

    xh_int = meas[0]
    xh_vel = torch.zeros(
        (meas.shape[1],meas.shape[2]), device=device, dtype=torch.float, requires_grad=True)
    xh_width = 0.25*torch.ones((meas.shape[1],meas.shape[2]), device=device, dtype=torch.float)
    xh_width = xh_width.requires_grad_()
    if OPTIMIZER == 'ADAM':
        optimizer = optim.Adam([xh_vel, xh_width], lr=LR)
    xwidths = []
    xvels = []

    for i in tqdm(range(maxiters)):
        if OPTIMIZER is not None:
            optimizer.zero_grad()
        if DATA_FIDELITY == 'L1':
            loss = torch.mean(abs(meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=True)))
        elif DATA_FIDELITY == 'L2': 
            loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=True))**2)
        if USE_TV_LOSS:
            loss += lam_v*tv_loss(xh_vel) + lam_w*tv_loss(xh_width)
        loss.backward()

        if OPTIMIZER is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                xh_vel -= mu_vel * xh_vel.grad
                xh_vel.grad.zero_()
                xh_width -= mu_width * xh_width.grad
                xh_width.grad.zero_()

        losses.append(loss.detach().cpu().numpy())

        if i%1000==0:
            xwidths.append(xh_width.detach().cpu().numpy())
            xvels.append(xh_vel.detach().cpu().numpy())
    
    return (xh_int.detach().cpu().numpy(), 
        xh_vel.detach().cpu().numpy(), 
        xh_width.detach().cpu().numpy(), 
        losses,
        np.array(xwidths),
        np.array(xvels)
    )
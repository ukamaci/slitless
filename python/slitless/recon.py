import torch, copy
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from slitless.forward import Source, Imager, forward_op_torch, forward_op
from slitless.measure import cycle_loss, compare_ssim, tv_loss
from scipy.optimize import minimize
from tqdm.auto import tqdm


class Reconstructor():
    def __init__(
        self,
        *,
        imager=None,
        solver=None,
        **solver_params
    ):
        self.imager = imager 
        self.source = imager.srpix 
        self.solver = solver
        self.solver_params = solver_params

    def solve(
        self,
        num_realizations=1
    ):
        self.num_realizations = num_realizations
        recons = []
        losses = []
        for i in range(num_realizations):
            _ = self.imager.get_measurements(
                dbsnr=self.imager.dbsnr, max_count=self.imager.max_count, 
                noise_model=self.imager.noise_model
            )
            recon, loss = self.solver(
                imager = self.imager,
                **self.solver_params
            )
            recons.append(recon)
            losses.append(loss)
        self.recons = Recon(recon=np.array(recons), losses=np.array(losses), 
                            imager=self.imager, source=self.source)
        self.recons.eval()
        return self.recons
            
        
class Recon():
    def __init__(
            self,
            *,
            recon=None,
            losses=None,
            imager=None,
            source=None
    ):
        self.recon = recon
        self.losses = losses
        self.imager = imager
        self.source = source
        self.losses_avg = np.mean(self.losses, axis=0)

    def plot(
            self,
            compare=False,
            index=0,
            title=''
    ):
        sr = Source(
            param3d = self.recon[index],
            pix=True
        )
        if compare is True:
            assert self.source is not None, "Source is not given!"
            truth = self.source.param3d

            fig, ax = plt.subplots(2,3, figsize=(15,8))
            plt.suptitle(title)
            i0=ax[0,0].imshow(sr.inten, cmap='hot')
            ax[0,0].set_title('Intensity')
            fig.colorbar(i0, ax=ax[0,0])
            i0=ax[0,1].imshow(sr.vel, cmap='seismic')
            fig.colorbar(i0, ax=ax[0,1])
            ax[0,1].set_title('Velocity [pix]')
            i0=ax[0,2].imshow(sr.width, cmap='plasma')
            fig.colorbar(i0, ax=ax[0,2])
            ax[0,2].set_title('Linewidth [pix]')

            i0=ax[1,0].imshow(truth[0], cmap='hot')
            ax[1,0].set_title('True Intensity')
            fig.colorbar(i0, ax=ax[1,0])
            i0=ax[1,1].imshow(truth[1], cmap='seismic')
            fig.colorbar(i0, ax=ax[1,1])
            ax[1,1].set_title('True Velocity [pix]')
            i0=ax[1,2].imshow(truth[2], cmap='plasma')
            fig.colorbar(i0, ax=ax[1,2])
            ax[1,2].set_title('True Linewidth [pix]')

            plt.tight_layout()
            plt.show()
        else:
            sr.plot(title=title)

    def plot_loss(self):
        plt.figure()
        plt.title('Loss vs Iter')
        plt.plot(self.losses_avg, linewidth=2)
        plt.show()

    def eval(self):
        truth_pix = self.source.param3d
        truth_pix_mean = np.ones_like(truth_pix) * truth_pix.mean(axis=(1,2))[:,None,None]
        recon_pix = self.recon
        truth_pix = np.repeat(truth_pix[np.newaxis,:], len(recon_pix), axis=0)
        truth_pix_mean = np.repeat(truth_pix_mean[np.newaxis,:], len(recon_pix), axis=0)
        truth_phy = self.imager.frompix(truth_pix, width_unit='km/s', array=True)
        truth_phy_mean = self.imager.frompix(truth_pix_mean, width_unit='km/s', array=True)
        recon_phy = self.imager.frompix(recon_pix, width_unit='km/s', array=True)  

        self.ssim = compare_ssim(truth=truth_pix, estimate=recon_pix)
        self.rmse_pix = np.sqrt(np.mean((recon_pix-truth_pix)**2, axis=(-1,-2)))
        self.rmse_phy = np.sqrt(np.mean((recon_phy-truth_phy)**2, axis=(-1,-2)))
        self.mae_pix = np.mean(abs(recon_pix-truth_pix), axis=(-1,-2))
        self.mae_phy = np.mean(abs(recon_phy-truth_phy), axis=(-1,-2))
        self.ssim_m = compare_ssim(truth=truth_pix, estimate=truth_pix_mean)
        self.rmse_pix_m = np.sqrt(np.mean((truth_pix_mean-truth_pix)**2, axis=(-1,-2)))
        self.rmse_phy_m = np.sqrt(np.mean((truth_phy_mean-truth_phy)**2, axis=(-1,-2)))
        self.mae_pix_m = np.mean(abs(truth_pix_mean-truth_pix), axis=(-1,-2))
        self.mae_phy_m = np.mean(abs(truth_phy_mean-truth_phy), axis=(-1,-2))


def grad_descent_solver(
    imager=None,
    truth=None,
    OPTIMIZER = 'ADAM',
    USE_TV_LOSS = True,
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    lam_i = 1e-2, # TV norm regularization parameter for intensity
    lam_v = 1e-2, # TV norm regularization parameter for velocity
    lam_w = 1e-2, # TV norm regularization parameter for width
    LR = 1e-2, # learning rate of the optimizer
    maxiter = 10000,
    # return_arrays=False, # if true, return every 1000th vel&width recon
    savepath=None
):
    """
    It solves for the intensity, velocity and width using gradient descent.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    meas = imager.meas3dar.copy()

    if type(meas) is not torch.Tensor:
        meas = torch.from_numpy(meas).to(device=device, dtype=torch.float)

    mask = imager.mask.copy()
    if type(mask) is not torch.Tensor:
        mask = torch.from_numpy(mask).to(device=device, dtype=torch.float)

    xh_int = copy.deepcopy(meas[...,0,:,:])
    xh_int = xh_int.requires_grad_()
    xh_vel = torch.zeros_like(
        xh_int, device=device, dtype=torch.float, requires_grad=True)
    xh_width = 1.25*torch.ones_like(xh_int, device=device, dtype=torch.float)
    xh_width = xh_width.requires_grad_()

    if OPTIMIZER.upper() == 'ADAM':
        optimizer = optim.Adam([xh_int, xh_vel, xh_width], lr=LR)
    if OPTIMIZER.upper() == 'SGD':
        optimizer = optim.SGD([xh_int, xh_vel, xh_width], lr=LR)
    xhs = []
    if truth is not None:
        diffs_vel = []
        diffs_width = []

    for i in tqdm(range(maxiter), leave=False):
        optimizer.zero_grad()
        if DATA_FIDELITY == 'L1':
            loss = torch.mean(abs(meas-imager.forward_op(xh_int,xh_vel,xh_width)))
        elif DATA_FIDELITY == 'L2': 
            loss = torch.mean((meas-imager.forward_op(xh_int,xh_vel,xh_width))**2)
        if USE_TV_LOSS:
            loss += lam_i*tv_loss(xh_int) + lam_v*tv_loss(xh_vel) + lam_w*tv_loss(xh_width)
        loss.backward()

        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        if truth is not None:
            diff_vel = torch.sum((truth[:,1]-xh_vel)**2)/torch.sum(truth[:,1]**2)
            diffs_vel.append(diff_vel.detach().cpu().numpy())

            diff_width = torch.sum((truth[:,2]-xh_width)**2)/torch.sum(truth[:,2]**2)
            diffs_width.append(diff_width.detach().cpu().numpy())

        if (savepath is not None) & (i>0) & (i%10000==0):
            xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
            np.save(savepath+f'recons_{i}.npy', xhs0)

        # if return_arrays & (i>0) & (i%1000==0):
        #     xhs0 = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()
        #     xhs.append(xhs0)

    # xhs = np.array(xhs)
    losses = np.array(losses)
    recon = torch.stack((xh_int,xh_vel,xh_width), axis=-3).detach().cpu().numpy()

    # recon = Recon(recon=recon, losses=losses, imager=imager)

    return recon, losses


def scipy_solver(
    imager=None,
    OPTIMIZER = 'L-BFGS-B',
    DATA_FIDELITY = 'L2', # 'L1' or 'L2'
    lam_i = 5e2, # TV norm regularization parameter for intensity
    lam_v = 5e2, # TV norm regularization parameter for velocity
    lam_w = 1e0, # TV norm regularization parameter for width
    maxiter=10000
    ):

    def obj_ls(x, meas=None, mask=None, lam_i=1e1, lam_v=1e1, lam_w=1e1):
        aa, bb = meas.shape[1:]
        if mask is None:
            a3=0
        intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
        diff = forward_op(intensity, doppler, linewidth, pixelated=imager.pixelated, mask=mask) - meas
        regu = (
            lam_v*np.sum(np.diff(doppler, axis=0)**2) + 
            lam_w*np.sum(np.diff(linewidth, axis=0)**2) +
            lam_i*np.sum(np.diff(intensity, axis=0)**2)
        )

        if DATA_FIDELITY == 'L2':
            return np.sum(diff**2) + regu
        elif DATA_FIDELITY == 'L2':
            return np.sum(abs(diff)) + regu

    meas = imager.meas3dar.copy()
    mask = imager.mask.copy()
    aa, bb = meas[0].shape

    int0 = meas[0].copy()
    vel0 = np.zeros_like(int0)
    width0 = np.ones_like(int0)
    x0 = np.stack((int0, vel0, width0), axis=0).flatten()

    rec = np.zeros((3,aa,bb))
    for i in tqdm(range(bb)):
        x0 = np.stack( ( int0[:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
        recon = minimize(
            obj_ls, 
            x0, 
            args=(meas[:,:,[i]], mask[:,[i]], lam_i, lam_v, lam_w), 
            method=OPTIMIZER,
            options={'disp':False, 'maxiter':maxiter, 'adaptive':True}
        )
        rec[:,:,i] = recon.x.reshape(3,aa)

    losses = []
    return rec, losses
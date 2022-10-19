from skimage.metrics import structural_similarity as skimage_ssim
from mas.decorators import _vectorize
from slitless.forward import forward_op_torch
import numpy as np
import torch
import torch.nn as nn

@_vectorize(signature='(a,b),(a,b)->()', included=['truth', 'estimate'])
def compare_ssim(*, truth, estimate):
    return skimage_ssim(estimate, truth, data_range=np.max(truth) - np.min(truth))

@_vectorize(signature='(a,b),(a,b)->()', included=['truth', 'estimate'])
def nrmse(*, truth, estimate, normalization=None):
    if normalization=='sigma':
        norm = np.std(truth)
    elif normalization=='minmax':
        norm = truth.max() - truth.min()
    elif normalization is None:
        norm = 1
    return np.sqrt(np.mean((truth-estimate)**2)) / (norm)

@_vectorize(signature='(a,b),(a,b)->()', included=['truth', 'estimate'])
def compare_psnr(*, truth, estimate):
    """
    Computes PSNR.
    maxval is the maximum value that the true array can possibly take.
    """
    return 10*np.log10(truth.max()**2/np.mean((truth-estimate)**2))

def tv_loss(img):
    """
    Computes the (mean) total variation of the input image.
    """
    dif1 = torch.abs(img[..., 1:, :] - img[..., :-1, :])
    dif2 = torch.abs(img[..., :, 1:] - img[..., :, :-1])
    return dif1.mean() + dif2.mean()

dy = lambda img: img[..., 1:, :] - img[..., :-1, :]
dx = lambda img: img[..., :, 1:] - img[..., :, :-1]
def grad_res_loss(res, loss='L2'):
    """
    Computes the loss of the first and second order derivatives of the residual.
    based on "Shan, Q., Jia, J., & Agarwala, A. (2008). High-quality motion
    deblurring from a single image. Acm transactions on graphics (tog), 27(3),
    1-10."
    """
    dxres = dx(res); dyres = dy(res)
    dxxres = dx(dxres); dyyres = dy(dyres); dxyres = dx(dyres)
    grads = [dxres, dyres, dxxres, dyyres, dxyres]
    mus = [0.5, 0.5, 0.25, 0.25, 0.25]
    lossum = 0
    for mu, grad in zip(mus,grads):
        if loss.upper()=='L1':
            lossum += mu * torch.mean(abs(grad))
        elif loss.upper()=='L2':
            lossum += mu * torch.mean(grad**2)
    return lossum

def nmse_torch(truth, estimate):
    """
    Normalized MSE with the (max-min) of the truth array. Normalization is 
    done in the last two dimensions (per channel). Torch implementation.

    Args:
        truth (Tensor): 4d tensor of the true data.
        estimate (Tensor): 4d tensor of the network output.

    Returns:
        loss (float): scalar loss value
    """
    maxmin = torch.amax(truth, dim=(2,3)) - torch.amin(truth, dim=(2,3))
    return torch.mean(((truth-estimate)/maxmin[:,:,None,None])**2)

def cycle_loss(meas, out, mode='L2'):
    """
    Loss metric in the measurement domain. Takes the network output, passes it 
    through the forward op, and takes the MSE with the measurement.

    Args:
        meas (Tensor): 4d tensor of the network input.
        out (Tensor): 4d tensor of the network output.

    Returns:
        loss (float): scalar loss value
    """
    if len(out.shape) == 4:
        inten, vel, width = out.transpose(1,0)
    elif len(out.shape) == 3:
        inten, vel, width = out
    if mode=='L2':
        return torch.mean((forward_op_torch(inten, vel, width) - meas)**2)
    elif mode=='L1':
        return torch.mean(torch.abs((forward_op_torch(inten, vel, width) - meas)))

def combine_losses(
    losses_param=[nn.L1Loss()], 
    losses_meas=[cycle_loss], 
    lam=[1,1], 
    outch_type='all'
):
    """
    Given the desired loss functions in `losses_param` and `losses_meas` with 
    corresponding weights, combines them all and returns a combined loss 
    function which takes as input the network input, output, and the labels 
    (truth array) and returns a scalar loss.

    Args:
        losses_param (list): list of loss functions that take as input the
            network output and the true labels.
        losses_meas (list): list of loss functions that take as input the
            network inputs (meas) and the network outputs.
        lam (list): list of weights for the losses with the same ordering as the 
            input loss functions.
        outch_type (str): string specifying the output channels of the network
            to be trained. It's either one of the (int,vel,width) channels, or
            all of them.

    Returns:
        combined (func): a function that takes as input the network input,
            output, and the true labels, and calculates the combined loss using
            the given loss functions.

    """
    assert len(lam) == len(losses_param) + len(losses_meas), "wrong length for lam"
    def combined(**kwargs):
        if outch_type == 'all':
            truth = kwargs['truth']
            out = kwargs['out']
        elif outch_type == 'int':
            truth = kwargs['truth'][:,[0]]
            out = kwargs['truth'].clone()
            out[:,[0]] = kwargs['out']
        elif outch_type == 'vel':
            truth = kwargs['truth'][:,[1]]
            out = kwargs['truth'].clone()
            out[:,[1]] = kwargs['out']
        elif outch_type == 'width':
            truth = kwargs['truth'][:,[2]]
            out = kwargs['truth'].clone()
            out[:,[2]] = kwargs['out']
        
        loss = 0
        ctr = 0
        if len(losses_param)>0:
            for loss_func in losses_param:
                loss += lam[ctr]*loss_func(truth, kwargs['out'])
                ctr += 1
        
        if len(losses_meas)>0:
            for loss_func in losses_meas:
                loss += lam[ctr]*loss_func(kwargs['meas'], out)
                ctr += 1

        return loss

    return combined
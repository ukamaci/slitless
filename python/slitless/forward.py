import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import camera, shepp_logan_phantom, cell
from scipy.optimize import minimize

def gauss(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * np.exp(-0.5*((x-mean)/sigma)**2)

def forward_op(true_intensity,true_doppler,true_linewidth):
    aa, bb = true_intensity.shape
    out = np.zeros((len(a_list)+1,)+(aa,bb))
    out[0] = true_intensity.copy()
    # assume columns of detector are independent
    for z,a in enumerate(a_list):
        for col in range(bb):
            for row in range(aa):
                out[z+1,:,col] += true_intensity[row,col] * gauss(
                    np.arange(aa)-row, a*true_doppler[row,col], abs(a)*true_linewidth[row,col]
                    )
    return out

def obj_ls(x, meas=None):
    aa, bb = meas.shape[1:]
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth) - meas
    return np.sum(diff**2)

if __name__ == '__main__':
    aa, bb = (300,300)
    detector_size = (aa,bb)
    true_intensity = resize(camera(), detector_size)
    true_doppler = resize(shepp_logan_phantom(), detector_size)-0.5
    true_linewidth = 5*resize(cell(), detector_size)
    # true_doppler = 5*(true_intensity.copy() - 0.5)
    # true_linewidth = true_intensity.copy() * 5
    # true_intensity = np.zeros((aa,bb))
    # true_intensity[5] = 1
    a_list=(-1,1)
    meas = forward_op(true_intensity, true_doppler, true_linewidth)
    # amo = np.zeros((aa,bb)) + 0.05
    # x0 = np.stack( ( meas[0], amo, amo ), axis=0 ).flatten()
    # recon = minimize(obj_ls, x0, args=(meas,), method='Nelder-Mead',
    #     options={'disp':True, 'maxiter':1000, 'adaptive':True})
    # rec = recon.x.reshape(3,aa,bb)

# 2022-10-31
# Ulas Kamaci
from slitless.forward import Source, Imager, forward_op
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm

# %% impulse
M = 64 # detector size
inten = np.outer(np.sin(np.arange(M)),np.sin(np.arange(M)))/5+0.8
vel = np.outer(np.cos(np.arange(M)),np.cos(np.arange(M)))
width = np.outer(np.sin(np.arange(M)),np.cos(np.arange(M)))/4+1.25
spectral_orders=[0,-1,1]
aa, bb = inten.shape

meas = forward_op(inten, vel, width, pixelated=True)

def obj_ls(x, meas=None):
    aa, bb = meas.shape[1:]
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth) - meas
    return np.sum(diff**2)

vel0 = np.zeros_like(vel)
width0 = np.ones_like(width)
x0 = np.stack( ( meas[0], vel0, width0 ), axis=0 ).flatten()

rec = np.zeros((3,aa,bb))
for i in tqdm(range(bb)):
    x0 = np.stack( ( meas[0,:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
    # recon = minimize(obj_ls, x0, args=(meas,), method='Nelder-Mead',
    recon = minimize(obj_ls, x0, args=(meas[:,:,[i]],), method='BFGS',
        options={'disp':True, 'maxiter':10000, 'adaptive':True})
    rec[:,:,i] = recon.x.reshape(3,aa)

# %% plot
sr = Source(
    inten=inten,
    vel=vel,
    width=width,
    pix=True
)

sr_r = Source(
    inten=rec[0],
    vel=rec[1],
    width=rec[2],
    pix=True
)

sr.plot('Orig')
sr_r.plot('Recon')
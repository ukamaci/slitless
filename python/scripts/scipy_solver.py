# 2022-10-31
# Ulas Kamaci
from slitless.forward import Source, Imager, forward_op
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm

# %% impulse
M = 20 # detector size
inten = np.outer(np.sin(np.arange(M)),np.sin(np.arange(M)))/5+0.8
vel = np.outer(np.cos(np.arange(M)),np.cos(np.arange(M)))
width = np.outer(np.sin(np.arange(M)),np.cos(np.arange(M)))/4+1.25
spectral_orders=[0,-1,1]
aa, bb = inten.shape

path_data = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v1/meta/selected_scans_train/'
date='20071211_002416' #APJ2019 image
# Rotate the params so that the effective dispersion direction is horizontal
sr = Source(
    inten=np.rot90(np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202]),
    vel=np.rot90(np.load(path_data+'vel_{}.npy'.format(date))[149:169, 182:202]),
    width=np.rot90(np.load(path_data+'width_{}.npy'.format(date))[149:169, 182:202]),
    pix=False
)
imgr = Imager(pixelated=False)
imgr.topix(sr)
inten, vel, width = imgr.srpix.param3d
# inten, vel, width = imgr.srpix.param3d.transpose(0,2,1)

# meas = forward_op(inten, vel, width, pixelated=False)
meas = imgr.get_measurements(max_count=50**2/0.9, model='poisson',no_noise=False)

def obj_ls(x, meas=None, apj_regu=False, a1=1e1, a2=1e1):
    aa, bb = meas.shape[1:]
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth, pixelated=False) - meas
    regu = 0
    if apj_regu:
        regu = (a1*np.sum(np.diff(doppler, axis=0)**2) + 
                a2*np.sum(np.diff(linewidth, axis=0)**2))

    return np.sum(diff**2) + regu

vel0 = np.zeros_like(vel)
width0 = np.ones_like(width)
x0 = np.stack( ( meas[0], vel0, width0 ), axis=0 ).flatten()

rec = np.zeros((3,aa,bb))
fig, ax = plt.subplots()
for i in tqdm(range(bb)):
    x0 = np.stack( ( meas[0,:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
    # recon = minimize(obj_ls, x0, args=(meas,), method='Nelder-Mead',
    recon = minimize(obj_ls, x0, args=(meas[:,:,[i]], True, 5e2, 5e2), method='L-BFGS-B',
        options={'disp':False, 'maxiter':10000, 'adaptive':True})
    rec[:,:,i] = recon.x.reshape(3,aa)
    ax.imshow(np.rot90(rec[1],k=3), cmap='seismic')
    plt.pause(0.1)
    plt.show()

rec = np.rot90(rec, k=3, axes=(1,2))
# inten, vel, width = imgr.srpix.param3d
# rec = rec.transpose(0,2,1)
# %% plot
sr_r = Source(
    inten=rec[0],
    vel=rec[1],
    width=rec[2],
    pix=True
)

sr_o = Source(param3d=np.rot90(imgr.srpix.param3d,k=3,axes=(1,2)), pix=True)

sr_o.plot('Orig')
sr_r.plot('Recon')
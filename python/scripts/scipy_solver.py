# 2022-10-31
# Ulas Kamaci
from slitless.forward import Source, Imager, forward_op
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.auto import tqdm
from scipy.signal import convolve2d

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
# inten=np.rot90(np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202])
# inten /= inten.max()
sr = Source(
    # inten=inten,
    inten=np.rot90(np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202]),
    vel=np.rot90(np.load(path_data+'vel_{}.npy'.format(date))[149:169, 182:202]),
    width=np.rot90(np.load(path_data+'width_{}.npy'.format(date))[149:169, 182:202]),
    pix=False
)
mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
# mask= np.ones_like(mask)
print('Mask0')
imgr = Imager(pixelated=False, mask=mask)
imgr.topix(sr)
inten, vel, width = imgr.srpix.param3d
# inten, vel, width = imgr.srpix.param3d.transpose(0,2,1)

# meas = forward_op(inten, vel, width, pixelated=False)
meas = imgr.get_measurements(max_count=50**2/0.9, noise_model='poisson',no_noise=False)

def obj_ls(x, meas=None, apj_regu=False, mask=mask, a1=1e1, a2=1e1, a3=1e1):
    aa, bb = meas.shape[1:]
    if mask is None:
        a3=0
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth, pixelated=False, mask=mask) - meas
    regu = 0
    if apj_regu:
        regu = (a1*np.sum(np.diff(doppler, axis=0)**2) + 
                a2*np.sum(np.diff(linewidth, axis=0)**2) +
                a3*np.sum(np.diff(intensity, axis=0)**2))

    return np.sum(diff**2) + regu

int0 = meas[0].copy()
amo = convolve2d(meas[0], 0.25*np.ones((3,3)), mode='same', boundary='fill', fillvalue=150)
int0[mask==0] = amo[mask==0]

vel0 = np.zeros_like(vel)
width0 = np.ones_like(width)
x0 = np.stack( ( meas[0], vel0, width0 ), axis=0 ).flatten()

rec = np.zeros((3,aa,bb))
fig, ax = plt.subplots()
for i in tqdm(range(bb)):
    # x0 = np.stack( ( meas[0,:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
    x0 = np.stack( ( int0[:,i], vel0[:,i], width0[:,i] ), axis=0 ).flatten()
    # recon = minimize(obj_ls, x0, args=(meas,), method='Nelder-Mead',
    recon = minimize(obj_ls, x0, args=(meas[:,:,[i]], True, mask[:,[i]], 5e2, 5e2, 1e-2), method='L-BFGS-B',
        options={'disp':False, 'maxiter':10000, 'adaptive':True})
    rec[:,:,i] = recon.x.reshape(3,aa)
    # ax.imshow(np.rot90(rec[1],k=3), cmap='seismic')
    # plt.pause(0.1)
    # plt.show()

# inten, vel, width = imgr.srpix.param3d
# rec = rec.transpose(0,2,1)
# %% plot
sr_r = Source(param3d=np.rot90(rec, k=3, axes=(1,2)), pix=True)
sr_o = Source(param3d=np.rot90(imgr.srpix.param3d,k=3,axes=(1,2)), pix=True)

sr_o.plot('Orig')
sr_r.plot('Recon')

# %% eval
srp_r = imgr.frompix(sr_r)
srp_o = imgr.frompix(sr_o)

int_phy, vel_phy, width_phy = srp_o.param3d
int_phy_r, vel_phy_r, width_phy_r = srp_r.param3d

int_er = int_phy_r - int_phy
int_er_n = int_er/int_phy*100
vel_er = vel_phy_r - vel_phy
vel_er_n = vel_er/(vel_phy+1e0)*100
width_er = width_phy_r - width_phy
width_er_n = width_er/width_phy*100
width_er = width_er/sr_r.wavelength*300000

print('Intensity RMS Error = {:.2f} erg/cm2.s.sr'.format(np.sqrt(np.mean(int_er**2))))
print('Doppler Velocity RMS Error = {:.2f} km/s'.format(np.sqrt(np.mean(vel_er**2))))
print('Line Width RMS Error = {:.2f} km/s'.format(np.sqrt(np.mean(width_er**2))))

fig, ax = plt.subplots(1,3, figsize=(13.8,4.8))
ax[0].hist(int_er_n.flatten(), edgecolor='Black', color='sandybrown')
ax[0].set_title(r'Intensity RMS Error = {:.2f} $\mathrm{{erg \, cm^{{-2}} \, s^{{-1}} \, sr^{{-1}}}}$'.format(np.sqrt(np.mean(int_er**2))))
ax[0].set_xlabel('Percent Deviation (%)')
ax[0].set_ylabel('Number of Occurances')
ax[1].hist(vel_er_n.flatten(), edgecolor='Black', color='sandybrown')
ax[1].set_title('Doppler Velocity RMS Error = {:.2f} km/s'.format(np.sqrt(np.mean(vel_er**2))))
ax[1].set_xlabel('Percent Deviation (%)')
ax[1].set_ylabel('Number of Occurances')
ax[2].hist(width_er_n.flatten(), edgecolor='Black', color='sandybrown')
ax[2].set_title('Line Width RMS Error = {:.2f} km/s'.format(np.sqrt(np.mean(width_er**2))))
ax[2].set_xlabel('Percent Deviation (%)')
ax[2].set_ylabel('Number of Occurances')
plt.tight_layout()
plt.show()
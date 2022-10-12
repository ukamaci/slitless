import scipy.io, glob
import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager
from slitless.evaluate import predict, net_loader
from slitless.measure import compare_ssim, nrmse
from slitless.recon import grad_descent_solver
import matplotlib

matplotlib.use('Agg')

figpath = '/home/kamo/resources/slitless/figures/tmp/'

# Comparison parameters
imsize = 100

# Path to the Matlab folder
folder_path = '/home/kamo/resources/tip2014/'

# Constants in Figen's code
lambda0 = 195.12 # in A
LightSpeed = 300000 # in km/s
Dispersion = 50 # dispersion scale, in mA/pixels

# read the dictionary of data from Figen's folder
amo = scipy.io.loadmat(folder_path + 'realData1_3.mat')
# assign the true parameters
inten = amo['intensity'][:imsize,:imsize]
inten /= inten.max()
vel = amo['velocity'][:imsize,:imsize] * lambda0*1000/LightSpeed/Dispersion
width = amo['width'][:imsize,:imsize] * 1000/Dispersion
# vel[vel>0.2]=0.2

# assign the measurements simulated by Figen
meas_0 = amo['dispersedI0'][:imsize,:imsize]
meas_1 = amo['dispersedI1'][:imsize,:imsize]
meas_m1 = amo['dispersedI2'][:imsize,:imsize]

# assign the estimates by Figen
est_inten = amo['estF'][:imsize,:imsize]
est_vel = amo['estD'][:imsize,:imsize]
est_width = amo['estW'][:imsize,:imsize]

sr_true = Source(
    inten=inten[:imsize,:imsize],
    vel=vel[:imsize,:imsize],
    width=width[:imsize,:imsize],
    pix=True
)

sr_fig = Source(
    inten=est_inten[:imsize,:imsize],
    vel=est_vel[:imsize,:imsize],
    width=est_width[:imsize,:imsize],
    pix=True
)
ssims_fig = compare_ssim(truth=sr_true.param3d, estimate=sr_fig.param3d)
rmses_fig = nrmse(truth=sr_true.param3d, estimate=sr_fig.param3d)

im = Imager(pixelated=True)
im.get_measurements(sr_true)

sr_true.plot('Sources')
plt.savefig(figpath+'sources.png')
sr_fig.plot('Figen Estimates', ssims=ssims_fig, rmses=rmses_fig)
plt.savefig(figpath+'figo.png')
im.plot('Kamo Meas')
plt.savefig(figpath+'meas.png')

foldname0 = '2022_08_06__14_36_01*'
foldpath = glob.glob('../results/saved/'+foldname0)[0]
net_vel = net_loader(foldpath)
foldname0 = '2022_08_29__17_12_59*'
foldpath = glob.glob('../results/saved/'+foldname0)[0]
net_width = net_loader(foldpath)

meas3d = np.stack((meas_0,meas_m1,meas_1))
meas3d /= meas3d[0].max()
meas3d = meas3d

# im.meas3dar[:,-4:,:] = meas3d[:,-4:,:]

pred_vel = predict(net_vel, meas3d)
pred_width = predict(net_width, meas3d)

# meass = np.random.normal(loc=im.meas3dar, scale=2/900.)
# meass = np.random.normal(loc=im.meas3dar, scale=0.)
# meass[:,-4:] = meas3d[:,-4:]
meass = meas3d

_, gd_vel, gd_width, gd_loss, gd_widths, gd_vels = grad_descent_solver(
    meass, 
    lam_v=2e-2,
    lam_w=2e-2,
    maxiters=20000,
    LR=1e-2
)

sr_unet = Source(
    inten=meas_0,
    vel=pred_vel,
    width=pred_width,
    pix=True
)

sr_gd = Source(
    inten=meas_0,
    vel=gd_vel,
    width=gd_width,
    pix=True
)
ssims_unet = compare_ssim(truth=sr_true.param3d, estimate=sr_unet.param3d)
rmses_unet = nrmse(truth=sr_true.param3d, estimate=sr_unet.param3d)

ssims_gd_vel = compare_ssim(truth=sr_true.vel, estimate=gd_vels)
ssims_gd_width = compare_ssim(truth=sr_true.width, estimate=gd_widths)

ssims_gd = compare_ssim(truth=sr_true.param3d, estimate=sr_gd.param3d)
rmses_gd = nrmse(truth=sr_true.param3d, estimate=sr_gd.param3d)

# %% plots
sr_unet.plot('Unet', ssims=ssims_unet, rmses=rmses_unet)
plt.savefig(figpath+'unet.png')
sr_gd.plot('Grad Descent', ssims=ssims_gd, rmses=rmses_gd)
plt.savefig(figpath+'gd.png')

plt.figure()
plt.plot(ssims_gd_vel, label='VEL')
plt.plot(ssims_gd_width, label='WIDTH')
plt.title('GD SSIM vs Iterations, vel={:.3f}, width={:.3f}'.format(
    ssims_gd_vel.max(), ssims_gd_width.max()))
plt.grid(which='both', axis='both')
plt.xlabel('Iterations / 1000')
plt.ylabel('SSIM')
plt.legend()
plt.show()
plt.savefig(figpath+'gd_ssim_iters.png')
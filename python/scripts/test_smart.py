import numpy as np
from slitless.forward import Imager, datacube_generator
from slitless.recon import smart2

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data = np.load(path_data + 'eis_train_50_dsetv5.npy', allow_pickle=True).item()

samples = range(len(data['param3d']))
M = 64
L = 21
DISP = 0.022275
C = 299792.458
RW = 195.117937907451
WC = 195.119

Imgr = Imager(pixelated=True, spectral_orders=[0, -1, 1], intenscale=1)

vel_two = []
wid_two = []
vel_edges_two = []
vel_center_two = []
wid_edges_two = []
wid_center_two = []

CENT1 = -1.13*(195.11794/299792.458)+195.119
WID_PIX = 42.74*(195.11794/299792.458) / DISP
FRAC1 = 0.8555
FRAC2 = 0.0521
FRAC_BG = 0.0924
CENT2 = 195.17803
BG_SHAPE = np.ones(L) / L

for idx in samples:
    truth = data['param3d'][idx]
    meas = data['meas_damped'][idx]
    truth_wid_km = truth[2] * C / RW

    Imgr.meas3dar_nn = meas[:3]
    Imgr.meas3dar = meas[:3]

    # Stage 1: pw=0, default const init
    recon1, _ = smart2(
        imager=Imgr, fitter='mpfit', psi=0.2, maxouter=5, maxinner=20,
        prior_weight=0,
        cent1=CENT1, wid1=WID_PIX*DISP, wid2=WID_PIX*DISP,
        n_jobs=-1,
    )

    # Build 3-component init_cube using stage-1 velocity
    int0 = meas[0]
    vel2_offset = (CENT2 - CENT1) / DISP  # pixel separation between lines

    v1_0 = recon1[1]
    w1_0 = WID_PIX * np.ones_like(int0)
    cube1 = datacube_generator(np.stack((int0 * FRAC1, v1_0, w1_0), axis=0), lamdim=L)

    v2_0 = v1_0 + vel2_offset
    w2_0 = WID_PIX * np.ones_like(int0)
    cube2 = datacube_generator(np.stack((int0 * FRAC2, v2_0, w2_0), axis=0), lamdim=L)

    bg_cube = BG_SHAPE[:, np.newaxis, np.newaxis] * (int0 * FRAC_BG)[np.newaxis, :, :]
    init_cube = cube1 + cube2 + bg_cube

    # Stage 2: pw=1
    recon2, _ = smart2(
        imager=Imgr, fitter='mpfit', psi=0.2, maxouter=5, maxinner=20,
        prior_weight=1,
        cent1=CENT1, wid1=WID_PIX*DISP, wid2=WID_PIX*DISP,
        init_cube=init_cube,
        n_jobs=-1,
    )
    recon2_phys = Imgr.frompix(recon2, width_unit='km/s', array=True)

    ve = recon2_phys[1] - truth[1]
    we = recon2_phys[2] - truth_wid_km
    vel_two.append([np.sqrt(np.mean(ve**2)), np.mean(ve)])
    wid_two.append([np.sqrt(np.mean(we**2)), np.mean(we)])

    for region, y_slice in [('edge', [slice(0,10), slice(54,64)]), ('center', [slice(27,37)])]:
        for sl in y_slice:
            ve_s = ve[sl, :]
            we_s = we[sl, :]
            if region == 'edge':
                vel_edges_two.append(ve_s.flatten())
                wid_edges_two.append(we_s.flatten())
            else:
                vel_center_two.append(ve_s.flatten())
                wid_center_two.append(we_s.flatten())

r = np.array(vel_two)
w = np.array(wid_two)
ve = np.concatenate(vel_edges_two)
vc = np.concatenate(vel_center_two)
we = np.concatenate(wid_edges_two)
wc = np.concatenate(wid_center_two)

print("\n" + "=" * 60)
print("  TWO-STAGE SMART2 (pw=0 -> re-init -> pw=1)")
print("=" * 60)
print(f"  Overall:  vel RMSE={r[:,0].mean():.2f} bias={r[:,1].mean():.2f}  |  wid RMSE={w[:,0].mean():.2f} bias={w[:,1].mean():.2f} km/s")
print(f"  Edges:    vel RMSE={np.sqrt(np.mean(ve**2)):.2f} bias={np.mean(ve):.2f}  |  wid RMSE={np.sqrt(np.mean(we**2)):.2f} bias={np.mean(we):.2f} km/s")
print(f"  Center:   vel RMSE={np.sqrt(np.mean(vc**2)):.2f} bias={np.mean(vc):.2f}  |  wid RMSE={np.sqrt(np.mean(wc**2)):.2f} bias={np.mean(wc):.2f} km/s")
print()
# baselines
print("  Single-stage baselines (from earlier):")
print(f"  pw=1 const init: vel RMSE=7.28 bias=-0.27  |  wid RMSE=1.92 bias=-0.01")
print(f"  pw=0 const init: vel RMSE=4.82 bias=+1.82  |  wid RMSE=15.46 bias=+15.29")

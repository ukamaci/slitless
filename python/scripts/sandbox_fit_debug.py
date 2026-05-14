import numpy as np
from slitless.recon import smart2
from slitless.forward import Imager
import matplotlib.pyplot as plt

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data_file='eis_train_50_dsetv5.npy' 

data = np.load(path_data+data_file, allow_pickle=True).item()
param4dar, meas4dar = data['param3d'], data['meas_damped']

meas = meas4dar[0, :3].copy() # Take first realization, 3 detectors

Imgr = Imager(pixelated=True, dbsnr=30, avg_count=30**2, noise_model=None, 
    spectral_orders=[0,-1,1])
Imgr.meas3dar = meas

recon, cube = smart2(
    meas=meas,
    imager=Imgr,
    fitter='pmf', # use pmf to avoid mpfit/eispac dependency if not needed
    psi=0.2,
    maxouter=5,
    maxinner=20,
    prior_weight=0,
    cent1=-1.13*(195.11794/299792.458)+195.11803,
    wid1=42.74*(195.11794/299792.458),
    wid2=42.74*(195.11794/299792.458)
)
print("Finished!")

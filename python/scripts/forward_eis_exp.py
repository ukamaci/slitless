# Written for checking if the forward op works fine.

import numpy as np
import time
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager
from mas.forward_model import size_equalizer

path_data = '/home/kamo/resources/slitless/data/datasets/dset0_2022_06_14/'
path_data = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v1/meta/selected_scans_train/'
date='20071211_002416'

# sr = Source(
#     inten=np.load(path_data+'int.npy'),
#     vel=np.load(path_data+'vel.npy'),
#     width=np.load(path_data+'wid.npy'),
#     pix=False
# )
sr = Source(
    inten=np.load(path_data+'int_{}.npy'.format(date))[149:169, 182:202],
    vel=np.load(path_data+'vel_{}.npy'.format(date))[149:169, 182:202],
    width=np.load(path_data+'width_{}.npy'.format(date))[149:169, 182:202],
    pix=False
)

# dim=100
# sr = Source(
#     # inten=size_equalizer(np.ones((dim//2+1,dim//2+1)), (dim,dim)),
#     inten=np.ones((dim,dim)),
#     vel=np.ones((dim,dim)),
#     # vel=(np.arange(10000).reshape((100,100))%2-0.5)*300,
#     width=np.ones((dim,dim))*0.30,
#     pix=True
# )

imgr1 = Imager(pixelated=False)
imgr2 = Imager(pixelated=True)

t0 = time.time()
imgr1.get_measurements(sr)
t1 = time.time()
print('Time pix=False: {:.2f} secs'.format(t1-t0))
imgr2.get_measurements(sr)
t2 = time.time()
print('Time pix=True: {:.2f} secs'.format(t2-t1))

# %% plot
imgr1.srpix.plot()
imgr1.plot(title='Pixelated=False')
imgr2.plot(title='Pixelated=True')

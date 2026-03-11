# Written for checking if the forward op works fine.

import numpy as np
import time
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager
from mas.forward_model import size_equalizer
from slitless.plotting import generate_plus_cross, generate_horizontal_lines


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

# inten = np.array([
#     [1, 0, 0, 1, 0, 0, 1],
#     [0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 1, 1, 1, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0],
#     [1, 0, 0, 1, 0, 0, 1]
# ])

# # inten = generate_plus_cross(size=100, line_width=10, sigma=3)
# inten = generate_horizontal_lines(size=21, num_lines=3, line_width=1, sigma=0.01)
# # vel = 2*np.outer(np.ones(len(inten)), np.cos(np.linspace(0, 4*np.pi, len(inten[0]))))
# vel = np.zeros_like(inten)
# # width = 0.08*np.outer(np.ones(len(inten)), np.arange(len(inten[0])))
# width = 1*np.outer(np.ones(len(inten)), 1+np.random.rand(len(inten[0])))

# sr = Source(
#     inten=inten,
#     vel=vel,
#     width=width,
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

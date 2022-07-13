# Written for checking if the forward op works fine.

import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager
from mas.forward_model import size_equalizer

path_data = '/home/kamo/resources/slitless/data/datasets/dset0_2022_06_14/'

sr = Source(
    inten=np.load(path_data+'int.npy'),
    vel=np.load(path_data+'vel.npy'),
    width=np.load(path_data+'wid.npy')
)

# sr = Source(
#     inten=size_equalizer(np.ones((51,51)), (100,100)),
#     vel=np.ones((100,100))*500,
#     # vel=(np.arange(10000).reshape((100,100))%2-0.5)*300,
#     width=np.ones((100,100))*0.30
# )

imgr = Imager()

imgr.get_measurements(sr)

# %% plot
imgr.srpix.plot()
imgr.plot()

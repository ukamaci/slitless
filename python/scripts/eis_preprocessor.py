# 2023-01-02
# Ulas Kamaci
# Read npy files of intensity, velocity and width and project them onto 
# [0, 6000] erg/cm2/s/sr, [-68.5, 68.5] km/s, and [0.022,-] A ranges.

import numpy as np
import glob
from tqdm import tqdm
from slitless.forward import Source
import matplotlib
from matplotlib import pyplot as plt

pathin = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v0_raw/'
pathout = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v0_trimmed/'
figout = '/home/kamo/resources/slitless/data/eis_data/datasets/figures_v0_trimmed/'

def preprocessor():
    files_int = glob.glob(pathin + 'int*')
    files_vel = glob.glob(pathin + 'vel*')
    files_width = glob.glob(pathin + 'width*')

    for i in tqdm(range(len(files_int))):
        inten = np.load(files_int[i])
        vel = np.load(files_vel[i])
        width = np.load(files_width[i])

        inten[inten>6000] = 6000
        inten[inten<0] = 0

        vel[vel>68.5] = 68.5
        vel[vel<-68.5] = -68.5

        width[width<0.022] = 0.022

        np.save(pathout+files_int[i].split('/')[-1], inten)
        np.save(pathout+files_vel[i].split('/')[-1], vel)
        np.save(pathout+files_width[i].split('/')[-1], width)

def visualizer():
    files_int = glob.glob(pathout + 'int*')
    files_vel = glob.glob(pathout + 'vel*')
    files_width = glob.glob(pathout + 'width*')
    files_int.sort()
    files_vel.sort()
    files_width.sort()

    matplotlib.use('Agg')
    for i in tqdm(range(len(files_int))):
        inten = np.load(files_int[i])
        vel = np.load(files_vel[i])
        width = np.load(files_width[i])

        sr = Source(
            inten=inten,
            vel=vel,
            width=width,
            pix=False
        )

        sr.plot(title='{}. {}'.format(i+1, files_int[i][-19:-4]))
        plt.savefig(figout+f'{i+1}_{files_int[i][-19:-4]}', dpi=250)
        plt.close()
    
    matplotlib.use('QtAgg')
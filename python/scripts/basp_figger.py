import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, glob, os
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

dataset_path = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v0_12_scans/eistest256/'

dataset = BasicDataset(data_dir=dataset_path, fold='test', dbsnr=35)

savedir = dataset_path+'figs/'
if not os.path.exists(savedir):
    os.mkdir(savedir)

for i, data in enumerate(dataset):
    meas, params = data

    plt.imsave(savedir+'meas0_{:02d}.png'.format(i), meas[0], cmap='hot')
    plt.imsave(savedir+'meas-1_{:02d}.png'.format(i), meas[1], cmap='hot')
    plt.imsave(savedir+'meas1_{:02d}.png'.format(i), meas[2], cmap='hot')

    plt.imsave(savedir+'int_{:02d}.png'.format(i), params[0], cmap='hot')
    plt.imsave(savedir+'vel_{:02d}.png'.format(i), params[1], cmap='seismic')
    plt.imsave(savedir+'width_{:02d}.png'.format(i), params[2], cmap='plasma')
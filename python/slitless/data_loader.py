import logging, glob, os
from os import listdir
from os.path import splitext

import numpy as np
import torch
from slitless.forward import add_noise, forward_op
from PIL import Image
from torch.utils.data import Dataset
from multiprocessing import Pool

class BasicDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr

        if fold == 'train':
            self.train = True
            self.task_dir = os.path.join(data_dir, 'train')
        elif fold == 'val':
            self.val = True
            self.task_dir = os.path.join(data_dir, 'val')
        elif fold == 'test':
            self.test = True
            self.task_dir = os.path.join(data_dir, 'test')

        self.files = glob.glob(self.task_dir+'/data*.npy')
        self.files.sort()
        for file in self.files:
            data = np.load(file, allow_pickle=True).item()
            params = np.stack([data['int'], data['vel'], data['width']])
            meas = np.stack([data['meas_0'], data['meas_-1'], data['meas_1']])
            self.data.append((meas, params))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        meas, params = self.data[idx]

        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None, model='Gaussian')

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

class OntheflyDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr

        if fold == 'train':
            self.train = True
        elif fold == 'val':
            self.val = True

        self.task_dir = os.path.join(self.data_dir, fold)

        file = glob.glob(self.task_dir+'/otf*npy')[0]
        self.data = np.load(file)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data) // 3

    def __getitem__(self, idx):
        inten, vel, width = self.data[3*idx:3*idx+3]
        inten = (inten-inten.min())/(inten.max()-inten.min()+1e-6)
        vel = (vel-vel.min())/(vel.max()-vel.min()+1e-6)
        width = (width-width.min())/(width.max()-width.min()+1e-6)

        vel_max = 2 # pixels
        width_max = 2.5 # pixels
        width_min = 1 # pixels

        v0, v1 = np.random.uniform(0, vel_max, 2) * [-1,1]
        w0, w1 = np.sort(np.random.uniform(width_min, width_max, 2))
        
        vel = vel * (v1 - v0) + v0
        width = width * (w1 - w0) + w0

        meas3dar = forward_op(
            true_intensity=inten,
            true_doppler=vel,
            true_linewidth=width,
            spectral_orders=[0,-1,1],
            pixelated=True
        )

        params = np.stack((inten, vel, width))
        meas = add_noise(meas3dar, dbsnr=self.dbsnr, no_noise=self.dbsnr==None, model='Gaussian')

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

def data_rewrite(file):
    name = file.split('/')[-1]
    out_name = 'data/train2/{}'.format(name)
    data = np.load(file, allow_pickle=True).item()
    im_starry = data['image_stars']
    im_clean = data['image_ori']
    im_starry = np.where(np.isnan(im_starry), im_clean, im_starry)
    array = np.stack((im_starry,im_clean))
    np.save(out_name, array)

def data_rewriter_par():
    files = glob.glob('data/train/*')
    pool = Pool()
    pool.map(data_rewrite, files)

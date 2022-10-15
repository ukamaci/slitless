import logging, glob, os
from os import listdir
from os.path import splitext

import numpy as np
import torch, time
from slitless.forward import add_noise, forward_op, forward_op_torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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

class OntheflyDataset2(Dataset):
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
        data = np.load(file)
        intens, vels, widths = data.reshape(3,-1,64,64)

        vel_max = 2 # pixels
        width_max = 2.5 # pixels
        width_min = 1 # pixels

        v0, v1 = (np.random.uniform(0, vel_max, (2,len(vels))) * [[-1],[1]])[:,:,None,None]
        w0, w1 = np.sort(np.random.uniform(width_min, width_max, (2,len(vels))), axis=0)[:,:,None,None]
        
        vels = vels * (v1 - v0) + v0
        widths = widths * (w1 - w0) + w0

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda')
        intens = torch.from_numpy(intens).to(device=device, dtype=torch.float)
        vels = torch.from_numpy(vels).to(device=device, dtype=torch.float)
        widths = torch.from_numpy(widths).to(device=device, dtype=torch.float)

        meas = []
        nb = 50
        ind = len(intens) // nb

        for i in range(nb):
            meas.extend(forward_op_torch(
                true_intensity=intens[i*ind:(i+1)*ind],
                true_doppler=vels[i*ind:(i+1)*ind],
                true_linewidth=widths[i*ind:(i+1)*ind],
                spectral_orders=[0,-1,1],
                pixelated=True,
                device=device
            ).cpu().numpy())

        self.data = np.concatenate((meas,intens.cpu()[:,None],vels.cpu()[:,None],widths.cpu()[:,None]), axis=1)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meas = self.data[idx, :3]
        params = self.data[idx, 3:]

        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None, model='Gaussian')

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

def dloadertesting():
    dataset_path = glob.glob('../../data/datasets/dset6*')[0]
    t0 = time.time()
    trainset = OntheflyDataset2(data_dir=dataset_path, fold='train', dbsnr=35)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=16)
    print('Time Dataset Onthefly: {}'.format(time.time()-t0))
    t0 = time.time()
    for i,data in enumerate(trainloader):
        # tada = data
        if i==2000:
            break
    print('Time Loop Onthefly: {}'.format(time.time()-t0))

    dataset_path = glob.glob('../../data/datasets/dset5*')[0]
    t0 = time.time()
    trainset = BasicDataset(data_dir=dataset_path, fold='train', dbsnr=35)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=16)
    print('Time Dataset Basic: {}'.format(time.time()-t0))

    t0 = time.time()
    for i,data in enumerate(trainloader):
        # tada = data
        if i==2000:
            break
    print('Time Loop Basic: {}'.format(time.time()-t0))

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

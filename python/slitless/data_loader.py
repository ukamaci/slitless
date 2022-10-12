import logging, glob, os
from os import listdir
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from multiprocessing import Pool


class BasicDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform

        if fold == 'train':
            self.train = True
            self.task_dir = os.path.join(data_dir, 'train')
        elif fold == 'val':
            self.val = True
            self.task_dir = os.path.join(data_dir, 'val')
        elif fold == 'test':
            self.test = True
            self.task_dir = os.path.join(data_dir, 'test')

        self.files = glob.glob(self.task_dir+'/*')
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

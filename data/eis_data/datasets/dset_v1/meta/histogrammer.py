# 2023-01-23
# Ulas Kamaci
# This script aims at capturing the histograms of the velocith and width
# parameters of the training set to be used in the non-EIS datasets.

import numpy as np
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset, DataLoader

dset_path = '../'
figspath = 'figs_cropped_64/'

trainset = BasicDataset(data_dir=dset_path, fold='train')
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=8)

meas, params = next(iter(trainloader))


import numpy as np
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset, DataLoader
from slitless.forward import Source, Imager
import os

dset_path = '../'
figspath = 'figs_cropped_64/'

trainset = BasicDataset(data_dir=dset_path, fold='train', dbsnr=35)
valset = BasicDataset(data_dir=dset_path, fold='val', dbsnr=35)
testset = BasicDataset(data_dir=dset_path, fold='test', dbsnr=35)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
meas_train, params_train = next(iter(trainloader))
meas_val, params_val = next(iter(valloader))
meas_test, params_test = next(iter(testloader))

imgr = Imager(pixelated=True)

if not os.path.exists(figspath):
    os.mkdir(figspath)
figpath = figspath + 'train/'
if not os.path.exists(figpath):
    os.mkdir(figpath)

for i in range(trainloader.batch_size):
    sr = Source(inten=params_train[i,0], vel=params_train[i,1], 
        width=params_train[i,2], pix=True)
    sr.plot(title='Train {}'.format(i))
    plt.savefig(figpath+'params_{}.png'.format(i))
    plt.close()

    imgr.meas3dar = meas_train[i]
    imgr.plot(title='Train {}'.format(i))
    plt.savefig(figpath+'meas_{}.png'.format(i))
    plt.close()

figpath = figspath + 'val/'
if not os.path.exists(figpath):
    os.mkdir(figpath)

for i in range(valloader.batch_size):
    sr = Source(inten=params_val[i,0], vel=params_val[i,1], 
        width=params_val[i,2], pix=True)
    sr.plot(title='Val {}'.format(i))
    plt.savefig(figpath+'params_{}.png'.format(i))
    plt.close()

    imgr.meas3dar = meas_val[i]
    imgr.plot(title='Val {}'.format(i))
    plt.savefig(figpath+'meas_{}.png'.format(i))
    plt.close()

figpath = figspath + 'test/'
if not os.path.exists(figpath):
    os.mkdir(figpath)

for i in range(valloader.batch_size):
    sr = Source(inten=params_test[i,0], vel=params_test[i,1], 
        width=params_test[i,2], pix=True)
    sr.plot(title='Test {}'.format(i))
    plt.savefig(figpath+'params_{}.png'.format(i))
    plt.close()

    imgr.meas3dar = meas_test[i]
    imgr.plot(title='Test {}'.format(i))
    plt.savefig(figpath+'meas_{}.png'.format(i))
    plt.close()
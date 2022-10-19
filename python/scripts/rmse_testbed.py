# Ulas Kamaci
# 2022-10-18
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch, glob, os
from slitless.measure import compare_ssim, nrmse
from slitless.networks.unet import UNet
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset, OntheflyDataset

foldname0 = '2022_10_15__18_56_55_NF_64_BS_4_LR_0.001_EP_250_KSIZE_(3, 1)_MSE_LOSS_ADAM_all'
foldpath = glob.glob('../results/saved/'+foldname0)[0]+'/'
modpath = foldpath+'best_model.pth'
# modpath = foldpath+'nf_64_LR_0.001_EP_50.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(
    in_channels=3,
    out_channels=3,
    numlayers=4,
    outch_type='all',
    start_filters=64,
    bilinear=True,
    ksizes=[(3,1),(3,1),(3,1),(3,1)],
    residual=False)

net.load_state_dict(torch.load(modpath))
net.to(device)
dataset_path = glob.glob('../../data/datasets/dset6*')[0]
fold = 'test'

# valset = OntheflyDataset(data_dir=dataset_path, fold=fold, dbsnr=25)
valset = BasicDataset(data_dir=dataset_path, fold=fold, dbsnr=25)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=8)

net.eval()

ssims=[]
rmses=[]
yvec=[]
outvec=[]
for i, data in enumerate(valloader):
    # get the inputs
    inputs = data[0].to(device=device, dtype=torch.float)
    true_outputs = np.array(data[1].cpu())
    if not hasattr(net, 'outch_type'):
        net.outch_type = 'all'

    if net.outch_type == 'int':
        y1 = true_outputs[:,[0]]
        title_str = 'Intensity'
    elif net.outch_type == 'vel':
        y1 = true_outputs[:,[1]]
        title_str = 'Velocity'
    elif net.outch_type == 'width':
        y1 = true_outputs[:,[2]]
        title_str = 'Linewidth'
    elif net.outch_type == 'all':
        y1 = true_outputs

    with torch.no_grad():
        outputs = net(inputs)
        outputs = np.array(outputs.cpu())
        ssim0 = compare_ssim(truth=y1, estimate=outputs)
        rmse0 = nrmse(truth=y1, estimate=outputs, normalization=None)
        ssims.extend(ssim0.squeeze())
        rmses.extend(rmse0.squeeze())
        if i*valloader.batch_size<10000:
            if net.outch_type == 'all':
                yvec.extend(y1.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
                outvec.extend(outputs.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
            else:
                yvec.extend(y1.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))
                outvec.extend(outputs.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))

ssims = np.array(ssims)
rmses = np.array(rmses)
yvec = np.array(yvec).transpose(1,0)
outvec = np.array(outvec).transpose(1,0)
est_bias = np.mean(outvec - yvec, axis=1) 
est_std = np.std(outvec - yvec, axis=1)
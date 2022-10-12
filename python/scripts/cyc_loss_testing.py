import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, glob, os, copy
from slitless.measure import compare_ssim, nrmse
from slitless.networks.unet import UNet
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset
from slitless.forward import forward_op_torch

fold_mse = '2022_10_05__23_48_57_NF_64_BS_4_LR_0.001_EP_50_KSIZE_(3, 1)_MSE_LOSS_ADAM_all'
fold_cyc = '2022_10_08__18_57_19_NF_64_BS_8_LR_0.001_EP_10_KSIZE_(3, 1)_CYCLE_ONLY_LOSS_ADAM_all'
fold_mse = glob.glob('../results/saved/'+fold_mse)[0]+'/'
fold_cyc = glob.glob('../results/saved/'+fold_cyc)[0]+'/'
mod_mse = fold_mse+'best_model.pth'
mod_cyc = fold_cyc+'best_model.pth'
net_mse = UNet(
    in_channels=3,
    out_channels=3,
    numlayers=4,
    outch_type='all',
    start_filters=64,
    bilinear=True,
    ksizes=[(3,1),(3,1),(3,1),(3,1)],
    residual=False)
net_cyc = copy.deepcopy(net_mse)
net_mse.load_state_dict(torch.load(mod_mse))
net_cyc.load_state_dict(torch.load(mod_cyc))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_mse.to(device)
net_cyc.to(device)
net_mse.eval()
net_cyc.eval()
dataset_path = glob.glob('../../data/datasets/dset5*')[0]
fold = 'val'
dataset = BasicDataset(data_dir = dataset_path, fold=fold)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

running_mlm = 0.0
running_mlc = 0.0
running_clm = 0.0
running_clc = 0.0
for i, data in enumerate(dataloader):
    # get the inputs
    inputs = data[0].to(device=device, dtype=torch.float)
    true_outputs = data[1].to(device=device, dtype=torch.float)

    with torch.no_grad():
        outputs_mse = net_mse(inputs)
        inten_m, vel_m, width_m = outputs_mse.transpose(1,0)
        outputs_cyc = net_cyc(inputs)
        inten_c, vel_c, width_c = outputs_cyc.transpose(1,0)
        # FIXME
        # inten_c = true_outputs.transpose(1,0)[0]
        # inten_m = inten_c
        mse_loss_mse = torch.mean((outputs_mse-true_outputs)**2)
        mse_loss_cyc = torch.mean((outputs_cyc-true_outputs)**2)
        cyc_loss_mse = torch.mean((forward_op_torch(inten_m, vel_m, width_m) - inputs)**2)
        cyc_loss_cyc = torch.mean((forward_op_torch(inten_c, vel_c, width_c) - inputs)**2)

        # mse_loss_mse = torch.mean(torch.abs(outputs_mse-true_outputs))
        # mse_loss_cyc = torch.mean(torch.abs(outputs_cyc-true_outputs))
        # cyc_loss_mse = torch.mean(torch.abs(forward_op_torch(inten_m, vel_m, width_m) - inputs))
        # cyc_loss_cyc = torch.mean(torch.abs(forward_op_torch(inten_c, vel_c, width_c) - inputs))

    running_mlm += mse_loss_mse/len(dataloader)
    running_mlc += mse_loss_cyc/len(dataloader)
    running_clm += cyc_loss_mse/len(dataloader)
    running_clc += cyc_loss_cyc/len(dataloader)

print(f'Run. MLM: {running_mlm}')
print(f'Run. MLC: {running_mlc}')
print(f'Run. CLM: {running_clm}')
print(f'Run. CLC: {running_clc}')
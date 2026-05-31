"""
Regenerate post-training figures for existing dset_v6 runs using the updated
evaluate.py / plotting.py (units fixed, hexbin_scatter added).

Run from python/scripts/:
    python regen_train_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader

from slitless.networks.unet import UNet
from denoising_diffusion_pytorch import Unet as DiffusionUnet
from slitless.data_loader import (BasicDataset,
    meas_transform, param_transform)
from slitless.evaluate import plot_recons, plot_val_stats, eval_snrlist
from slitless.plotting import barplot_group, scatter_hexbin

DSET          = 'dset_v6'
dset_root     = f'../../data/eis_data/datasets/{DSET}'
data_dir      = f'{dset_root}/data/'
dset_stats    = np.load(f'{dset_root}/norm_stats.npy', allow_pickle=True).item()
NUM_FILT      = 64
numdetectors  = 3
dbsnr         = 100
noise_model   = None
dbsnr_l       = [10, 20, 30, None]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

_meas_tf  = partial(meas_transform,  stats=dset_stats)
_param_tf = partial(param_transform, stats=dset_stats)
testset   = BasicDataset(data_dir=data_dir, transform=_meas_tf,
                         target_transform=_param_tf, fold='test',
                         dbsnr=dbsnr, noise_model=noise_model,
                         numdetectors=numdetectors)

RUNS = [
    # unet figures already regenerated; uncomment to redo
    {
        'name':       '2026_05_26__01_37_14_unet_NF_64_BS_32_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_100_None_K_3_dset_v6_logzscale',
        'model':      'unet',
        'batch_size': 64,
    },
    # {
    #     'name':       '2026_05_26__01_46_01_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_200_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_100_None_K_3_dset_v6_logzscale',
    #     'model':      'diffusion_unet',
    #     'batch_size': 16,
    # },
]

for run in RUNS:
    name    = run['name']
    savedir = f'../results/saved/{name}/'
    print(f'\n=== {name} ===')

    testloader = DataLoader(testset, batch_size=run['batch_size'], shuffle=True,
                            num_workers=4, persistent_workers=True, pin_memory=True)

    torch.cuda.empty_cache()

    if run['model'] == 'unet':
        net = UNet(in_channels=numdetectors, out_channels=3, numlayers=4,
                   outch_type='all', start_filters=NUM_FILT, bilinear=True,
                   ksizes=[(3, 1)], residual=False).to(device)
    else:
        net = DiffusionUnet(dim=NUM_FILT, channels=numdetectors, out_dim=3,
                            dim_mults=(1, 2, 4, 8), flash_attn=False).to(device)
        net.outch_type = 'all'
        _fwd = net.forward
        net.forward = lambda x: _fwd(x, torch.zeros(x.shape[0], device=x.device))

    # net.load_state_dict(torch.load(savedir + 'best_model.pth', map_location=device))
    # net.eval()

    print('  plot_val_stats ...')
    ssims, rmses, yvec, outvec = plot_val_stats(net, testloader, savedir, stats=dset_stats)

    print('  scatter_hexbin ...')
    scatter_hexbin(yvec, outvec, method_name=name, save=True,
                   savepath=savedir + 'hexbin_scatter_ep200.png', show=False)

    # print('  plot_recons ...')
    # plot_recons(net, testloader, numim=32,
    #             savedir=savedir + 'figures/', denormalize=True)

    # print('  eval_snrlist ...')
    # ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, noise_model=noise_model,
    #                                 fold='test', data_dir=data_dir, net=net)
    # labels_gr = ['int', 'vel', 'width']
    # barplot_group(ssims_l.mean(axis=1).swapaxes(0, 1),
    #               labels_gr=labels_gr, labels_mem=[str(j) for j in dbsnr_l],
    #               ylabel='SSIM', title='SSIM vs dBsnr',
    #               savedir=savedir + 'snr_barplot.png')
    # np.save(savedir + 'ssims_l.npy', ssims_l)
    # np.save(savedir + 'rmses_l.npy', rmses_l)

    # print(f'  Done. Figures written to {savedir}')

print('\nAll runs complete.')

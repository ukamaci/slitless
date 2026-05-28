"""
Resume training from a completed or interrupted run.

Usage (run from python/scripts/):
    python resume_train.py <result_folder> <total_epochs>

    result_folder : path to the saved run directory, or just its name
                    (assumed to live in ../results/saved/ if not an absolute path)
    total_epochs  : the new total epoch count (must be > epochs already done)

Example:
    python resume_train.py ../results/saved/2026_05_26__01_46_01_diffusion_unet_... 100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import ast
import logging
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import partial

from slitless.networks.unet import UNet
from denoising_diffusion_pytorch import Unet as DiffusionUnet
from slitless.data_loader import (BasicDataset, meas_transform, param_transform)
from slitless.measure import nmse_torch, combine_losses, cycle_loss
from slitless.train import train_net
from slitless.evaluate import plot_recons, plot_val_stats, eval_snrlist, meanest_errcalc
from slitless.plotting import barplot_group, scatter_hexbin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_result_folder(arg):
    """Return absolute path to the result folder given a name or path."""
    if os.path.isabs(arg):
        return os.path.normpath(arg)
    # Check if it already is a relative path that resolves
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    # Assume it's just the folder name inside ../results/saved/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(scripts_dir, '..', 'results', 'saved', arg))
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f'Result folder not found: {arg}')


def parse_summary(summary_path):
    config = {}
    with open(summary_path) as f:
        for line in f:
            stripped = line.strip()
            if ' = ' in stripped and not stripped.startswith('#'):
                key, _, val = stripped.partition(' = ')
                config[key.strip()] = val.strip()
    return config


def detect_model(folder_name):
    if 'diffusion_unet' in folder_name:
        return 'diffusion_unet'
    # Older runs had no model prefix; newer unet runs have _unet_ after timestamp
    # Timestamp pattern: YYYY_MM_DD__HH_MM_SS
    m = re.match(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_(.+?)_NF_', folder_name)
    if m and m.group(1) == 'unet':
        return 'unet'
    return 'unet'  # safe default for old runs


def resolve_dataset_path(raw_path, result_folder):
    """
    summary.txt stores the dataset path as it was seen from python/slitless/
    (either absolute or relative to that dir).  Resolve it to an absolute path.
    """
    if os.path.isabs(raw_path) and os.path.isdir(raw_path):
        return raw_path
    # Infer python/slitless/ from the result folder location:
    # result_folder = .../python/results/saved/{name}  →  3 levels up = .../python/
    python_dir = os.path.normpath(os.path.join(result_folder, '..', '..', '..'))
    slitless_dir = os.path.join(python_dir, 'slitless')
    resolved = os.path.normpath(os.path.join(slitless_dir, raw_path))
    if os.path.isdir(resolved):
        return resolved
    raise FileNotFoundError(
        f'Cannot locate dataset path "{raw_path}". '
        f'Tried: {resolved}'
    )


def parse_training_log(log_path, modnum=5):
    """
    Recover training metrics from output.log.
    Returns lists/arrays matching the format returned by train_net.
    """
    epoch_trainloss = {}
    valloss = []
    train_ssim, val_ssim = [], []
    train_rmse, val_rmse = [], []

    with open(log_path) as f:
        for line in f:
            msg = line.split('; ', 1)[-1].strip()

            m = re.match(r'\[(\d+)\] Train loss: ([\d.]+)', msg)
            if m:
                epoch_trainloss[int(m.group(1))] = float(m.group(2))
                continue

            if re.match(r'Validation loss: ([\d.]+)', msg):
                valloss.append(float(re.match(r'Validation loss: ([\d.]+)', msg).group(1)))
                continue

            if 'Train SSIM' in msg:
                vals = [float(v) for v in re.findall(r"'([\d.]+)'", msg)]
                if vals:
                    train_ssim.append(vals)
                continue

            if re.search(r"'Val SSIM", msg):
                vals = [float(v) for v in re.findall(r"'([\d.]+)'", msg)]
                if vals:
                    val_ssim.append(vals)
                continue

            if 'Train RMSE' in msg:
                vals = [float(v) for v in re.findall(r"'([\d.]+)'", msg)]
                if vals:
                    train_rmse.append(vals)
                continue

            if re.search(r"'Val RMSE", msg):
                vals = [float(v) for v in re.findall(r"'([\d.]+)'", msg)]
                if vals:
                    val_rmse.append(vals)
                continue

    # train_loss_over_epochs only stores the loss at modnum epochs
    modnum_epochs = sorted(e for e in epoch_trainloss if e % modnum == 0)
    trainloss = [epoch_trainloss[e] for e in modnum_epochs]

    return (trainloss, valloss,
            np.array(train_ssim) if train_ssim else np.empty((0, 3)),
            np.array(val_ssim)   if val_ssim   else np.empty((0, 3)),
            np.array(train_rmse) if train_rmse else np.empty((0, 3)),
            np.array(val_rmse)   if val_rmse   else np.empty((0, 3)))


def save_curve_plots(trainloss, valloss, train_ssim_eps, val_ssim_eps,
                     train_rmse_eps, val_rmse_eps, save_dir):
    plt.figure()
    plt.semilogy(trainloss, label='Training Loss')
    plt.semilogy(valloss, label='Validation Loss')
    plt.title('Convergence Plot')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'convergence_plot.png'))
    plt.close()

    if train_ssim_eps.ndim == 2 and train_ssim_eps.shape[1] >= 3:
        plt.figure()
        plt.title('SSIMs vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('SSIM')
        plt.plot(train_ssim_eps[:, 0], marker='$t$', color='navy',        label='Int train')
        plt.plot(val_ssim_eps[:, 0],   marker='$v$', color='darkturquoise')
        plt.plot(train_ssim_eps[:, 1], marker='$t$', color='darkred',     label='Vel train')
        plt.plot(val_ssim_eps[:, 1],   marker='$v$', color='tomato')
        plt.plot(train_ssim_eps[:, 2], marker='$t$', color='darkgreen',   label='Width train')
        plt.plot(val_ssim_eps[:, 2],   marker='$v$', color='lime')
        plt.grid(which='both', axis='both')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'ssims_vs_epochs.png'))
        plt.close()

    if train_rmse_eps.ndim == 2 and train_rmse_eps.shape[1] >= 3:
        plt.figure()
        plt.title('RMSEs vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('RMSE')
        plt.plot(train_rmse_eps[:, 0], marker='$t$', color='navy',        label='Int train')
        plt.plot(val_rmse_eps[:, 0],   marker='$v$', color='darkturquoise')
        plt.plot(train_rmse_eps[:, 1], marker='$t$', color='darkred',     label='Vel train')
        plt.plot(val_rmse_eps[:, 1],   marker='$v$', color='tomato')
        plt.plot(train_rmse_eps[:, 2], marker='$t$', color='darkgreen',   label='Width train')
        plt.plot(val_rmse_eps[:, 2],   marker='$v$', color='lime')
        plt.grid(which='both', axis='both')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'rmses_vs_epochs.png'))
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Resume a slitless training run.')
    parser.add_argument('result_folder', help='Path or name of the result directory')
    parser.add_argument('total_epochs', type=int, help='New total epoch count (> epochs already done)')
    args = parser.parse_args()

    result_folder = resolve_result_folder(args.result_folder)
    total_epochs  = args.total_epochs
    folder_name   = os.path.basename(result_folder)

    # ---- load config from summary.txt ----
    summary_path = os.path.join(result_folder, 'summary.txt')
    cfg = parse_summary(summary_path)

    NUM_FILT     = int(cfg['Number of starting filters'])
    numlayers    = int(cfg['Number of Layers'])
    numdetectors = int(cfg['Number of Detectors'])
    BILINEAR     = cfg.get('Bilinear Interpolation for Upsampling (if False, use transposed convolution)', 'True') == 'True'
    ksizes       = ast.literal_eval(cfg.get('Kernel Size', '[(3, 1)]'))
    OUTCH        = cfg['Output Channels']
    out_channels = 3 if OUTCH == 'all' else 1
    OPTIMIZER    = cfg['Optimizer']
    LOSS         = cfg['Loss']
    CYC_LOSS     = cfg.get('Cycle Loss', 'False') == 'True'
    cyc_lam      = float(cfg.get('Cycle Loss Lam', '1'))
    LR           = float(cfg['Learning Rate'])
    orig_epochs  = int(cfg['Num Epochs'])
    BATCH_SIZE   = int(cfg['Training Batch Size'])
    noise_str, dbsnr_str = cfg['Noise Model / dbsnr'].split(' / ')
    noise_model  = None if noise_str == 'None' else noise_str
    dbsnr        = None if dbsnr_str == 'None' else int(float(dbsnr_str))

    raw_dataset_path = cfg['Dataset Path']
    dataset_path = resolve_dataset_path(raw_dataset_path, result_folder)
    dset_root    = os.path.dirname(dataset_path.rstrip('/'))
    dset_stats   = np.load(os.path.join(dset_root, 'norm_stats.npy'), allow_pickle=True).item()

    MODEL = detect_model(folder_name)

    # ---- determine how many epochs are already done ----
    checkpoint_path = os.path.join(result_folder, 'checkpoint.pth')
    has_checkpoint  = os.path.isfile(checkpoint_path)

    if has_checkpoint:
        print('Found checkpoint.pth — loading optimizer state and metrics.')
        ckpt          = torch.load(checkpoint_path, map_location='cpu')
        epochs_done   = ckpt['epoch']
        prev_best_val = ckpt['best_valloss']
        prev_trainloss     = ckpt['train_loss_over_epochs']
        prev_valloss       = ckpt['val_loss_over_epochs']
        prev_train_ssim    = np.array(ckpt['train_ssim_over_epochs'])
        prev_val_ssim      = np.array(ckpt['val_ssim_over_epochs'])
        prev_train_rmse    = np.array(ckpt['train_rmse_over_epochs'])
        prev_val_rmse      = np.array(ckpt['val_rmse_over_epochs'])
    else:
        print('No checkpoint.pth found — parsing output.log to recover metrics.')
        log_path = os.path.join(result_folder, 'output.log')
        (prev_trainloss, prev_valloss,
         prev_train_ssim, prev_val_ssim,
         prev_train_rmse, prev_val_rmse) = parse_training_log(log_path)
        epochs_done   = orig_epochs
        prev_best_val = min(prev_valloss) if prev_valloss else 1e6

    remaining = total_epochs - epochs_done
    if remaining <= 0:
        print(f'Already at {epochs_done} epochs (requested {total_epochs}). Nothing to do.')
        sys.exit(0)

    print(f'Resuming from epoch {epochs_done} → {total_epochs} ({remaining} more epochs).')

    # ---- set up logging (append to existing log) ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(result_folder, 'output.log'), mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s; %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'=== Resuming: {folder_name} | epochs {epochs_done} -> {total_epochs} ===')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # ---- datasets ----
    _meas_tf  = partial(meas_transform,  stats=dset_stats)
    _param_tf = partial(param_transform, stats=dset_stats)
    loader_kw = dict(num_workers=4, persistent_workers=True, pin_memory=True)
    trainset  = BasicDataset(data_dir=dataset_path, transform=_meas_tf,
                             target_transform=_param_tf, fold='train',
                             dbsnr=dbsnr, noise_model=noise_model,
                             numdetectors=numdetectors)
    valset    = BasicDataset(data_dir=dataset_path, transform=_meas_tf,
                             target_transform=_param_tf, fold='val',
                             dbsnr=dbsnr, noise_model=noise_model,
                             numdetectors=numdetectors)
    testset   = BasicDataset(data_dir=dataset_path, transform=_meas_tf,
                             target_transform=_param_tf, fold='test',
                             dbsnr=dbsnr, noise_model=noise_model,
                             numdetectors=numdetectors)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  **loader_kw)
    valloader   = DataLoader(valset,   batch_size=64,         shuffle=True,  **loader_kw)
    testloader  = DataLoader(testset,  batch_size=64,         shuffle=True,  **loader_kw)

    # ---- build model ----
    if MODEL == 'unet':
        net = UNet(in_channels=numdetectors, out_channels=out_channels,
                   numlayers=numlayers, outch_type=OUTCH,
                   start_filters=NUM_FILT, bilinear=BILINEAR,
                   ksizes=ksizes, residual=False).to(device)
    elif MODEL == 'diffusion_unet':
        net = DiffusionUnet(dim=NUM_FILT, channels=numdetectors,
                            out_dim=out_channels, dim_mults=(1, 2, 4, 8),
                            flash_attn=False).to(device)
        net.outch_type = OUTCH
        _fwd = net.forward
        net.forward = lambda x: _fwd(x, torch.zeros(x.shape[0], device=x.device))

    # load weights from the last-epoch model file
    last_model_glob = [f for f in os.listdir(result_folder)
                       if f.startswith('nf_') and f.endswith('.pth')]
    if last_model_glob:
        weights_path = os.path.join(result_folder, sorted(last_model_glob)[-1])
        print(f'Loading weights from {os.path.basename(weights_path)}')
        net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        raise FileNotFoundError('No nf_*.pth weight file found in result folder.')

    # ---- optimizer ----
    if OPTIMIZER == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=LR)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    if has_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # ---- loss ----
    if LOSS == 'MSE':
        losses_param = [nn.MSELoss()]
    elif LOSS == 'L1':
        losses_param = [nn.L1Loss()]
    elif LOSS == 'NMSE':
        losses_param = [nmse_torch]
    else:
        losses_param = []

    losses_meas = [cycle_loss] if CYC_LOSS else []
    lam = [1, cyc_lam] if CYC_LOSS else [1]

    criterion = combine_losses(
        losses_param=losses_param,
        losses_meas=losses_meas,
        lam=lam,
        outch_type=net.outch_type,
    )

    # ---- train remaining epochs ----
    try:
        t0 = time.time()
        (new_trainloss, new_valloss,
         new_train_ssim, new_val_ssim,
         new_train_rmse, new_val_rmse, net) = train_net(
            net=net,
            device=device,
            trainloader=trainloader,
            otf=None,
            valloader=valloader,
            epochs=remaining,
            optimizer=optimizer,
            criterion=criterion,
            save_dir=result_folder,
            start_epoch=epochs_done,
            prev_best_valloss=prev_best_val,
        )
        extra_time = datetime.timedelta(seconds=int(time.time() - t0))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(result_folder, 'INTERRUPTED.pth'))
        sys.exit(0)

    # save new final weights
    torch.save(net.state_dict(),
               os.path.join(result_folder, f'nf_{NUM_FILT}_LR_{LR}_EP_{total_epochs}.pth'))

    # ---- stitch metrics ----
    all_trainloss  = list(prev_trainloss) + list(new_trainloss)
    all_valloss    = list(prev_valloss)   + list(new_valloss)

    def vstack_safe(a, b):
        if a.size == 0:
            return b
        if b.size == 0:
            return a
        return np.vstack([a, b])

    all_train_ssim = vstack_safe(prev_train_ssim, new_train_ssim)
    all_val_ssim   = vstack_safe(prev_val_ssim,   new_val_ssim)
    all_train_rmse = vstack_safe(prev_train_rmse, new_train_rmse)
    all_val_rmse   = vstack_safe(prev_val_rmse,   new_val_rmse)

    # ---- regenerate curve plots ----
    save_curve_plots(all_trainloss, all_valloss,
                     all_train_ssim, all_val_ssim,
                     all_train_rmse, all_val_rmse,
                     result_folder)

    # ---- evaluation on best model ----
    net.load_state_dict(torch.load(os.path.join(result_folder, 'best_model.pth'),
                                   map_location=device))
    net.eval()

    savedir = result_folder + '/'
    ssims, rmses, yvec, outvec = plot_val_stats(net, testloader, savedir, stats=dset_stats)
    if net.outch_type == 'all':
        scatter_hexbin(yvec, outvec, method_name=folder_name, save=True,
                       savepath=os.path.join(result_folder, 'hexbin_scatter.png'), show=False)
    plot_recons(net, testloader, numim=32,
                savedir=os.path.join(result_folder, 'figures') + '/',
                denormalize=True, stats=dset_stats)
    dbsnr_l = [10, 20, 30, None]
    ssims_l, rmses_l = eval_snrlist(dbsnr_list=dbsnr_l, noise_model=noise_model,
                                    fold='test', data_dir=dataset_path, net=net)
    labels_gr = ['int', 'vel', 'width'] if net.outch_type == 'all' else [net.outch_type]
    barplot_group(ssims_l.mean(axis=1).swapaxes(0, 1),
                  labels_gr=labels_gr, labels_mem=[str(j) for j in dbsnr_l],
                  ylabel='SSIM', title='SSIM vs dBsnr',
                  savedir=os.path.join(result_folder, 'snr_barplot.png'))
    est_bias = np.mean(outvec - yvec, axis=1)
    est_std  = np.std(outvec - yvec, axis=1)
    np.save(os.path.join(result_folder, 'ssims_l.npy'), ssims_l)
    np.save(os.path.join(result_folder, 'rmses_l.npy'), rmses_l)

    # ---- update summary.txt ----
    resume_note = [
        f'\n############## Resume ############## \n',
        f'Resumed from epoch {epochs_done} to {total_epochs} \n',
        f'Additional Training Time: {str(extra_time)} \n',
        f'Final Validation Loss: {all_valloss[-1]:.7f} \n',
        f'Minimum Validation Loss: {np.min(all_valloss):.7f} \n',
        f'Final Training Loss: {all_trainloss[-1]:.7f} \n',
        f'Minimum Training Loss: {np.min(all_trainloss):.7f} \n',
    ]
    if net.outch_type == 'all':
        resume_note += [
            'SSIM Mean+/-Std: i: {:.3f}+/-{:.3f}   v: {:.3f}+/-{:.3f}   w: {:.3f}+/-{:.3f} \n'.format(
                ssims[:,0].mean(), ssims[:,0].std(), ssims[:,1].mean(), ssims[:,1].std(),
                ssims[:,2].mean(), ssims[:,2].std()),
            'RMSE Mean+/-Std [erg/cm2/s/sr, km/s, km/s]: i: {:.2e}+/-{:.2e}   v: {:.2f}+/-{:.2f}   w: {:.2f}+/-{:.2f} \n'.format(
                rmses[:,0].mean(), rmses[:,0].std(), rmses[:,1].mean(), rmses[:,1].std(),
                rmses[:,2].mean(), rmses[:,2].std()),
            'Bias+/-Std [erg/cm2/s/sr, km/s, km/s]: i: {:.2e}+/-{:.2e}   v: {:.2f}+/-{:.2f}   w: {:.2f}+/-{:.2f} \n'.format(
                est_bias[0], est_std[0], est_bias[1], est_std[1], est_bias[2], est_std[2]),
        ]

    with open(os.path.join(result_folder, 'summary.txt'), 'a') as f:
        for line in resume_note:
            f.write(line)

    print(f'\nDone. Results updated in {result_folder}')


if __name__ == '__main__':
    main()
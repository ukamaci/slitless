import logging, glob, os
from collections import defaultdict
from os import listdir
from os.path import splitext

import numpy as np
import torch, time
from slitless.forward import add_noise, forward_op, forward_op_torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool

WAVELENGTH = 195.117937907451
SPEEDOFLIGHT = 299792.458

def _log(x, eps=1.0):
    return torch.log(x + eps) if torch.is_tensor(x) else np.log(x + eps)

def _exp(x):
    return torch.exp(x) if torch.is_tensor(x) else np.exp(x)

# mode='log_zscore': z-score of log(x + 1.0); requires norm_stats.npy keys
#   int_log_mean/std, meas_log_mean/std (see scripts/update_v5_log_stats.py).
# mode='linear': divide intensity by 6000 (legacy). Vel/width always z-scored.

def meas_transform(meas, stats=None, mode='log_zscore'):
    if mode == 'linear':
        return meas/6000
    elif mode == 'log_zscore':
        if stats is None:
            raise ValueError("stats required for log_zscore mode")
        # Noisy measurements can fall below the physical floor (clean meas_min)
        # and even below -1, which would make log(meas+1) NaN. Clamp to meas_min
        # first; a no-op at high SNR where clean meas >= meas_min.
        floor = stats['meas_min']
        meas = meas.clamp(min=floor) if torch.is_tensor(meas) else np.maximum(meas, floor)
        return (_log(meas) - stats['meas_log_mean']) / stats['meas_log_std']
    raise ValueError(f"Unknown mode: {mode!r}")

def meas_inv_transform(meas, stats=None, mode='log_zscore'):
    if mode == 'linear':
        return meas*6000
    elif mode == 'log_zscore':
        if stats is None:
            raise ValueError("stats required for log_zscore mode")
        return _exp(meas * stats['meas_log_std'] + stats['meas_log_mean']) - 1.0
    raise ValueError(f"Unknown mode: {mode!r}")

def param_transform(params, stats=None, mode='log_zscore'):
    if stats is None:
        raise ValueError("stats required for param_transform")
    if mode == 'linear':
        params[0] = params[0] / 6000
    elif mode == 'log_zscore':
        params[0] = (_log(params[0]) - stats['int_log_mean']) / stats['int_log_std']
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    params[1] = (params[1] - stats['vel_mean']) / stats['vel_std']
    params[2] = (params[2] - stats['width_mean']) / stats['width_std']
    return params

def param_inv_transform(params, w_kms=False, stats=None, mode='log_zscore'):
    if stats is None:
        raise ValueError("stats required for param_inv_transform")
    if mode == 'linear':
        params[...,0,:,:] = params[...,0,:,:] * 6000
    elif mode == 'log_zscore':
        params[...,0,:,:] = _exp(
            params[...,0,:,:] * stats['int_log_std'] + stats['int_log_mean']
        ) - 1.0
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    params[...,1,:,:] = stats['vel_std'] * params[...,1,:,:] + stats['vel_mean']
    params[...,2,:,:] = stats['width_std'] * params[...,2,:,:] + stats['width_mean']
    if w_kms: # conversion from A to km/s
        params[...,2,:,:] *= SPEEDOFLIGHT/WAVELENGTH
    return params

# Fixed seed for the partition (dataset-size / no-leakage) ablation. Shared with
# the denoising_diffusion_pytorch repo so a given (partno, partnum) selects the
# identical scans in both codebases.
PARTITION_SEED = 42


def _scan_id(path):
    """Scan id from a 'data_<date>_<time>_<patchno>.npy' filename, i.e. the name
    with the trailing '_<patchno>' (and '.npy') stripped. dset_v6 patches are
    64x64 crops of larger EIS scans; grouping by scan id keeps every patch of a
    scan in one partition (no data leakage across partitions)."""
    return os.path.basename(path)[:-4].rsplit('_', 1)[0]


def partition_files(files, partno, partnum, seed=PARTITION_SEED):
    """Return the subset of `files` belonging to partition `partno` of `partnum`.

    Patches are grouped by scan id so a scan never straddles two partitions.
    Scan ids are sorted (canonical, repo/OS-independent order) then seeded-
    shuffled — the shuffle breaks the temporal ordering encoded in the scan
    date, spreading scan times homogeneously across partitions. Whole scans are
    then greedily assigned to the currently-smallest partition (by patch count)
    so partition sizes stay tightly balanced. Reproducible and identical across
    repos given the same files, seed, partno and partnum.
    """
    assert isinstance(partno, int) and isinstance(partnum, int), 'partno/partnum must be ints'
    assert partnum >= 1, f'partnum must be >= 1, got {partnum}'
    assert 1 <= partno <= partnum, f'partno must be in 1..{partnum}, got {partno}'
    if partnum == 1:
        return list(files)

    groups = defaultdict(list)
    for f in files:
        groups[_scan_id(f)].append(f)

    scan_ids = sorted(groups)
    np.random.default_rng(seed).shuffle(scan_ids)

    loads = [0] * partnum
    keep = []
    for sid in scan_ids:
        i = min(range(partnum), key=lambda j: loads[j])   # currently-smallest partition
        loads[i] += len(groups[sid])
        if i == partno - 1:
            keep.extend(groups[sid])
    return sorted(keep)


class BasicDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None, noise_model=None, numdetectors=3,
        partno=1, partnum=1):
        self.data_dir = data_dir
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr
        self.noise_model = noise_model
        self.numdetectors = numdetectors
        stats_path = os.path.normpath(os.path.join(data_dir, '..', 'norm_stats.npy'))
        self.stats = np.load(stats_path, allow_pickle=True).item() if os.path.exists(stats_path) else None

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
        # Dataset-size / no-leakage ablation: split the *training* set into
        # partnum leakage-free partitions (whole scans never straddle a
        # partition) and keep partition partno. Identical selection to the ddpm
        # repo's EISDataset. val/test folds are always used in full.
        if self.train:
            self.files = partition_files(self.files, partno, partnum)
        # Stack into two contiguous arrays (instead of a list of N tuples) so
        # DataLoader workers share memory cleanly via copy-on-write — Python
        # refcounts touching per-sample tuples otherwise break COW and balloon
        # memory across many workers.
        meas_list, params_list = [], []
        for file in self.files:
            data = np.load(file, allow_pickle=True).item()
            params_list.append(np.stack([data['int'], data['vel'], data['width']]))
            meas_list.append(np.stack([data['meas_0'], data['meas_-1'],
                data['meas_1'], data['meas_-2'], data['meas_2']])[:numdetectors])
        self.meas = np.stack(meas_list).astype(np.float32)      # (N, K, H, W)
        self.params = np.stack(params_list).astype(np.float32)  # (N, 3, H, W)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Copy so the in-place transforms below don't mutate the shared base
        # arrays — critical when DataLoader uses persistent_workers=True.
        meas = self.meas[idx].copy()
        params = self.params[idx].copy()

        # Noise must be applied in the raw photon-count domain (Poisson is
        # ill-defined on negative/normalized values), so we noise first then
        # transform. Under linear scaling this is mathematically equivalent
        # to the previous transform-then-noise ordering.
        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None,
            noise_model=self.noise_model)

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

class OntheflyDataset(Dataset):
    def __init__(self, data_dir, fold='train', transform=None,
        target_transform=None, dbsnr=None, trpart=None):
        self.data_dir = data_dir
        self.data = []
        self.train = False
        self.val = False
        self.transform = transform
        self.target_transform = target_transform
        self.dbsnr = dbsnr
        self.trpart = trpart

        if fold == 'train':
            self.train = True
        elif fold == 'val':
            self.val = True

        self.task_dir = os.path.join(self.data_dir, fold)

        if trpart is None:
            file = glob.glob(self.task_dir+'/otf*npy')[0]
        else:
            file = glob.glob(self.task_dir+'/otf_p{}*npy'.format(trpart))[0]
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
        ind = 1000

        for i in range(len(intens) // ind):
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

        meas = add_noise(meas, dbsnr=self.dbsnr, no_noise=self.dbsnr==None, 
            model='Gaussian')

        if self.transform is not None:
            meas = self.transform(meas)

        if self.target_transform is not None:
            params = self.target_transform(params)

        return meas, params

def dloadertesting():
    dataset_path = glob.glob('../../data/datasets/dset6*')[0]
    t0 = time.time()
    trainset = OntheflyDataset(data_dir=dataset_path, fold='train', dbsnr=35)
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

# def eis_data_plotter(folder):
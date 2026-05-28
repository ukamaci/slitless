import os
import glob
import numpy as np
import torch
from slitless.forward import forward_op_torch

SRC = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v5/data'
DST = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data'
INT_MAX_THRESH = 8000
LOG_EPS = 1.0

DISPERSION_SCALE = 0.022275          # Å/pixel
MID_WAVELENGTH   = 195.119           # Å  (detector array centre)
REST_WAVELENGTH  = 195.117937907451  # Å  (Fe XII rest wavelength)
SPEEDOFLIGHT     = 299792.458        # km/s

MEAS_KEYS   = ['meas_0', 'meas_-1', 'meas_1', 'meas_-2', 'meas_2']
MEAS_ORDERS = [0, -1, 1, -2, 2]
BATCH_SIZE  = 256


def simulate_meas_batch(int_batch, vel_batch, width_batch, device):
    """Simulate noiseless single-Gaussian measurements for a batch of patches."""
    # Match Imager.topix(): pixel shift is relative to detector mid_wavelength
    actual_wave = REST_WAVELENGTH * (1 + vel_batch / SPEEDOFLIGHT)
    vel_pix   = (actual_wave - MID_WAVELENGTH) / DISPERSION_SCALE
    width_pix = width_batch / DISPERSION_SCALE

    t_int   = torch.from_numpy(int_batch).to(device=device, dtype=torch.float)
    t_dop   = torch.from_numpy(vel_pix  ).to(device=device, dtype=torch.float)
    t_wid   = torch.from_numpy(width_pix).to(device=device, dtype=torch.float)

    with torch.no_grad():
        meas = forward_op_torch(
            true_intensity=t_int,
            true_doppler=t_dop,
            true_linewidth=t_wid,
            pixelated=True,
            spectral_orders=MEAS_ORDERS,
            device=device,
        )  # (B, 5, H, W)
    return meas.cpu().numpy()


def generate_split(split, device):
    src_dir = os.path.join(SRC, split)
    dst_dir = os.path.join(DST, split)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, '*.npy')))

    # --- filter pass ---
    kept_data = []
    skipped = 0
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        if data['int'].max() > INT_MAX_THRESH:
            skipped += 1
            continue
        kept_data.append((os.path.basename(f), data['int'], data['vel'], data['width']))

    print(f'{split}: kept {len(kept_data)}, discarded {skipped} '
          f'({100*skipped/(len(kept_data)+skipped):.1f}%)')

    # --- simulate measurements in batches ---
    names  = [d[0] for d in kept_data]
    ints   = np.stack([d[1] for d in kept_data])   # (N, H, W)
    vels   = np.stack([d[2] for d in kept_data])
    widths = np.stack([d[3] for d in kept_data])

    all_meas = []
    for start in range(0, len(ints), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_meas = simulate_meas_batch(ints[start:end], vels[start:end],
                                         widths[start:end], device)
        all_meas.append(batch_meas)
    all_meas = np.concatenate(all_meas, axis=0)   # (N, 5, H, W)

    # --- write output files ---
    for i, name in enumerate(names):
        out = {
            'int':   ints[i],
            'vel':   vels[i],
            'width': widths[i],
        }
        for j, key in enumerate(MEAS_KEYS):
            out[key] = all_meas[i, j]
        np.save(os.path.join(dst_dir, name), out)

    return len(kept_data)


def compute_norm_stats():
    train_dir = os.path.join(DST, 'train')
    files = sorted(glob.glob(os.path.join(train_dir, '*.npy')))

    int_vals, vel_vals, width_vals, meas_vals = [], [], [], []
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        int_vals.append(data['int'].ravel())
        vel_vals.append(data['vel'].ravel())
        width_vals.append(data['width'].ravel())
        meas_vals.append(np.stack([data[k] for k in MEAS_KEYS]).ravel())

    int_all   = np.concatenate(int_vals)
    vel_all   = np.concatenate(vel_vals)
    width_all = np.concatenate(width_vals)
    meas_all  = np.concatenate(meas_vals)

    log_int  = np.log(int_all  + LOG_EPS)
    log_meas = np.log(meas_all + LOG_EPS)

    stats = {
        'int_min':       float(int_all.min()),
        'int_max':       float(int_all.max()),
        'int_mean':      float(int_all.mean()),
        'int_log_mean':  float(log_int.mean()),
        'int_log_std':   float(log_int.std()),
        'vel_mean':      float(vel_all.mean()),
        'vel_std':       float(vel_all.std()),
        'width_mean':    float(width_all.mean()),
        'width_std':     float(width_all.std()),
        'meas_min':      float(meas_all.min()),
        'meas_mean':     float(meas_all.mean()),
        'meas_log_mean': float(log_meas.mean()),
        'meas_log_std':  float(log_meas.std()),
        'log_eps':       LOG_EPS,
    }
    return stats


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for split in ['train', 'val', 'test']:
        generate_split(split, device)

    print('\nComputing norm stats from training split...')
    stats = compute_norm_stats()
    for k, v in stats.items():
        print(f'  {k}: {v}')

    out_path = os.path.join(os.path.dirname(DST), 'norm_stats.npy')
    np.save(out_path, stats)
    print(f'\nSaved norm_stats to {out_path}')

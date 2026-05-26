import os
import glob
import numpy as np

SRC = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v5/data'
DST = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data'
INT_MAX_THRESH = 8000
LOG_EPS = 1.0


MEAS_KEYS = ['meas_0', 'meas_-1', 'meas_1', 'meas_-2', 'meas_2']


def generate_split(split):
    src_dir = os.path.join(SRC, split)
    dst_dir = os.path.join(DST, split)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, '*.npy')))
    kept, skipped = 0, 0
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        if data['int'].max() > INT_MAX_THRESH:
            skipped += 1
            continue
        out = {
            'int': data['int'], 'vel': data['vel'], 'width': data['width'],
            **{k: data[k] for k in MEAS_KEYS},
        }
        np.save(os.path.join(dst_dir, os.path.basename(f)), out)
        kept += 1

    print(f'{split}: kept {kept}, discarded {skipped} ({100*skipped/(kept+skipped):.1f}%)')
    return kept


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
    for split in ['train', 'val', 'test']:
        generate_split(split)

    print('\nComputing norm stats from training split...')
    stats = compute_norm_stats()
    for k, v in stats.items():
        print(f'  {k}: {v}')

    out_path = os.path.join(os.path.dirname(DST), 'norm_stats.npy')
    np.save(out_path, stats)
    print(f'\nSaved norm_stats to {out_path}')

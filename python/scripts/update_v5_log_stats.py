"""
Add log-space normalization statistics for intensity and measurements to
dset_v5/norm_stats.npy (used by data_loader.py in 'log_zscore' mode).

Computes mean/std of log(x + LOG_EPS) over the entire training split.
LOG_EPS must match the eps default in meas_transform/param_transform (1.0).
"""
import os, glob
import numpy as np
from tqdm import tqdm

STATS_PATH = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v5/norm_stats.npy'
TRAIN_DIR = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v5/data/train/'
LOG_EPS = 1.0  # must match eps default in meas_transform/param_transform

def main():
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    print('Existing stats:')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'data*.npy')))
    print(f'\nFound {len(files)} training files.')

    # Streaming mean/var via Welford-style accumulation to keep memory low.
    n_int = 0; m_int = 0.0; m2_int = 0.0
    n_meas = 0; m_meas = 0.0; m2_meas = 0.0
    int_min, int_max = np.inf, -np.inf
    meas_min, meas_max = np.inf, -np.inf

    for fp in tqdm(files):
        d = np.load(fp, allow_pickle=True).item()
        lint = np.log(d['int'] + LOG_EPS).ravel()
        lmeas = np.log(d['meas_0'] + LOG_EPS).ravel()

        int_min = min(int_min, float(d['int'].min()))
        int_max = max(int_max, float(d['int'].max()))
        meas_min = min(meas_min, float(d['meas_0'].min()))
        meas_max = max(meas_max, float(d['meas_0'].max()))

        for arr, (n, m, m2) in [(lint, (n_int, m_int, m2_int)),
                                (lmeas, (n_meas, m_meas, m2_meas))]:
            pass  # handled below explicitly to avoid tuple-rebind cost

        # intensity
        k = lint.size
        delta = lint - m_int
        m_int += delta.sum() / (n_int + k)
        m2_int += (delta * (lint - m_int)).sum()
        n_int += k

        # meas_0
        k = lmeas.size
        delta = lmeas - m_meas
        m_meas += delta.sum() / (n_meas + k)
        m2_meas += (delta * (lmeas - m_meas)).sum()
        n_meas += k

    int_log_mean = m_int
    int_log_std = np.sqrt(m2_int / n_int)
    meas_log_mean = m_meas
    meas_log_std = np.sqrt(m2_meas / n_meas)

    stats.update({
        'int_log_mean': float(int_log_mean),
        'int_log_std': float(int_log_std),
        'meas_log_mean': float(meas_log_mean),
        'meas_log_std': float(meas_log_std),
        'int_min': int_min,
        'meas_min': meas_min,
        'log_eps': LOG_EPS,
    })

    print('\nUpdated stats:')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    np.save(STATS_PATH, stats)
    print(f'\nSaved to {STATS_PATH}')

if __name__ == '__main__':
    main()

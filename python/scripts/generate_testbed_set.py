import os
import glob
import numpy as np
from slitless.forward import forward_op_tomo_3d

# --- config ---
DSET_VERSION = 6        # 5 or 6
NUM_DATA     = 10
SPLIT = 'val'
# --------------

DSET_ROOTS = {
    5: '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v5/data',
    6: '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data',
}
OUT_DIR   = '/home/kamo/resources/slitless/data/datasets/baseline/'
MEAS_KEYS = ['meas_0', 'meas_-1', 'meas_1', 'meas_-2', 'meas_2']


def generate_consolidated_set(num_patches=50, split='train', dset_version=5, output_filename=None):
    if output_filename is None:
        output_filename = f'eis_{split}_{num_patches}_dsetv{dset_version}.npy'

    split_dir = os.path.join(DSET_ROOTS[dset_version], split)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, output_filename)

    print(f"Scanning {split_dir}...")
    files = glob.glob(os.path.join(split_dir, '*.npy'))
    files.sort()

    np.random.seed(42)
    selected_files = np.random.choice(files, num_patches, replace=False)

    param3d_l, meas_l = [], []
    meas2_l, cube_l = [], []

    print(f"Stacking {num_patches} randomly selected patches...")
    for f in selected_files:
        data_i = np.load(f, allow_pickle=True).item()
        meas_l.append(np.stack([data_i[k] for k in MEAS_KEYS]))
        param3d_l.append(np.stack([data_i['int'], data_i['vel'], data_i['width']]))
        if dset_version == 5:
            cube_l.append(data_i['datacube'])
            meas2_l.append(forward_op_tomo_3d(data_i['datacube'].transpose(2, 0, 1), orders=[0, -1, 1, -2, 2]))

    out_data = {
        'meas': np.array(meas_l),
        'param3d': np.array(param3d_l),
    }
    if dset_version == 5:
        out_data['meas_damped'] = np.array(meas2_l)
        out_data['datacube'] = np.array(cube_l)

    np.save(out_path, out_data)
    print(f"\nSuccessfully saved to {out_path}")
    print(f"  param3d shape: {out_data['param3d'].shape}")
    print(f"  meas shape:    {out_data['meas'].shape}")
    if dset_version == 5:
        print(f"  meas_damped shape: {out_data['meas_damped'].shape}")
        print(f"  datacube shape:    {out_data['datacube'].shape}")


if __name__ == '__main__':
    generate_consolidated_set(num_patches=NUM_DATA, split=SPLIT, dset_version=DSET_VERSION)

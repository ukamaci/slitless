import glob, os
import numpy as np
from tqdm import tqdm

pathdir = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v3/data/'
paths = [pathdir + 'train/', pathdir + 'val/', pathdir + 'test/']
DISP_SCALE = 13.5*1.65/1000


for pth in tqdm(paths):
    all_files = glob.glob(pth + '*.npy')

    all_counter = 0
    remove_counter = 0
    for f in tqdm(all_files):
        all_counter += 1
        d = np.load(f, allow_pickle=True).item()

        mask = (
            (d['int'] > 100.0) & 
            (d['int'] < 6000.0) & 
            (np.abs(d['vel']) < 68.5) & 
            (d['width'] > 0.022)
        )
        if mask.sum() < d['int'].size:
            os.remove(f)
            remove_counter += 1
            continue
        
        d['meas_0'] *= DISP_SCALE
        d['meas_1'] *= DISP_SCALE
        d['meas_2'] *= DISP_SCALE
        d['meas_-1'] *= DISP_SCALE
        d['meas_-2'] *= DISP_SCALE

        np.save(f, d)

    print(f'In {pth} {remove_counter}/{all_counter}' +
    f' ({remove_counter/all_counter*100:.1f} %) of the data points have been removed')
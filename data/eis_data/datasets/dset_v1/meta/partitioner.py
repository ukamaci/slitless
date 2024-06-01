import glob, shutil
import numpy as np

files = glob.glob('./*.png')

filesday = np.array([a.split('_')[1] for a in files])
files_idx = np.argsort(filesday)
filesday = np.sort(filesday)

fd, idx, ct = np.unique(filesday, return_index=True, 
    return_counts=True)

fd_singles = fd[ct==1]
idx_s = idx[ct==1]

idx_val = idx_s[0::2][:100]
idx_test = idx_s[1::2][:100]

for i in range(len(idx_val)):
    shutil.move(files[files_idx[idx_val[i]]],
        '../figures_v0_val/')
    shutil.move(files[files_idx[idx_test[i]]],
        '../figures_v0_test/')

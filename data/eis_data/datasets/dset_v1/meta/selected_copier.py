# 2023-01-08
# Ulas Kamaci

import numpy as np
import glob, shutil

src_path = './all_scans_processed/'

files_test = np.load('files_test.npy')
files_train = np.load('files_train.npy')
files_val = np.load('files_val.npy')

for file in files_train:
    files_match = glob.glob('all_scans_processed/*{}*.npy'.format(
        file[-19:-4]))
    for f in files_match:
        shutil.copy(f, 'selected_scans_train/')

for file in files_val:
    files_match = glob.glob('all_scans_processed/*{}*.npy'.format(
        file[-19:-4]))
    for f in files_match:
        shutil.copy(f, 'selected_scans_val/')

for file in files_test:
    files_match = glob.glob('all_scans_processed/*{}*.npy'.format(
        file[-19:-4]))
    for f in files_match:
        shutil.copy(f, 'selected_scans_test/')

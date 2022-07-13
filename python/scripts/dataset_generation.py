import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import glob, itertools
from mas.forward_model import size_equalizer
from slitless.forward import Source, Imager
from pqdm.processes import pqdm

def patch_extractor(file_path, path_save, ps=(64,64)):
    """
    Given a path to the folder of input images, read the images and extract 5
    patches in a central region of the image with the specified size, then save
    them in jpg format into the given save directory.

    Args:
        file_path (str): path to the folder containing input images
        path_save (str): path to the folder for the patches to be saved
        ps (tuple): patch size for the patch extraction
    """
    files = glob.glob(file_path+'*.jpg')
    ctr = 0
    for file in files:
        try:
            im = rgb2gray(plt.imread(file))
        except:
            continue
        aa,bb = im.shape
        cp = [ # center_points
            (aa//2, bb//2),
            (aa//3, bb//3),
            (2*aa//3, bb//3),
            (aa//3, 2*bb//3),
            (2*aa//3, 2*bb//3)
        ]
        patches = []
        for i in range(len(cp)):
            i0=im[cp[i][0]-ps[0]//2:cp[i][0]+ps[0]//2, cp[i][1]-ps[1]//2:cp[i][1]+ps[1]//2]
            plt.imsave(path_save+'image_{}.jpg'.format(ctr), i0, cmap='gray')
            print('Saving image: {}'.format(ctr))
            ctr += 1

# %% fwd
def fwd_meas(i):
    """
    Parallelizable script for reading image patches from a folder, passing them
    through the slitless forward model, and saving the resulting datapoints
    containing the parameters and the corresponding simulated measurements as 
    .npy files in the specified directory.

    Args:
        i (int): datapoint number
    """
    width_min_min = 0.5
    width_min_max = 1.0
    width_max_min = 0.7
    width_max_max = 2.0

    vel_min = 0.1
    vel_max = 1.0

    spectral_orders = [0,-1,1]
    imgr = Imager(
        spectral_orders=spectral_orders
    )

    files = np.array(glob.glob(file_path+'*.jpg'))
    numfiles = len(files)

    f_int,f_vel,f_width = files[np.random.randint(0, numfiles, 3)]
    int_ar = plt.imread(f_int)[:,:,0]
    vel_ar = plt.imread(f_vel)[:,:,0]
    width_ar = plt.imread(f_width)[:,:,0]
    int_ar = (int_ar-int_ar.min())/(int_ar.max()-int_ar.min())
    vel_ar = (vel_ar-vel_ar.min())/(vel_ar.max()-vel_ar.min())
    width_ar = (width_ar-width_ar.min())/(width_ar.max()-width_ar.min())
    vel_scale = np.random.uniform(vel_min, vel_max)
    width_min = np.random.uniform(width_min_min, width_min_max)
    width_max = width_max_min + (width_max_max - width_max_min) / (width_min_max-width_min_min)*(width_min-width_min_min)
    vel_ar = 2 * vel_scale * vel_ar - vel_scale
    width_ar = width_ar * (width_max - width_min) + width_min
    sr = Source(
        inten=int_ar,
        vel=vel_ar,
        width=width_ar,
        pix=True
    )
    imgr.get_measurements(sr)
    out = {'int': int_ar, 'vel': vel_ar, 'width': width_ar}
    for j in imgr.spectral_orders:
        out[f'meas_{j}'] = imgr.meas3d[str(j)]
    np.save(path_save+f'data_{i}.npy', out)

if __name__ == '__main__':
    # file_path =  '/home/kamo/resources/slitless/data/datasets/dset1_2022_07_02_102flowers/'
    # path_save = '/home/kamo/resources/slitless/data/datasets/dset2_2022_07_03_102flowers_64_64_patches/'
    #
    # patch_extractor(file_path, path_save)

    trainsize = 10000
    valsize = 2000
    testsize = 1000

    numsize = valsize
    file_path = '/home/kamo/resources/slitless/data/datasets/dset2_2022_07_03_102flowers_64_64_patches/'
    path_save0 = '/home/kamo/resources/slitless/data/datasets/dset3_2022_07_04_102flowers_meas_params/'
    path_save = path_save0+'val/'

    args = np.arange(numsize)
    pqdm(args, fwd_meas, n_jobs=32)

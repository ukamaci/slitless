import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import glob, itertools, os
from mas.forward_model import size_equalizer
from slitless.forward import Source, Imager, forward_op_torch
from pqdm.processes import pqdm
import torch

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

def eis_datasetter(arrays_path, out_path, ar_dim=64, pixelated=True):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    int_files = glob.glob(arrays_path+'int*')
    int_files.sort()
    vel_files = glob.glob(arrays_path+'vel*')
    vel_files.sort()
    width_files = glob.glob(arrays_path+'width*')
    width_files.sort()

    spectral_orders = [0,-1,1]
    imgr = Imager(spectral_orders=spectral_orders, pixelated=pixelated)

    for i in range(len(int_files)):
        print('Scan {} of {}'.format(i+1, len(int_files)))
        assert int_files[i][-19:]==vel_files[i][-19:], 'data name mismatch!'
        inten0 = np.load(int_files[i])
        vel0 = np.load(vel_files[i])
        width0 = np.load(width_files[i])
        numx = vel0.shape[1]//ar_dim
        numy = vel0.shape[0]//ar_dim

        for j in range(numy):
            for k in range(numx):
                intt = inten0[j*ar_dim:(j+1)*ar_dim, k*ar_dim:(k+1)*ar_dim]
                intt /= intt.max()
                vel = vel0[j*ar_dim:(j+1)*ar_dim, k*ar_dim:(k+1)*ar_dim]
                width = width0[j*ar_dim:(j+1)*ar_dim, k*ar_dim:(k+1)*ar_dim]
                sr = Source(
                    inten=intt,
                    vel=vel,
                    width=width,
                    pix=False
                )
                imgr.get_measurements(sr)

                out = {'int': imgr.srpix.inten, 'vel': imgr.srpix.vel, 'width': imgr.srpix.width}
                for zz,jj in enumerate(imgr.spectral_orders):
                    out[f'meas_{jj}'] = imgr.meas3dar[zz]
                np.save(out_path+f'data_{i+1}_{j*numx+k+1}.npy', out)

def dataset_visualizer(path, ind=None):
    files = glob.glob(path+'*npy')
    files.sort()
    if ind is None:
        ind = np.random.randint(len(files))
    file = files[ind]
    data = np.load(file, allow_pickle=True).item()
    # params = np.stack([data['int'], data['vel'], data['width']])
    # meas = np.stack([data['meas_0'], data['meas_-1'], data['meas_1']])
    sr = Source(
        inten=data['int'],
        vel=data['vel'],
        width=data['width'],
        pix=True
    )
    sr.plot(title=f'data {ind+1}')

def imagenet_datasetter():
    pathdset = '/home/kamo/resources/slitless/data/datasets/imagenet64_train_p1/'
    d1 = np.load(pathdset+'train_data_batch_1.npz')['data']
    d2 = np.load(pathdset+'train_data_batch_2.npz')['data']
    d = np.concatenate((d1,d2))[:180000]
    dg = []
    for i in range(len(d)//10000):
        print(i)
        temp = np.rot90(rgb2gray(d[i*10000:(i+1)*10000].reshape(-1,64,64,3, order='F')), axes=(2,1))
        dg.extend(temp)
    dg = np.array(dg)
    dgmin = dg.min(axis=(1,2), keepdims=True)
    dgmax = dg.max(axis=(1,2), keepdims=True)
    dg = (dg-dgmin)/(dgmax-dgmin)
    np.save(pathdset+'first_150k_gray_rot90.npy', dg[:150000])
    np.save(pathdset+'second_15k_gray_rot90.npy', dg[150000:165000])
    np.save(pathdset+'third_15k_gray_rot90.npy', dg[165000:180000])

def imagenet_datasetter2():
    pathdset = '/home/kamo/resources/slitless/data/datasets/imagenet64_train_p1/'
    outpath = '/home/kamo/resources/slitless/data/datasets/dset8_imagenet_50000/'
    trainset=np.load(pathdset+'first_150k_gray_rot90.npy')
    valset=np.load(pathdset+'second_15k_gray_rot90.npy')
    testset=np.load(pathdset+'third_15k_gray_rot90.npy')

    dataset = trainset
    path_save = outpath + 'train/'
    intens, vels, widths = dataset.reshape(3,-1,64,64)

    vel_max = 2 # pixels
    width_max = 2.2 # pixels
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

    data = np.concatenate((meas,intens.cpu()[:,None],vels.cpu()[:,None],widths.cpu()[:,None]), axis=1)

    for i in range(len(data)):
        if i%100==0:
            print(i)
        out = {
            'meas_0': data[i,0],
            'meas_-1': data[i,1],
            'meas_1': data[i,2],
            'int': data[i,3],
            'vel': data[i,4],
            'width': data[i,5]
        }
        np.save(path_save+f'data_{i}.npy', out)

    return data

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
    width_min_min = 1.0
    width_min_max = 1.3
    width_max_min = 1.3
    width_max_max = 2.2

    vel_min = 0.1
    vel_max = 2.0

    spectral_orders = [0,-1,1]
    imgr = Imager(
        spectral_orders=spectral_orders,
        pixelated=True
    )

    files = np.array(glob.glob(file_path+'*.jpg'))
    numfiles = len(files)

    f_int,f_vel,f_width = files[np.random.randint(0, numfiles, 3)]
    int_ar = plt.imread(f_int)[:,:,0]
    vel_ar = plt.imread(f_vel)[:,:,0]
    width_ar = plt.imread(f_width)[:,:,0]
    int_ar = (int_ar-int_ar.min())/(int_ar.max()-int_ar.min()+1e-6)
    vel_ar = (vel_ar-vel_ar.min())/(vel_ar.max()-vel_ar.min()+1e-6)
    width_ar = (width_ar-width_ar.min())/(width_ar.max()-width_ar.min()+1e-6)
    vel_scale = np.random.uniform(vel_min, vel_max)
    width_min = np.random.uniform(width_min_min, width_min_max)
    width_max = width_max_min + (width_max_max - width_max_min) / (width_min_max-width_min_min+1e-6)*(width_min-width_min_min)
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
    for k,j in enumerate(imgr.spectral_orders):
        out[f'meas_{j}'] = imgr.meas3dar[k]
    np.save(path_save+f'data_{i}.npy', out)

if __name__ == '__main__':
    # file_path =  '/home/kamo/resources/slitless/data/datasets/dset1_2022_07_02_102flowers/'
    # path_save = '/home/kamo/resources/slitless/data/datasets/dset2_2022_07_03_102flowers_64_64_patches/'
    #
    # patch_extractor(file_path, path_save)

    trainsize = 50000
    valsize = 5000
    testsize = 5000

    numsize = testsize
    file_path = '/home/kamo/resources/slitless/data/datasets/dset2_2022_07_03_102flowers_64_64_patches/'
    path_save0 = '/home/kamo/resources/slitless/data/datasets/dset7_flowers_50000/'
    path_save = path_save0+'test/'

    args = np.arange(numsize)
    pqdm(args, fwd_meas, n_jobs=32)

from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from mas.decorators import _vectorize
import numpy as np

@_vectorize(signature='(a,b),(a,b)->()', included=['truth', 'estimate'])
def compare_ssim(*, truth, estimate):
    return skimage_ssim(estimate, truth, data_range=np.max(truth) - np.min(truth))

@_vectorize(signature='(a,b),(a,b)->()', included=['truth', 'estimate'])
def nrmse(*, truth, estimate, normalization='minmax'):
    if normalization=='sigma':
        norm = np.std(truth)
    elif normalization=='minmax':
        norm = truth.max() - truth.min()
    elif normalization is None:
        norm = 1
    
    return np.sqrt(np.mean((truth-estimate)**2)) / norm
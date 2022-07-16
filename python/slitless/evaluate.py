import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, glob
from slitless.measure import compare_ssim, nrmse

def plot_recons(net, valloader, numim, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    numim = min(numim, x.shape[0])
    x = x.to(device=device, dtype=torch.float)
    y = np.array(y.cpu())
    with torch.no_grad():
        out = net(x)
    out = np.array(out.cpu())

    ssims = compare_ssim(truth=y, estimate=out)
    rmses = nrmse(truth=y, estimate=out, normalization=None)
    # yvec = yy.transpose(1,0,2,3).reshape(3,-1)
    # outvec = out.transpose(1,0,2,3).reshape(3,-1)

    matplotlib.use('Agg')
    for i in range(numim):
        fig, ax = plt.subplots(2,3, figsize=(13,7))
        i0=ax[0,0].imshow(y[i,0], cmap='hot')
        ax[0,0].set_title('True Intensity')
        fig.colorbar(i0, ax=ax[0,0])
        i1=ax[0,1].imshow(y[i,1], cmap='seismic')
        ax[0,1].set_title('True Velocity')
        fig.colorbar(i1, ax=ax[0,1])
        i2=ax[0,2].imshow(y[i,2], cmap='plasma')
        ax[0,2].set_title('True Linewidth')
        fig.colorbar(i2, ax=ax[0,2])
        i3=ax[1,0].imshow(out[i,0], cmap='hot')
        ax[1,0].set_title(
            'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,0], rmses[i,0]
            )
        )
        fig.colorbar(i3, ax=ax[1,0])
        i4=ax[1,1].imshow(out[i,1], cmap='seismic')
        ax[1,1].set_title(
            'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,1], rmses[i,1]
            )
        )
        fig.colorbar(i4, ax=ax[1,1])
        i5=ax[1,2].imshow(out[i,2], cmap='plasma')
        ax[1,2].set_title(
            'Predicted Linewidth\n SSIM={:.3f} - RMSE={:.3f}'.format(
                ssims[i,2], rmses[i,2]
            )
        )
        fig.colorbar(i5, ax=ax[1,2])
        plt.tight_layout()
        plt.savefig(savedir+f'recons_{i}.png', dpi=300)
        plt.close()
    plt.close('all')
    matplotlib.use('QtAgg')

if __name__ == '__main__':
    pass
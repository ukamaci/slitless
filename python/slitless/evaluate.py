import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch, glob
from slitless.measure import compare_ssim, nrmse
from slitless.networks.unet import UNet
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

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

def plot_val_stats(net, valloader, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    x = x.to(device=device, dtype=torch.float)
    y = np.array(y.cpu())
    with torch.no_grad():
        out = net(x)
    out = np.array(out.cpu())

    ssims = compare_ssim(truth=y, estimate=out)
    rmses = nrmse(truth=y, estimate=out, normalization=None)
    yvec = y.transpose(1,0,2,3).reshape(3,-1)
    outvec = out.transpose(1,0,2,3).reshape(3,-1)
    est_bias = np.mean(outvec - yvec, axis=1) 
    est_std = np.std(outvec - yvec, axis=1)

    fig, ax = plt.subplots(1,3, figsize=(12.6,4.8))
    ax[0].hist(ssims[:,0], bins=20)
    ax[0].set_title('Intensity SSIM\n Mean SSIM={:.3f}'.format(ssims[:,0].mean()))
    ax[0].set_xlabel('SSIM')
    ax[0].set_ylabel('Counts')
    ax[0].axvline(ssims[:,0].mean(), color='r')
    ax[1].hist(ssims[:,1], bins=20)
    ax[1].set_title('Velocity SSIM\n Mean SSIM={:.3f}'.format(ssims[:,1].mean()))
    ax[1].set_xlabel('SSIM')
    ax[1].set_ylabel('Counts')
    ax[1].axvline(ssims[:,1].mean(), color='r')
    ax[2].hist(ssims[:,2], bins=20)
    ax[2].set_title('Linewidth SSIM\n Mean SSIM={:.3f}'.format(ssims[:,2].mean()))
    ax[2].set_xlabel('SSIM')
    ax[2].set_ylabel('Counts')
    ax[2].axvline(ssims[:,2].mean(), color='r')
    plt.tight_layout()
    # plt.show()
    plt.savefig(savedir+'ssim_stats.png', dpi=300)

    fig, ax = plt.subplots(1,3, figsize=(12.6,4.8))
    ax[0].hist(rmses[:,0], bins=20)
    ax[0].set_title('Intensity RMSE\n Mean RMSE={:.3f}'.format(rmses[:,0].mean()))
    ax[0].set_xlabel('RMSE')
    ax[0].set_ylabel('Counts')
    ax[0].axvline(rmses[:,0].mean(), color='r')
    ax[1].hist(rmses[:,1], bins=20)
    ax[1].set_title('Velocity RMSE\n Mean RMSE={:.3f}'.format(rmses[:,1].mean()))
    ax[1].set_xlabel('RMSE')
    ax[1].set_ylabel('Counts')
    ax[1].axvline(rmses[:,1].mean(), color='r')
    ax[2].hist(rmses[:,2], bins=20)
    ax[2].set_title('Linewidth RMSE\n Mean RMSE={:.3f}'.format(rmses[:,2].mean()))
    ax[2].set_xlabel('RMSE')
    ax[2].set_ylabel('Counts')
    ax[2].axvline(rmses[:,2].mean(), color='r')
    plt.tight_layout()
    # plt.show()
    plt.savefig(savedir+'rmse_stats.png', dpi=300)

    fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.1,0.1], gridsize=100)
    fg.fig.suptitle('Intensity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
    fg.set_axis_labels('Intensity', 'Intensity Error')
    fg.fig.tight_layout()
    plt.savefig(savedir+'intensity_stats.png', dpi=300)
    fg=sns.jointplot(x=yvec[1], y=outvec[1]-yvec[1], kind='hex', ylim=[-0.4,0.4], gridsize=100)
    fg.fig.suptitle('Velocity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[1], est_std[1]))
    fg.set_axis_labels('Velocity', 'Velocity Error')
    fg.fig.tight_layout()
    plt.savefig(savedir+'velocity_stats.png', dpi=300)
    fg=sns.jointplot(x=yvec[2], y=outvec[2]-yvec[2], kind='hex', ylim=[-0.5,0.5], gridsize=100)
    fg.fig.suptitle('Linewidth Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[2], est_std[2]))
    fg.set_axis_labels('Linewidth', 'Linewidth Error')
    fg.fig.tight_layout()
    plt.savefig(savedir+'linewidth_stats.png', dpi=300)
    return ssims, rmses, yvec, outvec

if __name__ == '__main__':
    net = UNet(in_channels=3,
        start_filters=16,
        bilinear=True,
        ksize=(3,1),
        residual=False
    )
    net.load_state_dict(torch.load('../results/saved/2022_07_13__16_28_04_NF_16_LR_0.001_EP_50_MSE_LOSS/best_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    dataset_path = glob.glob('../../data/datasets/dset3*')[0]
    valset = BasicDataset(data_dir = dataset_path, fold='val')
    valloader = DataLoader(valset, batch_size=len(valset), shuffle=True)
    savedir = '/home/kamo/resources/slitless/python/results/saved/2022_07_13__16_28_04_NF_16_LR_0.001_EP_50_MSE_LOSS/'
    ssims, rmses, yvec, outvec = plot_val_stats(net, valloader, savedir)
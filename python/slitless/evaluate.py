import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch, glob, os
from slitless.measure import compare_ssim, nrmse
from slitless.networks.unet import UNet
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

def predict(net, meas):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(meas.shape)==3:
        meas = meas[np.newaxis]
    meas = torch.from_numpy(meas)
    meas = meas.to(device=device, dtype=torch.float)
    net = net.to(device)
    with torch.no_grad():
        pred = net(meas)
    pred = pred.squeeze().cpu().numpy()
    return pred

def net_loader(path):
    modpath = path+'/best_model.pth'
    with open(path+'/summary.txt', 'r') as summary_text:
        lines = summary_text.readlines()
    parser = lambda key: [i for i in lines if key in i][0].split('= ')[-1].split(' \n')[0]
    start_filters = int(parser('Number of starting'))
    outch = parser('Output Channels')
    out_channels = 3 if outch=='all' else 1
    ksizes=eval(parser('Kernel Size'))
    bilinear=eval(parser('Bilinear Interpolation'))
    numlayers = len(ksizes)
    numlayers = 4 if numlayers==1 else numlayers
    net = UNet(
        in_channels=3,
        out_channels=out_channels,
        numlayers=numlayers,
        outch_type=outch,
        start_filters=start_filters,
        bilinear=bilinear,
        ksizes=ksizes,
        residual=False)
    net.load_state_dict(torch.load(modpath))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    return net

def plot_recons(net, valloader, numim, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = next(iter(valloader))
    numim = min(numim, x.shape[0])
    x = x.to(device=device, dtype=torch.float)
    y = np.array(y.cpu())
    net.eval()
    with torch.no_grad():
        out = net(x)
    out = np.array(out.cpu())
    x = np.array(x.cpu())
    if not hasattr(net, 'outch_type'):
        net.outch_type = 'all'

    if net.outch_type == 'int':
        y1 = y[:,[0]]
    elif net.outch_type == 'vel':
        y1 = y[:,[1]]
    elif net.outch_type == 'width':
        y1 = y[:,[2]]
    elif net.outch_type == 'all':
        y1 = y

    ssims = compare_ssim(truth=y1, estimate=out)
    rmses = nrmse(truth=y1, estimate=out, normalization=None)
    ssims = ssims.squeeze()
    rmses = rmses.squeeze()

    matplotlib.use('Agg')
    for i in range(numim):
        fig, ax = plt.subplots(3,3, figsize=(13,7))
        im=ax[0,0].imshow(x[i,0], cmap='hot')
        ax[0,0].set_title('Meas 0')
        fig.colorbar(im, ax=ax[0,0])
        im=ax[0,1].imshow(x[i,1], cmap='hot')
        ax[0,1].set_title('Meas -1')
        fig.colorbar(im, ax=ax[0,1])
        im=ax[0,2].imshow(x[i,2], cmap='hot')
        ax[0,2].set_title('Meas +1')
        fig.colorbar(im, ax=ax[0,2])
        im=ax[1,0].imshow(y[i,0], cmap='hot')
        ax[1,0].set_title('True Intensity')
        fig.colorbar(im, ax=ax[1,0])
        im=ax[1,1].imshow(y[i,1], cmap='seismic')
        ax[1,1].set_title('True Velocity')
        fig.colorbar(im, ax=ax[1,1])
        im=ax[1,2].imshow(y[i,2], cmap='plasma')
        ax[1,2].set_title('True Linewidth')
        fig.colorbar(im, ax=ax[1,2])
        if net.outch_type == 'all':
            im=ax[2,0].imshow(out[i,0], cmap='hot')
            ax[2,0].set_title(
                'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,0], rmses[i,0]
                )
            )
            fig.colorbar(im, ax=ax[2,0])
            im=ax[2,1].imshow(out[i,1], cmap='seismic')
            ax[2,1].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,1], rmses[i,1]
                )
            )
            fig.colorbar(im, ax=ax[2,1])
            im=ax[2,2].imshow(out[i,2], cmap='plasma')
            ax[2,2].set_title(
                'Predicted Linewidth\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i,2], rmses[i,2]
                )
            )
            fig.colorbar(im, ax=ax[2,2])
        elif net.outch_type == 'int':
            im=ax[2,0].imshow(out[i,0], cmap='hot')
            ax[2,0].set_title(
                'Predicted Intensity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,0])
        elif net.outch_type == 'vel':
            im=ax[2,1].imshow(out[i,0], cmap='seismic')
            ax[2,1].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,1])
        elif net.outch_type == 'width':
            im=ax[2,2].imshow(out[i,0], cmap='plasma')
            ax[2,2].set_title(
                'Predicted Velocity\n SSIM={:.3f} - RMSE={:.3f}'.format(
                    ssims[i], rmses[i]
                )
            )
            fig.colorbar(im, ax=ax[2,2])

        plt.tight_layout()
        plt.savefig(savedir+f'recons_{i}.png', dpi=300)
        plt.close()
    plt.close('all')
    matplotlib.use('QtAgg')

def plot_val_stats(net, valloader, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    # x, y = next(iter(valloader))
    # x = x.to(device=device, dtype=torch.float)
    # y = np.array(y.cpu())
    # with torch.no_grad():
    #     out = net(x)
    # out = np.array(out.cpu())

    # ssims = compare_ssim(truth=y, estimate=out)
    # rmses = nrmse(truth=y, estimate=out, normalization=None)
    # yvec = y.transpose(1,0,2,3).reshape(3,-1)
    # outvec = out.transpose(1,0,2,3).reshape(3,-1)
    # est_bias = np.mean(outvec - yvec, axis=1) 
    # est_std = np.std(outvec - yvec, axis=1)

    ssims=[]
    rmses=[]
    yvec=[]
    outvec=[]
    for i, data in enumerate(valloader):
        # get the inputs
        inputs = data[0].to(device=device, dtype=torch.float)
        true_outputs = np.array(data[1].cpu())
        if not hasattr(net, 'outch_type'):
            net.outch_type = 'all'

        if net.outch_type == 'int':
            y1 = true_outputs[:,[0]]
            title_str = 'Intensity'
        elif net.outch_type == 'vel':
            y1 = true_outputs[:,[1]]
            title_str = 'Velocity'
        elif net.outch_type == 'width':
            y1 = true_outputs[:,[2]]
            title_str = 'Linewidth'
        elif net.outch_type == 'all':
            y1 = true_outputs

        with torch.no_grad():
            outputs = net(inputs)
            outputs = np.array(outputs.cpu())
            ssim0 = compare_ssim(truth=y1, estimate=outputs)
            rmse0 = nrmse(truth=y1, estimate=outputs, normalization=None)
            ssims.extend(ssim0.squeeze())
            rmses.extend(rmse0.squeeze())
            if i*valloader.batch_size<10000:
                if net.outch_type == 'all':
                    yvec.extend(y1.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
                    outvec.extend(outputs.transpose(1,0,2,3).reshape(3,-1).transpose(1,0))
                else:
                    yvec.extend(y1.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))
                    outvec.extend(outputs.transpose(1,0,2,3).reshape(1,-1).transpose(1,0))

    ssims = np.array(ssims)
    rmses = np.array(rmses)
    yvec = np.array(yvec).transpose(1,0)
    outvec = np.array(outvec).transpose(1,0)
    est_bias = np.mean(outvec - yvec, axis=1) 
    est_std = np.std(outvec - yvec, axis=1)

    if net.outch_type == 'all':
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
    else:
        plt.figure()
        plt.hist(ssims, bins=20)
        plt.title('{} SSIM\n Mean SSIM={:.3f}'.format(title_str, ssims.mean()))
        plt.xlabel('SSIM')
        plt.ylabel('Counts')
        plt.axvline(ssims.mean(), color='r')
        plt.tight_layout()
        plt.savefig(savedir+'ssim_stats.png', dpi=300)

    if net.outch_type == 'all':
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
    else:
        plt.figure()
        plt.hist(rmses, bins=20)
        plt.title('{} RMSE\n Mean RMSE={:.3f}'.format(title_str, rmses.mean()))
        plt.xlabel('RMSE')
        plt.ylabel('Counts')
        plt.axvline(rmses.mean(), color='r')
        plt.tight_layout()
        plt.savefig(savedir+'rmse_stats.png', dpi=300)

    if net.outch_type == 'all':
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

        # Cross dependence of vel&width errors on intensity
        fg=sns.jointplot(x=yvec[0], y=outvec[1]-yvec[1], kind='hex', ylim=[-0.4,0.4], gridsize=100)
        fg.fig.suptitle('Velocity Error vs Intensity\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[1], est_std[1]))
        fg.set_axis_labels('Intensity', 'Velocity Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'velocity_stats_vs_inten.png', dpi=300)
        fg=sns.jointplot(x=yvec[0], y=outvec[2]-yvec[2], kind='hex', ylim=[-0.5,0.5], gridsize=100)
        fg.fig.suptitle('Linewidth Error vs Intensity\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[2], est_std[2]))
        fg.set_axis_labels('Intensity', 'Linewidth Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'linewidth_stats_vs_inten.png', dpi=300)


    if net.outch_type == 'int':
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.1,0.1], gridsize=100)
        fg.fig.suptitle('Intensity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Intensity', 'Intensity Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'intensity_stats.png', dpi=300)

    if net.outch_type == 'vel':
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.4,0.4], gridsize=100)
        fg.fig.suptitle('Velocity Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Velocity', 'Velocity Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'velocity_stats.png', dpi=300)

    if net.outch_type == 'width':
        fg=sns.jointplot(x=yvec[0], y=outvec[0]-yvec[0], kind='hex', ylim=[-0.5,0.5], gridsize=100)
        fg.fig.suptitle('Linewidth Error Distribution\n Bias: {:.4f} - Error Std: {:.4f}'.format(est_bias[0], est_std[0]))
        fg.set_axis_labels('Linewidth', 'Linewidth Error')
        fg.fig.tight_layout()
        plt.savefig(savedir+'linewidth_stats.png', dpi=300)

    return ssims, rmses, yvec, outvec

if __name__ == '__main__':
    foldname0 = '2022_10_05__23_48*'
    foldpath = glob.glob('../results/saved/'+foldname0)[0]+'/'
    # modpath = foldpath+'best_model.pth'
    modpath = foldpath+'nf_64_LR_0.001_EP_50.pth'
    net = UNet(
        in_channels=3,
        out_channels=3,
        numlayers=4,
        outch_type='all',
        start_filters=64,
        bilinear=True,
        ksizes=[(3,1),(3,1),(3,1),(3,1)],
        residual=False)
    net.load_state_dict(torch.load(modpath))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    dataset_path = glob.glob('../../data/datasets/dset5*')[0]
    fold = 'train'
    dataset = BasicDataset(data_dir = dataset_path, fold=fold)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    savedir = foldpath
    savedir = foldpath+f'{fold}_results_last/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    ssims, rmses, yvec, outvec = plot_val_stats(net, dataloader, savedir)
    plot_recons(net, dataloader, 32, savedir+'figures/')
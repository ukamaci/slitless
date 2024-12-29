import matplotlib.pyplot as plt
from slitless.plotting import comparison_sweep
import numpy as np

savepath = '/home/kamo/resources/slitless/figures/apj2024_plots/'

def get_metric_comp_plots(save=False):
    a=       [0.011 , 0.009 , 3.030 , 2.268 , 2.462 , 1.961]
    a.extend([0.011 , 0.009 , 3.102 , 2.450 , 2.672 , 2.252])
    a.extend([0.010 , 0.008 , 2.343 , 1.758 , 1.836 , 1.400])
    a.extend([0.011 , 0.009 , 2.478 , 1.835 , 2.591 , 2.002])
    a.extend([0.011 , 0.008 , 2.642 , 2.095 , 2.688 , 2.271])
    a.extend([0.010 , 0.008 , 1.578 , 1.235 , 1.577 , 1.222])
    a.extend([0.011 , 0.009 , 2.171 , 1.593 , 2.489 , 1.943])
    a.extend([0.010 , 0.008 , 2.404 , 1.920 , 2.731 , 2.326])
    a.extend([0.009 , 0.007 , 1.449 , 1.131 , 1.548 , 1.205])
    a.extend([0.011 , 0.008 , 1.826 , 1.341 , 2.377 , 1.851])
    a.extend([0.010 , 0.008 , 2.224 , 1.771 , 2.787 , 2.400])
    a.extend([0.009 , 0.007 , 1.267 , 0.988 , 1.493 , 1.169])

    ar_k = np.array(a).reshape(-1,6)

    a=       [0.018 , 0.014 , 3.272 , 2.513 , 2.346 , 1.851]
    a.extend([0.035 , 0.028 , 4.372 , 3.459 , 3.499 , 2.779])
    a.extend([0.016 , 0.012 , 2.681 , 2.011 , 2.121 , 1.626]) 
    a.extend([0.012 , 0.009 , 2.923 , 2.231 , 2.378 , 1.865]) 
    a.extend([0.021 , 0.017 , 3.246 , 2.575 , 2.960 , 2.420]) 
    a.extend([0.014 , 0.010 , 2.157 , 1.677 , 1.930 , 1.507]) 
    a.extend([0.011 , 0.009 , 2.478 , 1.835 , 2.591 , 2.002]) 
    a.extend([0.011 , 0.008 , 2.642 , 2.095 , 2.688 , 2.271]) 
    a.extend([0.010 , 0.008 , 1.578 , 1.235 , 1.577 , 1.222]) 
    a.extend([0.006 , 0.004 , 1.748 , 1.285 , 2.257 , 1.789]) 
    a.extend([0.005 , 0.004 , 2.456 , 1.947 , 2.617 , 2.241]) 
    a.extend([0.007 , 0.005 , 1.218 , 0.951 , 1.429 , 1.107]) 

    ar_snr = np.array(a).reshape(-1,6)

    parameters = ['Intensity','Intensity','Velocity','Velocity','Line Width','Line Width']
    metrics = ['RMSE','MAE','RMSE (km/s)','MAE (km/s)','RMSE (km/s)','MAE (km/s)']

    for i in range(6):
        fig = comparison_sweep(
            methods=['1D MAP', 'MART', 'U-Net'],
            parameter=parameters[i],
            swept_param='K (Num. of Orders)',
            metric=metrics[i],
            array=ar_k[:,i].reshape(4,3).T,
            swept_arr=[2,3,4,5]
        )
        if save==True:
            fig.savefig(savepath+'K_{}_{}.png'.format(parameters[i],metrics[i][:4]), transparent=True, dpi=300)

    # ############### NOISE SWEEP #################

    for i in range(6):
        fig = comparison_sweep(
            methods=['1D MAP', 'MART', 'U-Net'],
            parameter=parameters[i],
            swept_param=r'$\gamma$'+' (SNR)',
            metric=metrics[i],
            array=ar_snr[:,i].reshape(4,3).T,
            swept_arr=['15', '25', '50', '100']
        )
        if save==True:
            fig.savefig(savepath+'SNR_{}_{}.png'.format(parameters[i],metrics[i][:4]), transparent=True, dpi=300)

def get_recon_comp_figures(save=False):
    file = '/home/kamo/resources/slitless/python/results/recons/2024_08_22__20_39_21_smart_eis_5_64x64_K_3_poisson_dbsnr_25/Rec.pickle'
    idx_im = 0
    idx_n = 0
    Rec_mart = np.load(file, allow_pickle=True)
    rec_mart = Rec_mart.recons[idx_im].recon[idx_n]
    file = '/home/kamo/resources/slitless/python/results/recons/2024_08_23__02_40_34_scipy_solver_eis_5_64x64_K_3_poisson_dbsnr_25/Rec.pickle'
    Rec_map1d = np.load(file, allow_pickle=True)
    rec_map1d = Rec_map1d.recons[idx_im].recon[idx_n]
    file = '/home/kamo/resources/slitless/python/results/recons/2024_09_02__13_13_27_nn_solver_eis_5_64x64_K_3_poisson_dbsnr_25/Rec.pickle'
    Rec_unet = np.load(file, allow_pickle=True)
    rec_unet = Rec_unet.recons[idx_im].recon[idx_n]
    true = Rec_unet.sources[idx_im].param3d

    recs = [true, rec_unet, rec_map1d, rec_mart]
    titles = ['True', 'U-Net', '1D MAP', 'MART']

    fig, ax = plt.subplots(3, 4, figsize=(10,7), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for i in range(4):
        im = ax[0,i].imshow(recs[i][0], vmin=true[0].min(), vmax=true[0].max(), cmap='hot')
        ax[0,i].axes.get_xaxis().set_visible(False)
        ax[0,i].axes.get_yaxis().set_visible(False)
        # if i==3:
        #     fig.colorbar(im, ax=ax[0,i])
        im = ax[1,i].imshow(recs[i][1], vmin=true[1].min(), vmax=true[1].max(), cmap='seismic')
        ax[1,i].axes.get_xaxis().set_visible(False)
        ax[1,i].axes.get_yaxis().set_visible(False)
        # if i==3:
        #     fig.colorbar(im, ax=ax[1,i])
        im = ax[2,i].imshow(recs[i][2], vmin=true[2].min(), vmax=true[2].max(), cmap='plasma')
        ax[2,i].axes.get_xaxis().set_visible(False)
        ax[2,i].axes.get_yaxis().set_visible(False)
        # if i==3:
        #     fig.colorbar(im, ax=ax[2,i])

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.895, 0.673, 0.02, 0.305])  # Adjust the position and size
    cbar_ax2 = fig.add_axes([0.895, 0.348, 0.02, 0.305])
    cbar_ax3 = fig.add_axes([0.895, 0.021, 0.02, 0.305])

    fig.colorbar(ax[0, 0].images[0], cax=cbar_ax1, orientation='vertical')
    fig.colorbar(ax[1, 0].images[0], cax=cbar_ax2, orientation='vertical')
    fig.colorbar(ax[2, 0].images[0], cax=cbar_ax3, orientation='vertical')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbars

    # plt.tight_layout()
    plt.show()
    if save==True:
        fig.savefig(savepath+'recons.png', transparent=True, dpi=300)

get_recon_comp_figures(save=True)
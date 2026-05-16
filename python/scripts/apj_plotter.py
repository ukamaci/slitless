import matplotlib.pyplot as plt
import numpy as np

savepath = '/home/kamo/resources/slitless/figures/apj2024_plots/'

def get_recon_comp_figures_final(idx_im=13, config='K_3_dbsnr_10', save=False):
    SPEED_OF_LIGHT = 299792.458
    base = '/home/kamo/resources/slitless/python/results/recons/'

    # Translate K_X_dbsnr_Y -> actual folder name (K_X_poisson_dbsnr_Y or K_X_None_dbsnr_None)
    K_str    = config[2 : config.index('_dbsnr_')]
    dbsnr_str = config[config.index('_dbsnr_') + 7:]
    noise_seg = 'None' if dbsnr_str == 'None' else 'poisson'
    folder = f'K_{K_str}_{noise_seg}_dbsnr_{dbsnr_str}'

    Rec_unet  = np.load(base + f'2026_05_13__14_19_03_final_runner_UNET/{folder}/Rec.pickle',  allow_pickle=True)
    Rec_map1d = np.load(base + f'2026_05_13__22_56_48_final_runner_1DMAP/{folder}/Rec.pickle', allow_pickle=True)
    Rec_mart  = np.load(base + f'2026_05_14__16_11_47_final_runner_MART/{folder}/Rec.pickle',  allow_pickle=True)

    idx_n = 0
    rec_unet  = Rec_unet.imager.frompix(Rec_unet.recons[idx_im].recon[idx_n],   width_unit='km/s', array=True)
    rec_map1d = Rec_map1d.imager.frompix(Rec_map1d.recons[idx_im].recon[idx_n], width_unit='km/s', array=True)
    rec_mart  = Rec_mart.imager.frompix(Rec_mart.recons[idx_im].recon[idx_n],   width_unit='km/s', array=True)

    true = Rec_unet.sources[idx_im].param3d.copy()
    rest_wl = getattr(Rec_unet.sources[idx_im], 'rest_wavelength', 195.117937907451)
    true[2] = true[2] * SPEED_OF_LIGHT / rest_wl  # Angstroms -> km/s

    recs         = [true, rec_unet, rec_map1d, rec_mart]
    titles       = ['Ground Truth', 'U-Net', '1D MAP', 'MART']
    cmaps        = ['hot', 'seismic', 'plasma']
    param_names  = ['Intensity', 'Velocity', 'Line Width']
    param_units  = ['[erg/cm²/s/sr]', '[km/s]', '[km/s]']

    fig, ax = plt.subplots(3, 4, figsize=(10, 7))
    for row in range(3):
        vmin, vmax = true[row].min(), true[row].max()
        for col in range(4):
            ax[row, col].imshow(recs[col][row], vmin=vmin, vmax=vmax, cmap=cmaps[row])
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            if row == 0:
                ax[row, col].set_title(titles[col], fontsize=15)
        # Parameter name (large) in the upper half, units (small) in the lower half
        ax[row, 0].set_ylabel(param_names[row], fontsize=15)
        ax[row, 0].yaxis.set_label_coords(-0.11, 0.53)
        ax[row, 0].text(-0.06, 0.53, param_units[row],
            transform=ax[row, 0].transAxes,
            fontsize=9, ha='center', va='center', rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.canvas.draw()  # needed so get_position() returns post-layout values

    cbar_width = 0.02
    cbar_pad   = 0.01
    for row in range(3):
        pos = ax[row, -1].get_position()
        cbar_ax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])
        fig.colorbar(ax[row, 0].images[0], cax=cbar_ax, orientation='vertical')
    plt.show()
    if save:
        fig.savefig(savepath + 'recons_final.png', transparent=True, dpi=300)
    return fig

def loaderr(x='rec_d_10'):
    file = f'/home/kamo/resources/slitless/python/scripts/basp25figs/{x}.pickle'
    return np.load(file, allow_pickle=True)

def fig2(save=False):
    rec_d_1000 = loaderr('rec_d_1000')
    rec_d_10 = loaderr('rec_d_10')
    rec_d_m10 = loaderr('rec_d_m10')
    rec_u_1000 = loaderr('rec_u_1000')
    rec_u_10 = loaderr('rec_u_10')
    rec_u_m10 = loaderr('rec_u_m10')

    # def get_recon_comp_figures_basp(save=False):
    true = rec_d_10.sources[0].param3d
    meas = rec_d_10.imager.meas3dar_nn[2]
    meas_m10 = rec_d_m10.imager.meas3dar[2]
    meas_10 = rec_d_10.imager.meas3dar[2]
    meas_1000 = rec_d_1000.imager.meas3dar[2]

    rec_d1000 = rec_d_1000.recons[0].recon[0]
    rec_d10 = rec_d_10.recons[0].recon[0]
    rec_dm10 = rec_d_m10.recons[0].recon[0]
    rec_u1000 = rec_u_1000.recons[0].recon[0]
    rec_u10 = rec_u_10.recons[0].recon[0]
    rec_um10 = rec_u_m10.recons[0].recon[0]

    recs = [true, rec_d1000, rec_d10, rec_dm10, rec_u1000, rec_u10, rec_um10]
    meass = [meas, meas_1000, meas_10, meas_m10, meas_1000, meas_10, meas_m10]
    # titles = ['True', 'U-Net', '1D MAP', 'MART']

    fig, ax = plt.subplots(4, 7, figsize=(14,7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1]})
    for i in range(7):
        im = ax[0,i].imshow(recs[i][0], vmin=true[0].min(), vmax=true[0].max(), cmap='hot')
        ax[0,i].axes.get_xaxis().set_visible(False)
        ax[0,i].axes.get_yaxis().set_visible(False)

        im = ax[1,i].imshow(recs[i][1], vmin=true[1].min(), vmax=true[1].max(), cmap='seismic')
        ax[1,i].axes.get_xaxis().set_visible(False)
        ax[1,i].axes.get_yaxis().set_visible(False)

        im = ax[2,i].imshow(recs[i][2], vmin=true[2].min(), vmax=true[2].max(), cmap='plasma')
        ax[2,i].axes.get_xaxis().set_visible(False)
        ax[2,i].axes.get_yaxis().set_visible(False)

        im = ax[3,i].imshow(meass[i], vmin=meass[0].min(), vmax=meass[0].max(), cmap='hot')
        ax[3,i].axes.get_xaxis().set_visible(False)
        ax[3,i].axes.get_yaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbars
    plt.show()

    if save==True:
        fig.savefig(savepath+'recons.png', transparent=True, dpi=300)
    
# fig2(save=False)

def fig3(save=False):
    rec_d_1000 = loaderr('rec_d_1000')
    rec_d_10 = loaderr('rec_d_10')
    rec_d_m10 = loaderr('rec_d_m10')
    rec_u_1000 = loaderr('rec_u_1000')
    rec_u_10 = loaderr('rec_u_10')
    rec_u_m10 = loaderr('rec_u_m10')

    recs_ = [rec_d_1000, rec_d_10, rec_d_m10, rec_u_1000, rec_u_10, rec_u_m10]


    # def get_recon_comp_figures_basp(save=False):
    true = rec_d_10.sources[0].param3d
    meas = rec_d_10.imager.meas3dar_nn[2]
    meas_m10 = rec_d_m10.imager.meas3dar[2]
    meas_10 = rec_d_10.imager.meas3dar[2]
    meas_1000 = rec_d_1000.imager.meas3dar[2]

    rec_d1000 = rec_d_1000.recons[0].recon[0]
    rec_d10 = rec_d_10.recons[0].recon[0]
    rec_dm10 = rec_d_m10.recons[0].recon[0]
    rec_u1000 = rec_u_1000.recons[0].recon[0]
    rec_u10 = rec_u_10.recons[0].recon[0]
    rec_um10 = rec_u_m10.recons[0].recon[0]

    recs = [true, rec_d1000, rec_d10, rec_dm10, rec_u1000, rec_u10, rec_um10]
    meass = [meas, meas_1000, meas_10, meas_m10, meas_1000, meas_10, meas_m10]

    fig, ax = plt.subplots(4, 7, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1]})

    for i in range(7):  # Loop over the 7 columns
        # First row
        im = ax[0, i].imshow(recs[i][0], cmap='hot')
        ax[0, i].axes.get_xaxis().set_visible(False)
        ax[0, i].axes.get_yaxis().set_visible(False)

        # Second row
        im = ax[1, i].imshow(recs[i][1], cmap='seismic')
        ax[1, i].axes.get_xaxis().set_visible(False)
        ax[1, i].axes.get_yaxis().set_visible(False)

        # Third row
        im = ax[2, i].imshow(recs[i][2], cmap='plasma')
        ax[2, i].axes.get_xaxis().set_visible(False)
        ax[2, i].axes.get_yaxis().set_visible(False)

        # Fourth row
        im = ax[3, i].imshow(meass[i], cmap='hot')
        ax[3, i].axes.get_xaxis().set_visible(False)
        ax[3, i].axes.get_yaxis().set_visible(False)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 1])  # Leave space for colorbars
    plt.show()

    if save==True:
        fig.savefig(savepath+'recons.png', transparent=True, dpi=300)

    return recs_

if __name__ == '__main__':
    savepath = '/home/kamo/resources/slitless/figures/apj26_post_revision/'
    get_recon_comp_figures_final(
        idx_im=4, 
        config='K_3_dbsnr_None', 
        save=True
    )

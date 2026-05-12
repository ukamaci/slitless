import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import Source, Imager
from slitless.recon import scipy_solver_parallel, Reconstructor

def main():
    print("=== 1. Loading Real Ground Truth ===")
    path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
    data_file = 'eis_train_5_dsetv5.npy'
    
    # Load dataset
    data = np.load(path_data + data_file, allow_pickle=True).item()
    
    # Extract the first image's ground truth parameters
    param3d_phy = data['param3d'][0]
    
    # Constants matching your pipeline
    REST_WAVELENGTH = 195.117937907451
    MID_WAVELENGTH = 195.119
    DISP_SCALE = 0.022275
    
    print("=== 2. Simulating Measurements ===")
    source = Source(
        param3d=param3d_phy, 
        pix=False, 
        rest_wavelength=REST_WAVELENGTH
    )
    
    # intenscale = param3d_phy[0].max()
    intenscale = 1
    imager = Imager(
        pixelated=True, 
        spectral_orders=[0, -1, 1],
        mid_wavelength=MID_WAVELENGTH,
        dispersion_scale=DISP_SCALE,
        intenscale=intenscale
    )
    
    # This correctly scales and maps velocities to sub-pixel offsets
    imager.topix(source) 
    
    # Generate pure, noise-free measurements directly
    imager.get_measurements(noise_model=None, no_noise=True)
    
    print("=== 3. Reconstructing with 1D MAP (scipy_solver_parallel) ===")
    rec = Reconstructor(
        imager=imager,
        solver=scipy_solver_parallel,
        intenscale=intenscale,
        simulate_meas=False, # We already simulated them perfectly
        DATA_FIDELITY='L2',
        OPTIMIZER='L-BFGS-B',
        maxiter=10000,
        # Very low regularization so the optimizer can freely fit the pure data
        lam_i=1e-8, 
        lam_v=1e4 / (intenscale**2),
        lam_w=1e-2 / (intenscale**2),
        use_bg_prior=False,
        use_soft_bg_prior=False,
        n_jobs=-1
    )
    
    recons = rec.solve(num_realizations=1)
    
    # Extract reconstruction and un-scale back to physics
    recon_pix = recons.recon[0]
    recon_source_pix = Source(param3d=recon_pix, pix=True, rest_wavelength=REST_WAVELENGTH)
    recon_source_phy = imager.frompix(recon_source_pix, width_unit='A')
    recon_phy = recon_source_phy.param3d
    
    print("\n=== 4. Evaluation (Physical Units) ===")
    int_diff = recon_phy[0] - param3d_phy[0]
    vel_diff = recon_phy[1] - param3d_phy[1]
    wid_diff = recon_phy[2] - param3d_phy[2]
    
    SPEED_OF_LIGHT = 299792.458
    wid_diff_kms = wid_diff * SPEED_OF_LIGHT / REST_WAVELENGTH
    
    print(f"Intensity Bias: {int_diff.mean():9.4f}, RMSE: {np.sqrt((int_diff**2).mean()):9.4f}")
    print(f"Velocity  Bias: {vel_diff.mean():9.4f} km/s, RMSE: {np.sqrt((vel_diff**2).mean()):9.4f} km/s")
    print(f"Width     Bias: {wid_diff_kms.mean():9.4f} km/s, RMSE: {np.sqrt((wid_diff_kms**2).mean()):9.4f} km/s")
    
    # Let's use the object-oriented plotter we built to visualize the results
    fig, axes = recons.plot(compare=True, title='Sandbox 1D MAP Verification')
    
    # Adjust the plot to show we succeeded
    plt.tight_layout()
    plt.savefig('sandbox_1dmap_test.png', dpi=150)
    print("\nSaved plot to 'sandbox_1dmap_test.png'")
    plt.show()

if __name__ == '__main__':
    main()
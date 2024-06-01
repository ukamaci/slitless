from slitless.forward import Source, Imager, forward_op
from slitless.recon import grad_descent_solver, scipy_solver, Reconstructor_Multi
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import datetime, os

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
# data = 'apj19_20x20.npy' # 5 of 64x64 EIS dataset test images
data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset test images

param4dar = np.load(path_data+data)
if len(param4dar.shape)<4:
    param4dar = param4dar[np.newaxis]

M = param4dar.shape[-1]

mask = np.array([[(i + j) % 2 for j in range(M)] for i in range(M)])
mask = np.ones_like(mask)
Imgr = Imager(pixelated=True, mask=mask, max_count=50**2/0.9, noise_model='poisson')

lam_v_list = np.logspace(-1,1,3)
lam_w_list = np.logspace(-1,1,3)
rmse_list = np.zeros((len(lam_v_list), len(lam_w_list), 3))

for i, lam_v in tqdm(enumerate(lam_v_list)):
    for j, lam_w in tqdm(enumerate(lam_w_list), leave=False):
        print('Outer: {}/{}'.format(i+1, len(lam_v_list)))
        print('Inner:{}/{}'.format(j+1, len(lam_w_list)))

        Rec = Reconstructor_Multi(
            imager=Imgr,
            param4dar=param4dar,
            pix=True,
            solver=scipy_solver,
            DATA_FIDELITY='L2',
            OPTIMIZER='L-BFGS-B',
            maxiter=10000,
            lam_i=1e-4,
            lam_v=lam_v,
            lam_w=lam_w,
            # LR=5e-2
        )


        # Rec = Reconstructor_Multi(
        #     imager=Imgr,
        #     param4dar=param4dar,
        #     pix=True,
        #     solver=grad_descent_solver,
        #     maxiter=1000,
        #     lam_i=1e-8,
        #     lam_v=lam_v,
        #     lam_w=lam_w,
        #     LR=5e-2
        # )

        recons_gd = Rec.solve(num_realizations=1)
        rmse_list[i,j] = Rec.rmse_phy.mean(axis=(0,1))

ind_v = np.unravel_index(np.argmin(rmse_list[:,:,1]), rmse_list.shape[:2])
ind_w = np.unravel_index(np.argmin(rmse_list[:,:,2]), rmse_list.shape[:2])
ind_vw = np.unravel_index(np.argmin(rmse_list[:,:,2]+rmse_list[:,:,1]), rmse_list.shape[:2])

now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
if Rec.solver.__name__ == 'grad_descent_solver':
    name = (f'{now}_{Rec.solver.__name__}_LR_{Rec.solver_params['LR']}_' +
            f'Maxiter_{Rec.solver_params['maxiter']}_Mask_{int(mask[0,0])}{int(mask[0,1])}')
else:
    name = (f'{now}_{Rec.solver.__name__}_' +
            f'Maxiter_{Rec.solver_params['maxiter']}_Mask_{int(mask[0,0])}{int(mask[0,1])}')
os.mkdir('/home/kamo/resources/slitless/python/results/param_search/'+name)
search_summary = [
'###### Reconstructor Params ###### \n'
'Solver: {} \n'.format(Rec.solver.__name__),
'Solver Params: {} \n'.format(Rec.solver_params),
'mask: {} \n'.format(mask[:2,:2]),
'Num Realizations: {} \n'.format(Rec.num_realizations),
'lam_v_list: {} \n'.format(lam_v_list),
'lam_w_list: {} \n'.format(lam_w_list),
'Vel-Opt lam_v: {} \n'.format(lam_v_list[ind_v[0]]),
'Vel-Opt lam_w: {} \n'.format(lam_w_list[ind_v[1]]),
'Vel-Opt RMSE: {} \n'.format(rmse_list[ind_v]),
'Width-Opt lam_v: {} \n'.format(lam_v_list[ind_w[0]]),
'Width-Opt lam_w: {} \n'.format(lam_w_list[ind_w[1]]),
'Width-Opt RMSE: {} \n'.format(rmse_list[ind_w]),
'Vel-Width-Opt lam_v: {} \n'.format(lam_v_list[ind_vw[0]]),
'Vel-Width-Opt lam_w: {} \n'.format(lam_w_list[ind_vw[1]]),
'Vel-Width-Opt RMSE: {} \n'.format(rmse_list[ind_vw]),
]

with open(f'/home/kamo/resources/slitless/python/results/param_search/{name}/summary.txt', 'w') as file:
    for line in search_summary:
        file.write(line)

# %%
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].contourf(np.log10(lam_w_list),np.log10(lam_v_list),rmse_list[:,:,1])
im=ax[1].contourf(np.log10(lam_w_list),np.log10(lam_v_list),rmse_list[:,:,2])
ax[0].set_title('RMSE VEL')
ax[1].set_title('RMSE WIDTH')
ax[0].set_xlabel('log(lam_w)')
ax[0].set_ylabel('log(lam_v)')
ax[1].set_xlabel('log(lam_w)')
ax[1].set_ylabel('log(lam_v)')
ax[0].grid(which='both', axis='both')
ax[1].grid(which='both', axis='both')
ax[0].plot(np.log10(lam_w_list[ind_v[1]]),np.log10(lam_v_list[ind_v[0]]), 'D', color='red', markersize=10)
ax[1].plot(np.log10(lam_w_list[ind_w[1]]),np.log10(lam_v_list[ind_w[0]]), 'D', color='red', markersize=10)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
# plt.tight_layout()
plt.savefig(f'/home/kamo/resources/slitless/python/results/param_search/{name}/contour_plots.png')
plt.show()
# %%

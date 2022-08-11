# 2022-08-09 Ulas Kamaci
# Gradient descent for slitless using PyTorch Autograd
import torch, glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset
from slitless.forward import Source, Imager, forward_op_torch
from slitless.measure import cycle_loss
from tqdm.auto import tqdm

dataset_path = glob.glob('/home/kamo/resources/slitless/data/datasets/dset3*')[0]
dataset = BasicDataset(data_dir = dataset_path, fold='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
_, x = next(iter(dataloader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sr = Source(
    inten=np.array(x[0,0]),
    vel=np.array(x[0,1]),
    width=np.array(x[0,2]),
    pix=True # the input arrays are given in pixel units
)

pixelated = True

imgr = Imager(pixelated=pixelated)
imgr.get_measurements(sr)

mu_vel = 1e3 # GD step size for vel
mu_width = 2e3 # GD step size for width
maxiters = 10000
losses = []
diffs_vel = []
diffs_width = []

meas = torch.from_numpy(imgr.meas3dar).to(device=device, dtype=torch.float)
x = x.to(device=device, dtype=torch.float)

xh_int = x[0,0]
xh_vel = torch.zeros(
    (meas.shape[1],meas.shape[2]), device=device, dtype=torch.float, requires_grad=True)
# xh_vel = x[0,1]
xh_width = 0.75*torch.ones((meas.shape[1],meas.shape[2]), device=device, dtype=torch.float)
xh_width = xh_width.requires_grad_()
# xh_width = x[0,2]

for i in tqdm(range(maxiters)):
    loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=pixelated))**2)
    loss.backward()

    with torch.no_grad():
        xh_vel -= mu_vel * xh_vel.grad
        xh_vel.grad.zero_()
        xh_width -= mu_width * xh_width.grad
        xh_width.grad.zero_()

        losses.append(loss.detach().cpu().numpy())
        diff_vel = torch.sum((x[0,1]-xh_vel)**2)/torch.sum(x[0,1]**2)
        diffs_vel.append(diff_vel.detach().cpu().numpy())

    # loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel.detach(),xh_width, pixelated=pixelated))**2)
    # loss.backward()

    # with torch.no_grad():
        # xh_width -= mu_width * xh_width.grad
        # xh_width.grad.zero_()

    #     losses.append(loss.detach().cpu().numpy())
        diff_width = torch.sum((x[0,2]-xh_width)**2)/torch.sum(x[0,2]**2)
        diffs_width.append(diff_width.detach().cpu().numpy())

plt.figure()
plt.plot(losses)
plt.title('Loss vs Iter')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(diffs_vel)
plt.title('||x_vel-xh_vel|| vs Iter')
plt.xlabel('Iter')
plt.ylabel('||x_vel-xh_vel||')
plt.show()

plt.figure()
plt.plot(diffs_width)
plt.title('||x_width-xh_width|| vs Iter')
plt.xlabel('Iter')
plt.ylabel('||x_width-xh_width||')
plt.show()

sr2 = Source(
    inten=xh_int.detach().cpu().numpy(),
    vel=xh_vel.detach().cpu().numpy(),
    width=xh_width.detach().cpu().numpy()
)

sr.plot('Original')
plt.tight_layout()
sr2.plot('Estimated')
plt.tight_layout()
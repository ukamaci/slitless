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

mu = 1e3 # GD step size
maxiters = 20000
losses = []
diffs = []

meas = torch.from_numpy(imgr.meas3dar).to(device=device, dtype=torch.float)
x = x.to(device=device, dtype=torch.float)

# KEEP INT & WIDTH AS TRUE VALUES AND ONLY ESTIMATE VELOCITY
xh_int = x[0,0]
xh_vel = torch.zeros(
    (meas.shape[1],meas.shape[2]), device=device, dtype=torch.float, requires_grad=True)
# xh_width = 0.6*torch.ones(
#     (meas.shape[1],meas.shape[2]), device=device, dtype=torch.float, requires_grad=True)
xh_width = x[0,2]

for i in tqdm(range(maxiters)):
    # loss = cycle_loss(meas, xh)
    loss = torch.mean((meas-forward_op_torch(xh_int,xh_vel,xh_width, pixelated=pixelated))**2)
    loss.backward()

    with torch.no_grad():
        xh_vel -= mu * xh_vel.grad
        xh_vel.grad.zero_()

        losses.append(loss.detach().cpu().numpy())
        diff = torch.sum((x[0,1]-xh_vel)**2)/torch.sum(x[0,1]**2)
        diffs.append(diff.detach().cpu().numpy())

plt.figure()
plt.plot(losses)
plt.title('Loss vs Iter')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(diffs)
plt.title('||x-xh|| vs Iter')
plt.xlabel('Iter')
plt.ylabel('||x-xh||')
plt.show()

xhh = xh_vel.detach().cpu().numpy()
sr2 = Source(
    inten=x[0,0].cpu().numpy(),
    vel=np.array(xhh),
    width=x[0,2].cpu().numpy()
)

sr.plot('Orig')
sr2.plot('Est')
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from slitless.forward import forward_op, gauss, add_noise
from torch.utils.data import DataLoader
from slitless.data_loader import BasicDataset

gauss2 = lambda x, sigma: gauss(x=x, mean=0, sigma=sigma)

sig_true = 1.5
sig_sig = 0.6

inten = np.random.uniform(0, 1, size=(64,64))
vel = np.random.normal(np.zeros((64,64)), scale=0.2)
width = np.random.uniform(sig_true-sig_sig,sig_true+sig_sig, size=(64,64))

dataset_path = glob.glob('/home/kamo/resources/slitless/data/eis_data/eistest64*')[0]
dataset = BasicDataset(data_dir = dataset_path, fold='test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
_, x = next(iter(dataloader))
inten, vel, width = x[0].cpu().numpy()

# inten = np.random.uniform(0, 1, size=(64,64))
# vel = np.random.normal(np.zeros((64,64)), scale=0.0)

sig_true, sig_sig = width.mean(), width.std()

y0 = forward_op(inten, vel, width, pixelated=True)

y = add_noise(y0, dbsnr=35, no_noise=True)

y1f = np.fft.fft(y[1], axis=0)
y2f = np.fft.fft(y[2], axis=0)
y0f = np.fft.fft(y[0], axis=0)
ysf = (y1f+y2f)/2

div1 = (y1f/y0f).mean(axis=1)
divs = (ysf/y0f).mean(axis=1)
# div1 = np.median((y1f/y0f), axis=1)
# divs = np.median((ysf/y0f), axis=1)

div1a = np.fft.fftshift(abs(div1))
divsa = np.fft.fftshift(abs(divs))

# div1a -= div1a.min()
# divsa -= divsa.min()

xdata = np.arange(inten.shape[0]) - inten.shape[0] // 2
sig1 = 1/(curve_fit(gauss2, xdata, div1a/div1a.sum())[0]*2*np.pi/inten.shape[0])[0]
sigs = 1/(curve_fit(gauss2, xdata, divsa/divsa.sum())[0]*2*np.pi/inten.shape[0])[0]

print(f'Sig_True = {sig_true}+/-{sig_sig}')
print('Sig_1 = {:.2f}'.format(sig1))
print('Sig_s = {:.2f}'.format(sigs))

plt.figure()
plt.plot(div1a/div1a.sum())
plt.plot(divsa/divsa.sum())
plt.plot(gauss2(xdata, 1/sigs*inten.shape[0]/2/np.pi))
plt.show()

f1 = np.fft.fftshift(abs(np.fft.fft(inten, axis=0)).mean(axis=1))
plt.figure()
plt.plot(f1)
plt.show()
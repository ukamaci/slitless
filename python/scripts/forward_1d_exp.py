from slitless.forward import Source, Imager
import numpy as np
import matplotlib.pyplot as plt


# %% impulse
M = 21 # detector size
k = [7,13] # location of the impulse(s)
true_intensity = np.zeros((M,1))
true_intensity[k] = np.array([[1,1]]).T
true_doppler = np.zeros((M,1))
true_doppler[k] = np.array([[0.75,-0.75]]).T
true_linewidth = np.zeros((M,1))+0.01
true_linewidth[k] = np.array([[1.7,1.5]]).T
spectral_orders=[0,-1,1]

sr = Source(
    inten=true_intensity,
    vel=true_doppler,
    width=true_linewidth,
    pix=True
)

imgr = Imager(
    spectral_orders=spectral_orders
)

imgr.get_measurements(sr)
meas = imgr.meas3d

fig, ax = plt.subplots(5,1, figsize=(6.4,10))
fig.suptitle('Measurements')
ax[0].stem(meas['0'])
ax[0].set_title('Order 0')
ax[1].stem(meas['-1'])
ax[1].set_title('Order -1')
ax[2].stem(meas['1'])
ax[2].set_title('Order +1')
ax[3].stem(meas['1']+meas['-1'])
ax[3].set_title('Sum of Order +1&-1')
ax[4].stem(abs(meas['1']-meas['-1']))
ax[4].set_title('Abs Diff of Order +1&-1')
plt.tight_layout()
plt.show()

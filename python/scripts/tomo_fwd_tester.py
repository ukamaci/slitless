import numpy as np
import matplotlib.pyplot as plt
from slitless.forward import (forward_op_tomo_3d, forward_op_tomo_3d_v0, forward_op_tomo_3d_k3,
forward_op_tomo_3d_transpose_k3, forward_op_tomo_3d_transpose, datacube_generator, tomomtx_gen)

param3d = np.zeros((3,21,1))
param3d[2] = 1
param3d[:,10,:] = np.array([1,0.5,2])[:,None]

dc = datacube_generator(param3d)
M,N,R = dc.shape
orders=[0,-1,1,-2,2]
# orders=[0,-1,1]

print('dc shape: {}'.format(dc.shape))

mtx = tomomtx_gen(dc.shape[:2], orders=orders+['inf'])
mtx_t = np.einsum('ijk->ikj', mtx.reshape(-1,N,M*N))
mtx_s = (np.sum(mtx_t,axis=2)==0).astype(int).reshape(-1,M,N)[:,:,:,None]
mtx_s2 = (np.sum(mtx_t,axis=2)<1).astype(int).reshape(-1,M,N)[:,:,:,None]

m1 = forward_op_tomo_3d_k3(dc)
m2 = forward_op_tomo_3d_v0(dc, orders=orders, inf=True)
m3 = (mtx @ dc.reshape(-1,R)).reshape(m2.shape)

d2 = forward_op_tomo_3d_transpose(m2, orders=orders, inf=True)
d3 = np.einsum('ijk,ikm->ijm',mtx_t, m3).reshape(d2.shape)
d3+=mtx_s

m4 = m3*0+1
d4 = np.einsum('ijk,ikm->ijm',mtx_t, m4).reshape(d2.shape)
d4+=mtx_s
d5=np.ones_like(d4)*d4[:]
d5[mtx_s2==1]=1


print('tomo_3d_k check: np.allclose(m1, m2[:3]): {}'.format(np.allclose(m1,m2[:3])))

d1 = forward_op_tomo_3d_transpose_k3(m1)
print('d1 shape: {}'.format(d1.shape))

fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d1[0,:,:,0])
ax[0].set_title('Order 0')
ax[1].imshow(d1[1,:,:,0])
ax[1].set_title('Order -1')
ax[2].imshow(d1[2,:,:,0])
ax[2].set_title('Order 1')
plt.show()

plt.figure()
plt.imshow(d1[1,:,:,0])
plt.colorbar()
plt.title('Order -1')
plt.show()

d2 = forward_op_tomo_3d_transpose(m2, orders=[0,-1,1,-2,2])
print('d2 shape: {}'.format(d2.shape))
print('tomo_3d_trans_k check: np.allclose(d1, d2[:3]): {}'.format(np.allclose(d1,d2[:3])))

fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d2[0,:,:,0])
ax[0].set_title('Order 0')
ax[1].imshow(d2[1,:,:,0])
ax[1].set_title('Order -1')
ax[2].imshow(d2[2,:,:,0])
ax[2].set_title('Order 1')
plt.show()


fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d2[0,:,:,0])
ax[0].set_title('Order 0')
ax[1].imshow(d2[3,:,:,0])
ax[1].set_title('Order -2')
ax[2].imshow(d2[4,:,:,0])
ax[2].set_title('Order 2')
plt.show()


fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d3[0,:,:,0])
ax[0].set_title('Order 0 (mtx)')
ax[1].imshow(d3[3,:,:,0])
ax[1].set_title('Order -2 (mtx)')
ax[2].imshow(d3[4,:,:,0])
ax[2].set_title('Order 2 (mtx)')
plt.show()
        

fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d4[0,:,:,0])
ax[0].set_title('Order 0 (mtx_1)')
ax[1].imshow(d4[3,:,:,0])
ax[1].set_title('Order -2 (mtx_1)')
ax[2].imshow(d4[4,:,:,0])
ax[2].set_title('Order 2 (mtx)_1')
plt.show()

fig, ax = plt.subplots(1,3,figsize=(5,21))
ax[0].imshow(d5[0,:,:,0])
ax[0].set_title('Order 0 (mtx2_1)')
ax[1].imshow(d5[3,:,:,0])
ax[1].set_title('Order -2 (mtx2_1)')
ax[2].imshow(d5[4,:,:,0])
ax[2].set_title('Order 2 (mtx2_1)')
plt.show()
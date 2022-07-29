import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import camera, shepp_logan_phantom, cell
from scipy.optimize import minimize

def gauss(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * np.exp(-0.5*((x-mean)/sigma)**2)

def gauss_torch(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * torch.exp(-0.5*((x-mean)/sigma)**2)

def forward_op(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    param3d=None,
    spectral_orders=[0,-1,1]):
    """
    Given 2d arrays of intensity, doppler, and linewidth, calculate the noise
    free measurements at the specified spectral orders.

    Notes: This implementation uses einsum instead of for loops and is faster.

    Args:
        true_intensity (ndarray): 2d array of true intensities.
        true_doppler (ndarray): 2d array of true doppler shifts in the units of
            pixels.
        true_linewidth (ndarray): 2d array of true line widths in the units of
            pixels.
        param3d (ndarray): 3d array of the 2d arrays of intensity, velocity, 
            and line width. If provided, the 2d parameter inputs are ignored. 
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (ndarray): 3d array of measurements. The first dimension
            contains the specified spectral orders with the same ordering.
    """
    if param3d is not None:
        true_intensity, true_doppler, true_linewidth = param3d
    aa, bb = true_intensity.shape
    out = np.zeros((len(spectral_orders),)+(aa,bb))
    diffrange = np.arange(aa)[np.newaxis,:,np.newaxis]-np.arange(aa)[np.newaxis,np.newaxis,:]
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        if a == 0:
            out[z] = true_intensity.copy()
            continue
        out[z] = np.einsum(
            'kij,kj->ki',
            gauss(
                diffrange, 
                a*true_doppler.transpose(1,0)[:,np.newaxis,:], 
                abs(a)*true_linewidth.transpose(1,0)[:,np.newaxis,:]
            ), 
            true_intensity.transpose(1,0)
        ).transpose(1,0)
    return out

def forward_op_loopy(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    param3d=None,
    spectral_orders=[0,-1,1]):
    """
    Given 2d arrays of intensity, doppler, and linewidth, calculate the noise
    free measurements at the specified spectral orders.

    Warning: This is an old an slow implementation. 

    Args:
        true_intensity (ndarray): 2d array of true intensities.
        true_doppler (ndarray): 2d array of true doppler shifts in the units of
            pixels.
        true_linewidth (ndarray): 2d array of true line widths in the units of
            pixels.
        param3d (ndarray): 3d array of the 2d arrays of intensity, velocity, 
            and line width. If provided, the 2d parameter inputs are ignored. 
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (ndarray): 3d array of measurements. The first dimension
            contains the specified spectral orders with the same ordering.
    """
    if param3d is not None:
        true_intensity, true_doppler, true_linewidth = param3d
    aa, bb = true_intensity.shape
    out = np.zeros((len(spectral_orders),)+(aa,bb))
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        if a == 0:
            out[z] = true_intensity.copy()
            continue
        for col in range(bb):
            for row in range(aa):
                out[z,:,col] += true_intensity[row,col] * gauss(
                    np.arange(aa)-row, a*true_doppler[row,col], abs(a)*true_linewidth[row,col]
                    )
    return out

def forward_op_torch(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    spectral_orders=[0,-1,1]):
    """
    Given 2d (or 3d where the first dimension is batch dim) arrays of intensity,
    doppler, and linewidth, calculate the noise free measurements at the
    specified spectral orders.

    Args:
        true_intensity (Tensor): 2d or 3d array of true intensities.
        true_doppler (Tensor): 2d or 3d array of true doppler shifts in the
            units of pixels.
        true_linewidth (Tensor): 2d or 3d array of true line widths in the units
            of pixels.
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (Tensor): 4d array of measurements. The first dimension is
        the batch dim, and the second dim contains the specified spectral orders
        with the same ordering.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(true_intensity.shape) == 2:
        true_intensity = true_intensity[None]
        true_doppler = true_doppler[None]
        true_linewidth = true_linewidth[None]
    k, aa, bb = true_intensity.shape
    out = torch.zeros((k,) + (len(spectral_orders),)+(aa,bb))
    out = out.to(device=device, dtype=torch.float)
    diffrange = torch.arange(aa)[None,None,:,None]-torch.arange(aa)[None,None,None,:]
    diffrange = diffrange.to(device=device, dtype=torch.float)
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        a = torch.Tensor([a]).to(device=device, dtype=torch.float)
        if a == 0:
            out[:,z] = true_intensity.clone()
            continue
        out[:,z] = torch.einsum(
            'lkij,lkj->lki',
            gauss_torch(
                diffrange, 
                a*true_doppler.permute(0,2,1)[:,:,None,:], 
                torch.abs(a)*true_linewidth.permute(0,2,1)[:,:,None,:]
            ), 
            true_intensity.permute(0,2,1)
        ).permute(0,2,1)
    return out

def forward_op_torch_loopy(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    spectral_orders=[0,-1,1]):
    """
    Given 2d (or 3d where the first dimension is batch dim) arrays of intensity,
    doppler, and linewidth, calculate the noise free measurements at the
    specified spectral orders.

    Warning: This is an old an slow implementation. 

    Args:
        true_intensity (Tensor): 2d or 3d array of true intensities.
        true_doppler (Tensor): 2d or 3d array of true doppler shifts in the
            units of pixels.
        true_linewidth (Tensor): 2d or 3d array of true line widths in the units
            of pixels.
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (Tensor): 4d array of measurements. The first dimension is
        the batch dim, and the second dim contains the specified spectral orders
        with the same ordering.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(true_intensity.shape) == 2:
        true_intensity = true_intensity[None]
        true_doppler = true_doppler[None]
        true_linewidth = true_linewidth[None]
    k, aa, bb = true_intensity.shape
    out = torch.zeros((k,) + (len(spectral_orders),)+(aa,bb))
    out = out.to(device=device, dtype=torch.float)
    # assume columns of detector are independent
    for z,a in enumerate(spectral_orders):
        a = torch.Tensor([a]).to(device=device, dtype=torch.float)
        if a == 0:
            out[:,z] = true_intensity.clone()
            continue
        for col in range(bb):
            for row in range(aa):
                out[:,z,:,col] += true_intensity[:,[row],col] * gauss_torch(
                    torch.arange(aa)[None].to(device=device, dtype=torch.float)-row, 
                    a*true_doppler[:,[row],col], 
                    torch.abs(a)*true_linewidth[:,[row],col]
                    )
    return out

def obj_ls(x, meas=None):
    aa, bb = meas.shape[1:]
    intensity, doppler, linewidth = np.reshape(x, (3,aa,bb))
    diff = forward_op(intensity, doppler, linewidth) - meas
    return np.sum(diff**2)

class Source():
    """
    A class for holding the source parameters.

    Args:
        inten (ndarray): 2d array of intensity values.
        vel (ndarray): 2d array of Doppler velocity values. Expects either the
            units of pixels with the pix=True, or [km/s] with pix=False argument.
        width (ndarray): 2d array of line width values. Expects either the units
            of pixels with the pix=True, or [A] with pix=False argument.
        wavelength (float): wavelenght of interest in [A].
        pix (bool): True if the input vel & width are given in pixel units;
            False otherwise.
    
    Attributes:
        inten
        vel
        width
        wavelength
        pix
        param3d (ndarray): 3d array of the stacked (inten, vel, width) arrays.
    """
    def __init__(
        self,
        *,
        inten=None,
        vel=None,
        width=None,
        wavelength=195.119,
        pix=False
    ):
        self.inten = inten
        self.vel = vel
        self.width = width
        self.wavelength = wavelength
        self.param3d = np.stack((inten, vel, width))
        self.pix = pix
    def plot(self, topix=True):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        i0=ax[0].imshow(self.inten, cmap='hot')
        ax[0].set_title('Intensity')
        fig.colorbar(i0, ax=ax[0])
        i1=ax[1].imshow(self.vel, cmap='seismic')
        fig.colorbar(i1, ax=ax[1])
        i2=ax[2].imshow(self.width, cmap='plasma')
        fig.colorbar(i2, ax=ax[2])
        if hasattr(self, 'pix'):
            ax[1].set_title('Velocity [pix]')
            ax[2].set_title('Linewidth [pix]')
        else:
            ax[1].set_title('Velocity [km/s]')
            ax[2].set_title('Linewidth [A]')
        plt.show()

class Imager():
    """
    The class to hold the imager-specific parameters and measurement functions.

    Args:
        pixel_size (float): detector pixel size in um (micrometers).
        dispersion (float): dispersion of the imager in um/mA.
        dispersion_scale (float): dispersion scale of the imager in mA/pixels. 
            Has a precedence over `dispersion` if both are specified.
        instrument_psf (ndarray): 2d array of the instrument psf to be convolved
            with the measurements. [Not Implemented]
        spectral_orders (list): list of integers specifying the spectral orders
            of the measurements. this also determines the number of measurements
        sampling_method (str): The string to specify how the detector is
            sampling the continuous superposed Gaussians. `impulse` refers to 
            the point sampling which was implemented in Davila2019. This is the
            only option implemented now.

    Attributes:
        pixel_size
        dispersion
        dispersion_scale
        spectral_orders
        srpix (Source): a Source instance in pixel units.
        meas3d (dict): dictionary holding the 2d measurement arrays with the 
            keys corresponding to spectral orders
        meas3dar (ndarray): 3d array of the measurements, whose order is the 
            same as the ordering in the `spectral_order` list.
    """
    def __init__(
        self,
        *,
        pixel_size=13.5, # um
        dispersion=1/1.65, # um/mA
        dispersion_scale=None, # mA/pixels
        instrument_psf=None,
        spectral_orders=[0,-1,1],
        sampling_method='impulse'
    ):
        self.pixel_size = pixel_size
        self.dispersion_scale = pixel_size / dispersion
        if dispersion_scale is not None:
            self.dispersion_scale = dispersion_scale
            self.dispersion = pixel_size / dispersion_scale
        self.spectral_orders = spectral_orders

    def topix(self, source):
        """
        Takes as input a Source object which has the physical units of 
        velocity and line width, and creates another Source object as an attribute
        of the Imager, which has these parameters in the pixel units.
        """
        assert source.pix == False, "Source object is already in pixel dimensions"
        self.srpix = Source(
            inten=source.inten,
            vel=source.vel*(source.wavelength/300/self.dispersion_scale),
            width=source.width/self.dispersion_scale*1000,
            wavelength=source.wavelength,
            pix=True
        )

    def get_measurements(
        self,
        sources,
    ):
        """
        Given a Source object, simulate and save measurements as an attribute 
        of the Imager object using the forward_op function.

        Args:
            sources (Source): a Source instance holding the source parameters
        """
        if sources.pix == False:
            self.topix(sources)
        else:
            self.srpix = sources
        meas3d = forward_op(
            param3d=self.srpix.param3d,
            spectral_orders=self.spectral_orders
        )
        self.meas3d = {}
        for ct,i in enumerate(self.spectral_orders):
            self.meas3d[str(i)] = meas3d[ct]
        self.meas3dar = meas3d

    def plot(self):
        fig, ax = plt.subplots(1,len(self.spectral_orders), figsize=(15,5))
        for i,a in enumerate(self.spectral_orders):
            im=ax[i].imshow(self.meas3d[str(a)], cmap='hot')
            ax[i].set_title('Order {}'.format(a))
            fig.colorbar(im, ax=ax[i])
        plt.show()


if __name__ == '__main__':
    aa, bb = (300,300)
    detector_size = (aa,bb)
    true_intensity = resize(camera(), detector_size)
    true_doppler = (resize(shepp_logan_phantom(), detector_size)-0.5)*0.3
    true_linewidth = 0.2*resize(cell(), detector_size)+0.5
    # true_doppler = 5*(true_intensity.copy() - 0.5)
    # true_linewidth = true_intensity.copy() * 5
    # true_intensity = np.zeros((aa,bb))
    # true_intensity[5] = 1
    spectral_orders=(-1,1)
    meas = forward_op(
        true_intensity=true_intensity,
        true_doppler=true_doppler,
        true_linewidth=true_linewidth,
        spectral_orders=spectral_orders
    )
    # amo = np.zeros((aa,bb)) + 0.05
    # x0 = np.stack( ( meas[0], amo, amo ), axis=0 ).flatten()
    # recon = minimize(obj_ls, x0, args=(meas,), method='Nelder-Mead',
    #     options={'disp':True, 'maxiter':1000, 'adaptive':True})
    # rec = recon.x.reshape(3,aa,bb)

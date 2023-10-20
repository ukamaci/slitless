import numpy as np
import torch
from scipy.special import erf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.data import camera, shepp_logan_phantom, cell
from scipy.optimize import minimize
from mas.decorators import _vectorize
from scipy.stats import poisson

def gauss(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * np.exp(-0.5*((x-mean)/sigma)**2)

def gauss_pix(x, mean, sigma):
    return 0.5 * (
        erf((x-mean+0.5)/(2**0.5*sigma)) -
        erf((x-mean-0.5)/(2**0.5*sigma))
    )

def gauss_torch(x, mean, sigma):
    return 1 / sigma / (2*np.pi)**0.5 * torch.exp(-0.5*((x-mean)/sigma)**2)

def gauss_pix_torch(x, mean, sigma):
    return 0.5 * (
        torch.erf((x-mean+0.5)/(2**0.5*sigma)) -
        torch.erf((x-mean-0.5)/(2**0.5*sigma))
    )

def forward_op(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    param3d=None,
    pixelated=False,
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
        pixelated (bool): if True, take the integral of Gaussian along a pixel
            instead of impulse sampling at the midpoint.
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (ndarray): 3d array of measurements. The first dimension
            contains the specified spectral orders with the same ordering.
    """
    gauss_func = gauss_pix if pixelated else gauss
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
            gauss_func(
                diffrange, 
                a*true_doppler.transpose(1,0)[:,np.newaxis,:], 
                abs(a)*true_linewidth.transpose(1,0)[:,np.newaxis,:]
            ), 
            true_intensity.transpose(1,0)
        ).transpose(1,0)
    return out

def forward_op_torch(
    true_intensity=None,
    true_doppler=None,
    true_linewidth=None,
    pixelated=False,
    spectral_orders=[0,-1,1],
    device=None):
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
        pixelated (bool): if True, take the integral of Gaussian along a pixel
            instead of impulse sampling at the midpoint.
        spectral_orders (list): list of the spectral orders.

    Returns:
        measurements (Tensor): 3d or 4d array of measurements. The first
        dimension is the batch dim (optional in case input arrays are 2d), and
        the second dim contains the specified spectral orders with the same
        ordering.
    """
    gauss_func = gauss_pix_torch if pixelated else gauss_torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == None:
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
            gauss_func(
                diffrange, 
                a*true_doppler.permute(0,2,1)[:,:,None,:], 
                torch.abs(a)*true_linewidth.permute(0,2,1)[:,:,None,:]
            ), 
            true_intensity.permute(0,2,1)
        ).permute(0,2,1)
    if len(true_intensity.shape) == 2:
        return out[0]
    else:
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
        pix (bool): set to True if the input vel & width are given in pixel
        units; False otherwise.
    
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
        stack = torch.stack if type(self.inten)==torch.Tensor else np.stack
        self.param3d = stack((inten, vel, width))
        self.pix = pix
    def plot(self, title='', ssims=None, rmses=None, psnrs=None):
        if type(self.inten)==torch.Tensor:
            inten, vel, width = self.inten.cpu().numpy(), self.vel.cpu().numpy(), self.width.cpu().numpy()
        else:
            inten, vel, width = self.inten, self.vel, self.width
        str_int = 'Intensity'
        str_vel = 'Velocity [pix]' if self.pix else 'Velocity [km/s]'
        str_width = 'Linewidth [pix]' if self.pix else 'Linewidth [A]'
        for name, field in zip(('SSIM', 'RMSE', 'PSNR'),(ssims, rmses, psnrs)):
            if field is None:
                continue
            str_int += '\n {}: {:.3f}'.format(name, field[0])
            str_vel += '\n {}: {:.3f}'.format(name, field[1])
            str_width += '\n {}: {:.3f}'.format(name, field[2])
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        plt.suptitle(title)
        i0=ax[0].imshow(inten, cmap='hot')
        ax[0].set_title(str_int)
        fig.colorbar(i0, ax=ax[0])
        i1=ax[1].imshow(vel, cmap='seismic')
        fig.colorbar(i1, ax=ax[1])
        ax[1].set_title(str_vel)
        i2=ax[2].imshow(width, cmap='plasma')
        fig.colorbar(i2, ax=ax[2])
        ax[2].set_title(str_width)
        plt.tight_layout()
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
        pixelated (bool): if True, when simulating the measurements take the 
            integral of Gaussian along a pixel instead of impulse sampling at 
            the midpoint.

    Attributes:
        pixel_size [um]
        dispersion [um/mA]
        dispersion_scale [mA/pixels]
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
        pixel_size=13.5, # um/pixel
        dispersion=1/1.65, # um/mA
        dispersion_scale=None, # mA/pixel
        instrument_psf=None,
        spectral_orders=[0,-1,1],
        pixelated=False
    ):
        self.pixel_size = pixel_size
        self.dispersion_scale = pixel_size / dispersion
        if dispersion_scale is not None:
            self.dispersion_scale = dispersion_scale
            self.dispersion = pixel_size / dispersion_scale
        self.spectral_orders = spectral_orders
        self.pixelated = pixelated

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

    def frompix(self, source):
        """
        Takes as input a Source object which has the pixel units of 
        velocity and line width, and creates another Source object as an attribute
        of the Imager, which has these parameters in the physical units.
        """
        assert source.pix == True, "Source object is already in physical dimensions"
        self.srphy = Source(
            inten=source.inten,
            vel=source.vel/(source.wavelength/300/self.dispersion_scale),
            width=source.width*self.dispersion_scale/1000,
            wavelength=source.wavelength,
            pix=False
        )

    def get_measurements(
        self,
        sources=None,
        dbsnr=None,
        max_count=None,
        model=None
    ):
        """
        Given a Source object, simulate and save measurements as an attribute 
        of the Imager object using the forward_op function.

        Args:
            sources (Source): a Source instance holding the source parameters
        """
        if sources==None:
            sources = self.srpix
        if sources.pix == False:
            self.topix(sources)
        else:
            self.srpix = sources

        fwd_op = forward_op_torch if type(sources.inten)==torch.Tensor else forward_op

        self.meas3dar = fwd_op(
            true_intensity=self.srpix.inten,
            true_doppler=self.srpix.vel,
            true_linewidth=self.srpix.width,
            spectral_orders=self.spectral_orders,
            pixelated=self.pixelated
        )

        if model is not None:
            self.meas3dar_nn = self.meas3dar.copy()
            self.meas3dar = add_noise(
                self.meas3dar, dbsnr=dbsnr, max_count=max_count, model=model
            )
        
        return self.meas3dar

    def plot(self, title='', noise=True):
        fig, ax = plt.subplots(1,len(self.spectral_orders), figsize=(15,5))
        plt.suptitle(title)
        for i,a in enumerate(self.spectral_orders):
            if (noise==False) and hasattr(self, 'meas3dar_nn'):
                im=ax[i].imshow(self.meas3dar_nn[i], cmap='hot')
            else:
                im=ax[i].imshow(self.meas3dar[i], cmap='hot')
            ax[i].set_title('Order {}'.format(a))
            fig.colorbar(im, ax=ax[i])
        plt.tight_layout()
        plt.show()

def add_noise(signal, dbsnr=None, max_count=None, model='Gaussian', no_noise=False):
    """
    Add noise to the given signal at the specified level.

    Args:
        signal (ndarray): noise-free input signal
        dbsnr (float): signal to noise ratio in dB: for Gaussian noise model, it is
        defined as the ratio of variance of the input signal to the variance of
        the noise. For Poisson model, it is taken as the average snr where snr
        of a pixel is given by the square root of its value.
        max_count (int): Max number of photon counts in the given signal
        model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`
        no_noise (bool): (default=False) If True, return the clean signal

    Returns:
        ndarray that is the noisy version of the input
    """
    if no_noise is True:
        return signal
    assert model.lower() in ('gaussian', 'poisson'), "invalid noise model"

    sig = signal.cpu().numpy() if type(signal)==torch.Tensor else signal

    if model.lower() == 'gaussian':
        var_sig = np.var(sig, axis=(-1,-2), keepdims=True)
        std_noise = (var_sig / 10**(dbsnr / 10))**0.5
        out = np.random.normal(sig, std_noise)
    elif model.lower() == 'poisson':
        if max_count is not None:
            scalar = max_count / np.max(sig, axis=(-1,-2), keepdims=True)
            sig_scaled = sig * scalar
            # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
            out = poisson.rvs(sig_scaled) / scalar
        else:
            avg_brightness = 10**(dbsnr / 20)**2
            scalar = avg_brightness / sig.mean(axis=(-1,-2), keepdims=True)
            sig_scaled = sig * scalar
            out = poisson.rvs(sig_scaled) / scalar

    if type(signal)==torch.Tensor:
        out = torch.from_numpy(out).to(device=signal.device, dtype=signal.dtype)

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
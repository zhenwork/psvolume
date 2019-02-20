import numpy as np
import h5py
from scipy.ndimage.filters import median_filter


def make_2d_radius(size, center=None):
    (nx, ny) = size
    if center is None:
        cx = (nx-1.)/2.
        cy = (ny-1.)/2.
        center = (cx,cy)
    x = np.arange(nx) - center[0]
    y = np.arange(ny) - center[1]
    [xaxis, yaxis] = np.meshgrid(x, y, indexing='ij')
    radius = np.sqrt(xaxis**2 + yaxis**2)
    return radius


def make_circle_region(size, center=None, rmax=10, rmin=0):
    image = np.ones(size).astype(int)
    r = make_2d_radius(size, center=center)

    index = np.where(r < rmin**2)
    image[index] = 0

    index = np.where(r > rmax**2)
    image[index] = 0

    return image


def solid_angle_correction(image, Geo):
    detDistance = Geo['detDistance']
    pixelSize   = Geo['pixelSize']
    center = Geo['center']

    (nx, ny) = image.shape
    x = np.arange(nx) - center[0]
    y = np.arange(ny) - center[1]
    [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij")
    xaxis = xaxis.ravel()
    yaxis = yaxis.ravel()
    zaxis = np.ones(nx*ny)*detDistance/pixelSize
    norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    ascale = zaxis/norm**3
    ascale /= np.amax(ascale)
    ascale.shape = (nx,ny)
    return ascale

def polarization_correction(image, Geo):
    ## p=1 means y polarization. p=-1 means x polarization
    detDistance  = Geo['detDistance']
    pixelSize    = Geo['pixelSize']
    polarization = Geo['polarization']
    center = Geo['center']

    (nx, ny) = image.shape
    x = np.arange(nx) - center[0]
    y = np.arange(ny) - center[1]
    [xaxis, yaxis] = np.meshgrid(x, y,indexing="ij")
    xaxis = xaxis.ravel()
    yaxis = yaxis.ravel()
    zaxis = np.ones(nx*ny)*detDistance/pixelSize
    norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    
    if polarization is not None:
        pscale = (2.*zaxis**2 + (1+polarization)*xaxis**2 + (1-polarization)*yaxis**2 )/(2.*norm**2)
        print 'new polzaization correction'
    else: 
        pscale = np.ones(image.shape)

    pscale /= np.amax(pscale)
    pscale.shape = (nx,ny)
    return pscale

def remove_peak_alg1(img, _mask=None, sigma=15, cwin=(11,11)):
    """
    First throw away \pm sigma*std. Second throw away \pm sigma*std
    """
    
    if _mask is None: mask=np.ones(img.shape)
    else: mask = _mask.copy()
    
    index = np.where(img<0)
    mask[index] = 0
    
    image = img*mask
    median = median_filter(image, cwin)*mask
    submedian = image - median
    
    Tindex = np.where(mask==1)
    Findex = np.where(mask==0)
    
    ave = np.mean(submedian[Tindex])
    std = np.std( submedian[Tindex])
    index = np.where((submedian>ave+std*sigma)+(submedian<ave-std*sigma)==True)
    image[index] = -1
    submedian[index] = 0
    
    ave = np.mean(submedian[Tindex])
    std = np.std( submedian[Tindex])
    index = np.where((submedian>ave+std*sigma)+(submedian<ave-std*sigma)==True)
    image[index] = -1
    
    image[Findex] = -1
    
    return image

def remove_peak_alg2(img, mask=None, thr=(None, None), cwin=(11,11)):
    """
    Use a simple cut off method
    """
    if mask is None: mask=np.ones(img.shape)
    image = img*mask
    median = median_filter(image, cwin)*mask
    submedian = image - median

    if thr[0] is not None:
        index = np.where(submedian<thr[0])
        image[index] = -1
    if thr[1] is not None:
        index = np.where(submedian>thr[1])
        image[index] = -1
    return image

def medianf(image, mask=1., window=(5,5)):
    median = median_filter(image, window)*mask

def meanf(image, mask=None, window=(5,5)):
    (nx,ny) = image.shape
    if mask is None: mask = np.ones(image.shape)
    ex = (window[0]-1)/2
    ey = (window[1]-1)/2
    sx = ex*2+1
    sy = ey*2+1
    Data = np.zeros((sx*sy, nx+ex*2, ny+ey*2));
    Mask = np.zeros(data.shape);

    for i in range(sx):
        for j in range(sy):
            Data[i*sy+j, i:(i+nx), j:(j+ny)] = image.copy()
            Mask[i*sy+j, i:(i+nx), j:(j+ny)] = mask.copy()

    Mask = np.sum(Mask, axis=0);
    Data = np.sum(Data, axis=0);
    index = np.where(Mask>0);
    Data[index] /= Mask[index]
    return Data

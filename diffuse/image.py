import numpy as np
from numba import jit
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

    
class Masktbk:
    @staticmethod
    def circle_mask_keep(size,rmin=None,rmax=None,center=None):
        """
        size = (nx,ny), tuple
        """
        (nx, ny) = size
        if center is None: 
            cx=(nx-1.)/2.
            cy=(ny-1.)/2.
        else:
            (cx,cy) = center

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        xaxis, yaxis = np.meshgrid(x,y, indexing="ij")
        r = np.sqrt(xaxis**2+yaxis**2)

        mask = np.ones(size).astype(int)
        if rmin is not None:
            mask *= (r >= rmin).astype(int)
        if rmax is not None:
            mask *= (r <= rmax).astype(int)
        return mask

    @staticmethod
    def value_mask_keep(image, vmin=None, vmax=None):
        mask = np.ones(image.shape).astype(int) 
        if vmin is not None:
            mask *= (image>=vmin).astype(int)
        if vmax is not None:
            mask *= (image<=vmax).astype(int)
        return mask

    @staticmethod
    def expand_mask_reject(mask, expand_size=(1,1)):
        """
        expandSize is the half size of window
        """
        assert len(mask.shape) == len(expand_size)
        import scipy.ndimage
        kernel = np.ones([x*2+1 for x in expand_size]).astype(int)
        return (scipy.ndimage.convolve(mask, kernel, mode="mirror")==np.sum(kernel)).astype(int)


class Correctiontbx:

    @staticmethod
    def solid_angle_multiplier(size=None, detector_distance_mm=None, pixel_size_mm=None, detector_center_px=None):
        """
        Params: detectorDistance, pixelSize must have the same unit
        Returns: scaleMask -> image *= scaleMask
        Note: scaleMask -> min=1 (at center), value increases to the detector edge
        """
        (nx, ny) = size
        (cx, cy) = detector_center_px
        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        xaxis, yaxis = np.meshgrid(x, y, indexing="ij") 
        zaxis = np.ones((nx,ny))*detector_distance_mm/pixel_size_mm
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        solidAngle = zaxis * 1.0 / norm**3
        solidAngle /= np.amax(solidAngle)
        scaleMask = 1./solidAngle
        return scaleMask

    @staticmethod
    def polarization_multiplier(size=None, polarization_fr=-1, detector_distance_mm=None, pixel_size_mm=None, detector_center_px=None):
        """
        p =1 means y polarization
        p=-1 means x polarization
        # Default is p=-1 (x polarization)
        # Note: scaleMask -> min=1
        """
        (nx, ny) = size
        x = np.arange(nx) - center[0]
        y = np.arange(ny) - center[1]
        xaxis, yaxis = np.meshgrid(x, y, indexing="ij") 
        zaxis = np.ones((nx,ny))*detector_distance_mm/pixel_size_mm
        norm  = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        
        if polarization_fr is not None:
            detectScale = (2.*zaxis**2 + (1+polarization_fr)*xaxis**2 + (1-polarization_fr)*yaxis**2 )/(2.*norm**2)
        else: 
            detectScale = np.ones(size)

        detectScale /= np.amax(detectScale)
        scaleMask = 1. / detectScale
        return scaleMask

    @staticmethod
    def detector_absorption_multiplier(size, detector_distance_mm=None, pixel_size_mm=None, detector_center_px=None):
        (nx, ny) = size
        x = np.arange(nx) - center[0]
        y = np.arange(ny) - center[1]
        [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij") 
        zaxis = np.ones((nx,ny))*detectorDistance/pixelSize
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)

        delta = 1.99528
        thickness_mm = 0.32
        cos_angle = zaxis / norm
        E = 1. - np.exp( - thickness_mm * delta / cos_angle)
        return 1. / E / np.amin(1./E)

    @staticmethod
    def detector_parabox_multiplier(size, detector_distance_mm=None, pixel_size_mm=None, detector_center_px=None):
        return 


class Filtertbx:
    @staticmethod
    def median_filter(image=None,mask=None,window=(11,11)):
        ## currently the mask doesn't work
        from scipy.ndimage.filters import median_filter
        median = median_filter(image, window, mode="mirror") * 1.0
        return median

    @staticmethod
    def mean_filter(image=None,mask=None,window=(5,5)):
        assert len(window)==len(image.shape)
        import scipy.ndimage
        kernel = np.ones(window)
        if mask is not None:
            sum_image = scipy.ndimage.convolve(image * mask, kernel, mode="mirror")
            sum_mask = scipy.ndimage.convolve(mask, kernel, mode="mirror")
            index = np.where(sum_mask>0)
            sum_image[index] /= 1.0 * sum_mask[index]
            return sum_image * mask 
        else:
            sum_image = scipy.ndimage.convolve(image, kernel, mode="mirror")
            return sum_image * 1.0 / np.prod(window)

    @staticmethod
    def std_filter(image=None,mask=None,window=(5,5)):
        # sigma = sqrt( E(x^2) - E(x)^2 )
        assert len(window)==len(image.shape)
        import scipy.ndimage
        kernel = np.ones(window)
        if mask is not None:
            sum_image  = scipy.ndimage.convolve(image * mask,    kernel, mode="mirror")
            sum_square = scipy.ndimage.convolve(image**2 * mask, kernel, mode="mirror") 
            sum_mask   = scipy.ndimage.convolve(mask,            kernel, mode="mirror")
            index = np.where(sum_mask>0)
            sum_image[index] /= 1.0 * sum_mask[index]
            sum_square[index] /= 1.0 * sum_mask[index]
            return np.sqrt( sum_square*mask - (sum_image*mask)**2)
        else:
            sum_image  = scipy.ndimage.convolve(image ,    kernel, mode="mirror")
            sum_square = scipy.ndimage.convolve(image**2,  kernel, mode="mirror")
            return np.sqrt( sum_square - sum_image**2 )


def remove_extreme(image=None,algorithm=2,mask=None,sigma=15,vmin=None,vmax=None,window=(11,11)):
    """
    1. Throw away {+,-}sigma*std from (image - median)
    2. Throw away {+,-}sigma*std again from (image - median)
    """
    if algorithm == 1:
        raise Exception("## old method, not using now")
    if algorithm == 2:
        # _image=None, algorithm=1, _mask=None, _sigma=15, _vmin=None, _vmax=None, _window=(11,11)
        # without using the median background
        if mask is not None:
            imask = mask.copy() * Masktbk.value_mask_keep(image, vmin=None, vmax=None)
        else:
            imask = np.ones(image.shape).astype(int) * Masktbk.value_mask_keep(image, vmin=None, vmax=None)
        
        idata = image * imask
        mean_idata = Filtertbx.mean_filter(image=idata,mask=imask,window=window)
        std_idata  = Filtertbx.std_filter(image=idata,mask=imask,window=window)
        imask *= (idata >= mean_idata - sigma*std_idata)
        imask *= (idata <= mean_idata + sigma*std_idata)
        return idata * imask, imask
    else:
        return None

@jit
def angularDistri(arr, Arange=None, num=30, rmax=None, rmin=None, center=(None,None)):
    """
    # num denotes how many times you want to divide the full angle (360 degree)
    # This function is slow because it applies multiple for loops
    """
    assert len(arr.shape)==2
    (nx, ny) = arr.shape
    cx = center[0];
    cy = center[1];
    if cx is None: cx = (nx-1.)/2.
    if cy is None: cy = (ny-1.)/2.

    xaxis = np.arange(nx)-cx + 1.0e-5; 
    yaxis = np.arange(ny)-cy + 1.0e-5; 
    [x,y] = np.meshgrid(xaxis, yaxis, indexing='ij')
    r = np.sqrt(x**2+y**2)
    sinTheta = y/r;
    cosTheta = x/r; 
    angle = np.arccos(cosTheta);
    index = np.where(sinTheta<0);
    angle[index] = 2*np.pi - angle[index]
    if rmin is not None:
        index = np.where(r<rmin);
        angle[index] = -1
    if rmax is not None:
        index = np.where(r>rmax);
        angle[index] = -1
    if Arange is not None:
        index = np.where((angle>Arange[0]*np.pi/180.)*(angle<Arange[1]*np.pi/180.)==True);
        subData = arr[index].copy()
        aveIntens = np.mean(subData)
        aveAngle = (Arange[0]+Arange[1])/2.        
        return [aveAngle, aveIntens];

    rad = np.linspace(0, 2*np.pi, num+1);
    aveIntens = np.zeros(num)
    aveAngle = np.zeros(num)
    for i in range(num):
        index = np.where((angle>rad[i])*(angle<rad[i+1])==True);
        subData = arr[index].copy()
        aveIntens[i] = np.mean(subData)
        aveAngle[i] = (rad[i]+rad[i+1])/2.
    return [aveAngle, aveIntens]

@jit
def radial_profile(image, mask, center=None, vmin=None, vmax=None, rmin=None, rmax=None, stepSize=None, sampling=None, window=3):
    """
    # mask = 0 will be ignored
    # pixel value beyond (vmin, vmax) will be ignored
    # radius beyong (rmin, rmax) will be ignored
    # stepSize=1 is normally set
    # sampling is the number of radius points to collect
    # if stepSize is set by user, then sampling will be ignored
    # returns: aveRadius, aveIntens, sumCount
    """
    (nx, ny) = image.shape
    if center is None: 
        cx = (nx-1.)/2.
        cy = (ny-1.)/2.
    else:
        cx = center[0]
        cy = center[1]

    x = np.arange(nx)-cx 
    y = np.arange(ny)-cy 
    xaxis,yaxis = np.meshgrid(x, y, indexing='ij')
    radius = np.sqrt(xaxis**2+yaxis**2)

    if rmin is None: 
        rmin = np.amin(radius)
    if rmax is None:
        rmax = np.amax(radius)
    if stepSize is not None:
        aveRadius = np.arange(rmin, rmax+stepSize/2., stepSize)
    elif sampling is not None:
        aveRadius = np.linspace(rmin, rmax, sampling)
        stepSize = (rmax - rmin)/(sampling - 1.)
    else:
        aveRadius = np.arange( int(round(rmin)), int(round(rmax))+1 )
        stepSize = 1.
        
    # print "stepSize = ",np.around(stepSize,2)
    # print "rmin/rmax = ", np.around(aveRadius[0]),np.around(aveRadius[-1])

    notation = mask.copy()
    notation[radius < rmin] = 0
    notation[radius >= rmax] = 0 
    if vmax is not None:
        notation[image >= vmax] = 0   
    if vmin is not None:
        notation[image < vmin] = 0
    
    radius = np.around(radius / stepSize).astype(int)
    startR = int(np.around(rmin / stepSize))
    sumIntens = np.zeros(len(aveRadius))
    sumCount  = np.zeros(len(aveRadius))
    aveIntens = np.zeros(len(aveRadius))

    hwindow = int((window-1)/2.)
    
    for idx in range(nx):
        for jdx in range(ny):
            r = radius[idx, jdx]
            if notation[idx, jdx] == 0:
                continue
            #sumIntens[r-startR] += image[idx,jdx] * notation[idx,jdx]
            #sumCount[r-startR] += notation[idx,jdx]
            
            for h in range(-hwindow, hwindow+1):
                if r - startR + h <= len(sumIntens)-1 and r - startR + h >= 0:
                    sumIntens[r-startR+h] += image[idx,jdx] * notation[idx,jdx]
                    sumCount[r-startR+h] += notation[idx,jdx]     
            
    index = np.where(sumCount > 10)
    aveIntens[index] = sumIntens[index] * 1.0 / sumCount[index]

    return aveRadius, aveIntens, sumCount

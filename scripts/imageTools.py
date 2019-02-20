import numpy as np
from scipy.ndimage.filters import median_filter

class MaskTools:
    def circleMask(self, size, rmin=None, rmax=None, center=None):
        """
        size = (nx,ny), tuple
        """
        (nx, ny) = size
        if center is None: 
            cx=(nx-1.)/2.
            cy=(ny-1.)/2.
        else:
            (cx,cy) = center
        if rmin is None:
            rmin = -1
        if rmax is None:
            rmax = max(size)*2.

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        [xaxis, yaxis] = np.meshgrid(x,y, indexing="ij")
        r = np.sqrt(xaxis**2+yaxis**2)

        mask = np.zeros(size).astype(int)
        index = np.where((r <= rmax) & (r >= rmin))
        mask[index] = 1
        return mask

    def valueLimitMask(self, image, vmin=None, vmax=None):
        mask = np.zeros(image.shape).astype(int)
        (_vmin, _vmax) = (np.amin(image), np.amax(image))
        if vmin is None:
            vmin = _vmin - 1
        if vmax is None:
            vmax = _vmax + 1
        index = np.where((image>=vmin) & (image<=vmax))
        mask[index] = 1
        return mask

    def expandMask(self, mask, expandSize=(1,1), expandValue=0):
        """
        expandSize is the half size of window
        """
        (nx,ny) = mask.shape
        newMask = mask.copy()
        index = np.where(mask==expandValue)
        for i in range(-expandSize[0], expandSize[0]+1):
            for j in range(-expandSize[1], expandSize[1]+1):
                newMask[((index[0]+i)%nx, (index[1]+j)%ny)] = value
        return newMask


class ScalingTools:
    def solid_angle_scaler(self, size=None, detectorDistance=None, pixelSize=None, center=None):
        """
        Params: detectorDistance, pixelSize must have the same unit
        Returns: scaleMask -> image *= scaleMask
        Note: scaleMask -> min=1 (at center), value increases to the detector edge
        """
        (nx, ny) = size
        (cx, cy) = center
        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij") 
        zaxis = np.ones((nx,ny))*detectorDistance/pixelSize
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        solidAngle = zaxis * 1.0 / norm**3
        solidAngle /= np.amax(solidAngle)
        solidAngle.shape = (nx, ny)
        scaleMask = 1./solidAngle
        return scaleMask

    def polarization_scaler(self, size=None, polarization=-1, detectorDistance=None, pixelSize=None, center=None):
        """
        p =1 means y polarization
        p=-1 means x polarization
        # Default is p=-1 (x polarization)
        # Note: scaleMask -> min=1
        """
        (nx, ny) = size
        x = np.arange(nx) - center[0]
        y = np.arange(ny) - center[1]
        [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij") 
        zaxis = np.ones((nx,ny))*detectorDistance/pixelSize
        norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        
        if polarization is not None:
            detectScale = (2.*zaxis**2 + (1+polarization)*xaxis**2 + (1-polarization)*yaxis**2 )/(2.*norm**2)
        else: 
            detectScale = np.ones(size)

        detectScale /= np.amax(detectScale)
        detectScale.shape = (nx, ny)
        scaleMask = 1. / detectScale
        return scaleMask

class FilterTools:
    def median_filter(self, image=None, mask=None, window=(5,5)):
        median = median_filter(image, window) * 1.0 * mask
        return median

    def mean_filter(self, image=None, mask=None, window=(5,5)):
        (nx,ny) = image.shape
        if mask is None: 
            mask = np.ones(image.shape).astyep(int)
        ex = (window[0]-1)/2
        ey = (window[1]-1)/2
        sx = ex*2+1
        sy = ey*2+1
        Data = np.zeros((sx*sy, nx+ex*2, ny+ey*2));
        Mask = np.zeros(data.shape);

        for i in range(sx):
            for j in range(sy):
                Data[i*sy+j, i:(i+nx), j:(j+ny)] = image * mask
                Mask[i*sy+j, i:(i+nx), j:(j+ny)] = mask.copy()

        Mask = np.sum(Mask, axis=0);
        Data = np.sum(Data, axis=0);
        index = np.where(Mask>0);
        Data[index] /= 1.0 * Mask[index]

        Mask = None
        index = None
        return Data * 1.0 * mask


def removeExtremes(_image, algorithm=1, _mask=None, _sigma=15, _vmin=0, _vmax=None, _window=(11,11)):
    """
    1. Throw away {+,-}sigma*std from (image - median)
    2. Throw away {+,-}sigma*std again from (image - median)
    """
    if algorithm == 1:
        if _mask is None: 
            mask=np.ones(_image.shape).astype(int)
        else: 
            mask = _mask.astype(int)
        
        image = _image * mask
        mt = MaskTools()
        ft = FilterTools()

        ## remove value >vmax or <vmin
        mask  *= mt.valueLimitMask(image, vmin=_vmin, vmax=_vmax)
        image *= mask
        
        ## remove values {+,-}sigma*std
        median = ft.median_filter(image=image, mask=mask, window=_window)
        submedian = image - median
        
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>ave+std*sigma) | (submedian<ave-std*sigma))
        image[index] = 0
        mask[index] = 0

        submedian[index] = 0
        
        ## remove values {+,-}sigma*std
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>ave+std*sigma) | (submedian<ave-std*sigma))
        image[index] = 0
        mask[index] = 0
        
        return image, mask

    elif algorithm == 2:
        return None
    else:
        return None


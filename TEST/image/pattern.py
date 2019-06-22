import numpy as np
import image.filter as filtertools
import image.mask as masktools
from numba import jit

def removeExtremes(_image=None, algorithm=1, _mask=None, _sigma=15, _vmin=0, _vmax=None, _window=(11,11)):
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

        ## remove value >vmax or <vmin
        mask  *= masktools.value_mask(image, vmin=_vmin, vmax=_vmax)
        image *= mask
        
        ## remove values {+,-}sigma*std
        median = filtertools.median_filter(image=image, mask=mask, window=_window)
        submedian = image - median
        # tmp* submedian *= mask
        
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>=ave+std*_sigma) | (submedian<ave-std*_sigma))
        image[index] = 0
        mask[index] = 0
        submedian[index] = 0
        
        ## remove values {+,-}sigma*std
        Tindex = np.where(mask==1)
        Findex = np.where(mask==0)
        ave = np.mean(submedian[Tindex])
        std = np.std( submedian[Tindex])
        index = np.where((submedian>=ave+std*_sigma) | (submedian<ave-std*_sigma))
        image[index] = 0
        mask[index] = 0
        
        return image, mask
    
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
def radialProfile(image, mask, center=None, vmin=None, vmax=None, rmin=None, rmax=None, \
                    stepSize=None, sampling=None, window=3):
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
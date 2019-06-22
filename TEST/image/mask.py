import numpy as np


def make_mask(size, rmin=None, rmax=None, setvalue=0, center=None):
    """
    size = (nx,ny), tuple
    """
    (nx, ny) = size
    if center is None: 
        cx=(nx-1.)/2.
        cy=(ny-1.)/2.
    else:
        (cx,cy) = center

    mask = np.zeros(size).astype(int) + int(1-setvaule)

    if rmin is not None or rmax is not None:
        if rmin is None:
            rmin = -1
        if rmax is None:
            rmax = max(size)*2.

        x = np.arange(nx) - cx
        y = np.arange(ny) - cy
        [xaxis, yaxis] = np.meshgrid(x,y, indexing="ij")
        r = np.sqrt(xaxis**2+yaxis**2)

        index = np.where((r < rmax) & (r >= rmin))
        mask[index] = setvalue

    return mask


def value_mask(image, vmin=_vmin, vmax=_vmax):
    mask = np.ones(image.shape).astype(int)
    index = np.where( (image>=vmax) | (image<vmin) )
    mask[index] = 0
    return mask


def expand_mask(mask, expsize=(3,3), expvalue=0):
    """
    expsize is the half size of window
    """
    (nx,ny) = mask.shape
    newMask = mask.copy()
    index = np.where(mask==expvalue)
    expx = (expsize[0]-1)/2
    expy = (expsize[1]-1)/2
    for i in range(-expx, expx+1):
        for j in range(-expy, expy+1):
            newMask[((index[0]+i)%nx, (index[1]+j)%ny)] = expvalue
    return newMask


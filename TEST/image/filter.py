import numpy as np
from scipy.ndimage.filters import median_filter

def median_filter(image=None, mask=None, window=(11,11)):
    median = median_filter(image, window) * 1.0
    if mask is not None:
        median *= mask
    return median


def mean_filter(image=None, mask=None, window=(5,5)):
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
import numpy as np
from numba import jit
from fileManager import *
from imageMergeClient import Geometry, make_radius

def remove_bragg_peak(image, Geo):

    voxel = Geometry(image, Geo)

    Image = image.ravel()
    Rot = Geo['rotation']
    HKL = (Rot.dot(voxel)).T

    H = HKL[:,0];
    K = HKL[:,1];
    L = HKL[:,2];

    Hshift = np.abs(H-np.around(H));
    Kshift = np.abs(K-np.around(K));
    Lshift = np.abs(L-np.around(L));

    index = np.where(((Hshift<0.25)*(Kshift<0.25)*(Lshift<0.25))==True);
    Image[index] = -1.
    Image.shape = image.shape

    return Image

@jit
def get_Rindex(size, center=None, depth=3):

    radius = make_radius(size, center=center)
    radius = np.around(radius).astype(int)
    half = (depth-1)/2;
    Rindex = [None]
    rmax = np.amax(radius)
    for r in range(rmax+1):
        index = np.where((radius>=r-half)*(radius<=r+half)==True)
        Rindex.append(index)
    Rindex.pop(0)
    return Rindex

@jit
def remove_peak_alg3(image, Rindex):
    indexMark = (image>=0)
    Image = image.copy()
    rmax = len(Rindex)-1
    for r in range(rmax+1):
        index = Rindex[r]; 
        if len(index[0]) < 16:
            Image[index] = -1024;
            continue
        else:
            Image[index] -= np.sum(image[index]*indexMark[index])*1.0/np.sum(indexMark[index]);
    Image[np.where(image<0)] = -1024;
    return Image
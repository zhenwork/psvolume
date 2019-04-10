import numpy as np  
import h5py

def Geometry(image, Geo):
    """
    The unit of wavelength is A
    """
    waveLength = Geo['waveLength']
    center = Geo['center']

    (nx, ny) = image.shape
    x = np.arange(nx) - center[0]
    y = np.arange(ny) - center[1]
    [xaxis, yaxis] = np.meshgrid(x, y)
    xaxis = xaxis.T.ravel()
    yaxis = yaxis.T.ravel()
    zaxis = np.ones(nx*ny)*Geo['detDistance']/Geo['pixelSize']
    norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    ## The first axis is negative
    voxel = (np.array([xaxis,yaxis,zaxis])/norm - np.array([[0.],[0.],[1.]]))/waveLength
    return voxel


def Image2hkl(fileName):

    f = h5py.File(fileName, "r")
    image = f["image"].value
    Geo = {}
    Geo["waveLength"] = f["waveLength"].value
    Geo["pixelSize"] = f["pixelSize"].value
    Geo["detDistance"] = f["detDistance"].value
    Geo["center"] = f["center"].value
    f.close()

    voxel = Geometry(image, Geo)
    Image = image.ravel()
    Rot = Geo['rotation']
    HKL = (Rot.dot(voxel)).T
    HKL = np.around(HKL).astype(int)

    h = HKL[:,0].copy()
    k = HKL[:,0].copy()
    l = HKL[:,0].copy()
    ######
    h.shape = image.shape
    k.shape = image.shape
    l.shape = image.shape
    ######

    return h,k,l
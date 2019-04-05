import numpy as np 
from numba import jit
import mathTools

def mapPixel2RealXYZ(size=None, center=None, pixelSize=None, detectorDistance=None):
    """
    Input: only 2D size are accepted
    Returns: Real space pixel position (xyz): N*N*3 numpy array
    xyz: unit (meter)
    """
    (nx, ny) = size
    (cx, cy) = center
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    xaxis, yaxis = np.meshgrid(x, y, indexing="ij")
    zaxis = np.ones((nx,ny))
    xyz = np.stack([xaxis*pixelSize, yaxis*pixelSize, zaxis*detectorDistance], axis=2)
    return xyz


def mapRealXYZ2Reciprocal(xyz=None, waveLength=None):
    """
    The unit of wavelength is A
    xyz: Real space pixel position (xyz): N*N*3 numpy array
    xyz: unit (meter)
    reciprocal: N*N*3 numpy array
    """
    if len(xyz.shape)==2:
        norm = np.sqrt(np.sum(xyz**2, axis=1, keepdims=True))
        ## scaled array doesn't have units
        scaled = xyz/norm
        scaled[:,2] -= 1.0
        reciprocal = scaled / waveLength
        return reciprocal
    elif len(xyz.shape)==3:
        norm = np.sqrt(np.sum(xyz**2, axis=2, keepdims=True))
        ## scaled array doesn't have units
        scaled = xyz/norm
        scaled[:,:,2] -= 1.0
        reciprocal = scaled / waveLength
        return reciprocal
    else:
        return None


def mapReciprocal2Voxel(Amat=None, Bmat=None, returnFormat="HKL", \
                    reciprocal=None, voxelSize=1.0, Phi=0., rotAxis="x"):
    """
    Return: voxel (N*N*3 or N*3) 
    voxelSize: 0.015 for 'cartesian' coordinate; 1.0 for "hkl" coordinate
    """
    Phimat = mathTools.quaternion2rotation(mathTools.phi2quaternion(Phi, rotAxis=rotAxis))
    if returnFormat.lower() == "hkl":
        voxel = reciprocal.dot(np.linalg.inv(Phimat.dot(Amat)).T) / voxelSize 
    elif returnFormat.lower() == "cartesian":
        voxel = reciprocal.dot(np.linalg.inv(Phimat.dot(Amat)).T).dot(Bmat.T) / voxelSize 
    return voxel


def mapImage2Voxel(image=None, size=None, Amat=None, Bmat=None, xvector=None, Phi=0., \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                returnFormat=None, voxelSize=1.0, rotAxis="x"):

    """
    # This function combines mapPixel2RealXYZ, mapRealXYZ2Reciprocal and mapReciprocal2Voxel. 
    # Input: real 2D image in N*N
    # Output: voxel in N*N*3 shape
    """
    if image is not None:
        size = image.shape

    if xvector is None:
        xyz = mapPixel2RealXYZ(size=size, center=center, pixelSize=pixelSize, \
                                detectorDistance=detectorDistance)
        reciprocal = mapRealXYZ2Reciprocal(xyz=xyz, waveLength=waveLength)
    else:
        reciprocal = xvector

    voxel = mapReciprocal2Voxel(Amat=Amat, Bmat=Bmat, returnFormat="HKL", rotAxis=rotAxis, \
                            reciprocal=reciprocal, voxelSize=1.0, Phi=Phi)

    return voxel


@jit
def PeakMask(Amat=None, _image=None, size=None, xvector=None, window=(0, 0.25), hRange=(-1000,1000), kRange=(-1000,1000), lRange=(-1000,1000), \
            waveLength=None, pixelSize=None, center=None, detectorDistance=None, Phi=0., rotAxis="x"):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    """
    voxel = mapImage2Voxel(image=_image, size=size, Amat=Amat, xvector=xvector, \
            Phi=Phi, waveLength=waveLength, pixelSize=pixelSize, center=center, rotAxis=rotAxis, \
            detectorDistance=detectorDistance)

    ## For Loop to map one image
    if size is None:
        size = _image.shape

    Npixels = np.prod(size)
    peakMask = np.zeros(Npixels).astype(int)
    voxel =  voxel.reshape((Npixels, 3)) 
    shift = np.abs(np.around(voxel) - voxel)

    for t in range(Npixels):
        
        hshift = shift[t, 0]
        kshift = shift[t, 1]
        lshift = shift[t, 2]

        hh = voxel[t, 0]
        kk = voxel[t, 1]
        ll = voxel[t, 2]

        if (hshift>=window[1]) or (kshift>=window[1]) or (lshift>=window[1]):
            continue
        if (hshift<window[0]) and (kshift<window[0]) and (lshift<window[0]):
            continue
        if hh < hRange[0] or hh >= hRange[1]:
            continue
        if kk < kRange[0] or kk >= kRange[1]:
            continue
        if ll < lRange[0] or ll >= lRange[1]:
            continue
        
        peakMask[t] = 1

    return peakMask.reshape(size)


@jit
def Image2Volume(volume=None, weight=None, Amat=None, Bmat=None, _image=None, _mask=None, \
                keepPeak=False, returnFormat="HKL", xvector=None, window=(0.25, 0.5), \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                Vcenter=60, Vsize=121, voxelSize=1., Phi=0., rotAxis="x"):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    """
    voxel = mapImage2Voxel(image=_image, size=None, Amat=Amat, Bmat=Bmat, xvector=xvector, \
            Phi=Phi, waveLength=waveLength, pixelSize=pixelSize, center=center, rotAxis=rotAxis, \
            detectorDistance=detectorDistance, returnFormat=returnFormat, voxelSize=voxelSize)

    ## For Loop to map one image
    Npixels = np.prod(_image.shape)
    image = _image.ravel()
    mask  =  _mask.ravel()
    voxel =  voxel.reshape((Npixels, 3)) 
    
    if volume is None:
        volume = np.zeros((Vsize, Vsize, Vsize))
        weight = np.zeros((Vsize, Vsize, Vsize))
        
    for t in range(Npixels):

        if mask[t] == 0:
            continue
        
        hkl = voxel[t] + Vcenter
        
        h = hkl[0] 
        k = hkl[1] 
        l = hkl[2] 
        
        inth = int(round(h)) 
        intk = int(round(k)) 
        intl = int(round(l)) 

        if (inth<0) or inth>(Vsize-1) or (intk<0) or intk>(Vsize-1) or (intl<0) or intl>(Vsize-1):
            continue
        
        hshift = abs(h-inth)
        kshift = abs(k-intk)
        lshift = abs(l-intl)

        ## window[0] <= box < window[1]
        if (hshift>=window[1]) or (kshift>=window[1]) or (lshift>=window[1]):
            continue
        if (hshift<window[0]) and (kshift<window[0]) and (lshift<window[0]):
            continue
        
        weight[inth,intk,intl] += 1
        volume[inth,intk,intl] += image[t]

    return volume, weight


def mapImage2Resolution(image=None, size=None, waveLength=None, detectorDistance=None, detectorCenter=None, pixelSize=None, format="res"):
    if image is not None:
        size = image.shape

    xyz = mapPixel2RealXYZ(size=size, center=detectorCenter, pixelSize=pixelSize, detectorDistance=detectorDistance)
    rep = mapRealXYZ2Reciprocal(xyz=xyz, waveLength=waveLength)
    repNorm = np.sqrt(np.sum(rep**2, axis=2))

    if format == "res":
        res = np.zeros(size)
        index = np.where(repNorm>0)
        res[index] = 1./repNorm[index]
        res[repNorm==0] = np.amax(res)
        return res
    else:
        return repNorm
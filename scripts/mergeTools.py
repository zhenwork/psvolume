import numpy as np 
from numba import jit
import mathTools


def mapPixel2RealXYZ(size=None, center=None, pixelSize=None, detectorDistance=None):
    """
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
    norm = np.sqrt(np.sum(xyz**2, axis=2, keepdims=True))
    ## scaled array doesn't have units
    scaled = xyz/norm
    scaled[:,:,2] -= 1.0
    reciprocal = scaled / waveLength
    return reciprocal


def mapReciprocal2Voxel(Amat=None, Bmat=None, returnFormat="HKL", \
                    reciprocal=None, voxelSize=1.0, Phi=0.):
    """
    Return: voxel (N*N*3) 
    voxelSize: 0.015 for 'cartesian' coordinate; 1.0 for "hkl" coordinate
    """
    Phimat = mathTools.quaternion2rotation(mathTools.phi2quaternion(Phi))
    if returnFormat.lower() == "hkl":
        voxel = reciprocal.dot(Phimat.dot(Amat)) / voxelSize 
    elif returnFormat.lower() == "cartesian":
        voxel = reciprocal.dot(Phimat.dot(Amat)).dot(Bmat.T) / voxelSize 
    return voxel


def mapImage2Voxel(image=None, size=None, Amat=None, Bmat=None, xvector=None, Phi=0., \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                returnFormat=None, voxelSize=1.0):

    if image is not None:
        size = image.shape

    if xvector is None:
        xyz = mapPixel2RealXYZ(size=size, center=center, pixelSize=pixelSize, \
                                detectorDistance=detectorDistance)
        reciprocal = mapRealXYZ2Reciprocal(xyz=xyz, waveLength=waveLength)
    else:
        reciprocal = xvector

    voxel = mapReciprocal2Voxel(Amat=None, Bmat=None, returnFormat="HKL", \
                            reciprocal=None, voxelSize=1.0, Phi=0.)

    return voxel


@jit
def Image2Volume(volume, weight, Amat=None, Bmat=None, _image=None, _mask=None, \
                KeepPeak=False, returnFormat="HKL", xvector=None, \
                waveLength=None, pixelSize=None, center=None, detectorDistance=None, \
                Vsample=1, Vcenter=60, Vsize=121, voxelSize=1., Phi=0.):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    """
    voxel = mapImage2Voxel(image=_image, size=None, Amat=Amat, Bmat=Bmat, xvector=xvector, \
            Phi=Phi, waveLength=waveLength, pixelSize=pixelSize, center=center, \
            detectorDistance=detectorDistance, returnFormat=returnFormat, voxelSize=voxelSize)

    ## For Loop to map one image
    Npixels = np.prod(_image.shape)
    image = _image.ravel()
    mask  =  _mask.ravel()
    voxel =  voxel.reshape((Npixels, 3)) 
    
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
        
        hshift = abs(h/Vsample-round(h/Vsample))
        kshift = abs(k/Vsample-round(k/Vsample))
        lshift = abs(l/Vsample-round(l/Vsample))

        if (hshift<0.25) and (kshift<0.25) and (lshift<0.25) and not KeepPeak:
            continue
        
        weight[inth,intk,intl] += 1
        volume[inth,intk,intl] += image[t]

    return volume, weight



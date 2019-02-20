"""
Angstrom (A): wavelength, crystal lattice
Meter (m): detectorDistance, 
"""

import numpy as np 
import json
import crystalTools
import fileManager
import imageTools

def xdsIndex2psvm(fileGXPARMS):
    """
    The unit of lattice constant, invAmat is A
    The unit of angles is degree
    The unit of pixelSize is meter
    The unit of detectorDistance is meter
    The unit of waveLength is A
    """
    psvmParms = {}

    f = open(fileGXPARMS)
    content = f.readlines()
    f.close()

    psvmParms['pixelSize'] = float(content[7].split()[3])
    psvmParms['polarization'] = -1.0
    psvmParms['waveLength'] = float(content[2].split()[0])
    psvmParms['angleStep'] = float(content[1].split()[2])
    psvmParms['detectorDistance'] = float(content[8].split()[2])
    psvmParms['detectorCenter'] = (float(content[8].split()[1]), float(content[8].split()[0]))

    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat = np.zeros((3,3))
    for i in range(4,7):
        for j in range(3):
            invAmat[i-4,j] = float(content[i].split()[j])
    Amat = np.linalg.inv(invAmat)

    ## calculate B matrix
    ## Bmat is the user's defined coordinates
    (a,b,c,alpha,beta,gamma) = [float(each) for each in content[3].split()[1:]]
    latticeConstant = np.array([a,b,c,alpha,beta,gamma])
    Bmat = crystalTools.psvmStandardBmat(latticceConstant=(a,b,c,alpha,beta,gamma))
    psvmParms["latticeConstant"] = latticeConstant
    psvmParms["Amat"] = Amat
    psvmParms["Bmat"] = Bmat
    return psvmParms


def cbf2psvm(fileName):
    cbfhandler = fileManager.CBFManager()
    image, header = cbfhandler.getDataHeader(fileName)

    psvmParms = {}
    psvmParms["startAngle"] = header['start_angle'] 
    psvmParms["currentAngle"] = header['phi']
    psvmParms["Phi"] = header['phi'] - header['start_angle'] 
    psvmParms["angleStep"] = header['angle_increment']
    psvmParms["exposureTime"] = header['exposure_time']
    psvmParms["waveLength"] = header['wavelength']
    psvmParms["pixelSizeX"] = header['x_pixel_size']
    psvmParms["pixelSizeY"] = header['y_pixel_size']
    nx = header['pixels_in_x']
    ny = header['pixels_in_y']

    ## Detector is flipped
    if not image.shape == (ny, nx):
        image=image.T
        if not image.shape == (ny, nx):
            raise Exception("!! Image shape doesn't fit")

    psvmParms["image"] = image
    cbfhandler = None
    return psvmParms


def DataExtraction(fileName, format="cbf"):
    if format.lower()=="cbf":
        psvmParms = cbf2psvm(fileName)
        return psvmParms
    elif format.lower()=="numpy":
        return psvmParms
    elif format.lower()=="h5py":
        return psvmParms
    else:
        raise Exception("!! file format not supported")


def ichSpecificParams():
    #mT = imageTools.MaskTools()
    #mask = mT.valueLimitMask(image, vmin=0.001, vmax=100000)
    #mask = mT.circleMask(size, rmin=40, rmax=None, center=None)
    psvmParms = {}
    mask = np.ones((2527,2463)).astype(int)
    mask[1255:1300,1235:2463] = 0
    mask[1255:1305,1735:2000] = 0
    mask[1255:1310,2000:2463] = 0
    psvmParms["mask"] = mask
    return psvmParms
"""
Angstrom (A): wavelength, crystal lattice
"""

import numpy as np  
import crystalTools
import fileManager
import imageTools
PsvolumeManager = fileManager.PsvolumeManager()

def xds2psvm(fileGXPARMS):
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
    psvmParms['detectorCenter'] = (float(content[8].split()[0]), float(content[8].split()[1]))

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
    Bmat = crystalTools.standardBmat(latticceConstant=(a,b,c,alpha,beta,gamma))
    
    psvmParms["latticeConstant"] = latticeConstant
    psvmParms["Amat"] = Amat
    psvmParms["Bmat"] = Bmat
    
    return psvmParms


def cbf2psvm(fileName):
    cbfhandler = fileManager.CBFManager()
    image, header = cbfhandler.getDataHeader(fileName)

    number = float(fileName.split("/")[-1].split(".")[0].split("_")[-1])

    psvmParms = {}
    psvmParms["startAngle"] = header['phi']
    psvmParms["currentAngle"] = header['start_angle']  
    psvmParms["angleStep"] = header['angle_increment']
    psvmParms["phi"] = psvmParms["angleStep"] * (number-1.)  # psvmParms["currentAngle"] - psvmParms["startAngle"]
    psvmParms["exposureTime"] = header['exposure_time']
    psvmParms["waveLength"] = header['wavelength']
    psvmParms["pixelSizeX"] = header['x_pixel_size']
    psvmParms["pixelSizeY"] = header['y_pixel_size']
    nx = header['pixels_in_x']
    ny = header['pixels_in_y']

    ## Detector is flipped
    if not image.shape == (nx, ny):
        #print "## flip image x/y"
        image=image.T
        if not image.shape == (nx, ny):
            raise Exception("!! Image shape doesn't fit")

    psvmParms["image"] = image
    cbfhandler = None
    return psvmParms

def h5py2psvm(fileName):
    psvmParams = PsvolumeManager.h5py2psvm(fileName)
    return psvmParams


def loadfile(filename, fileType=None):
    if filename is None:
        print "!! Only Support .xds, .cbf, .h5, .npy"
        return {}
    if fileType is None:
        fileType = filename.split(".")[-1]

        
    if fileType.lower() == "xds":
        return xds2psvm(filename)
    elif fileType.lower() == "cbf":
        return cbf2psvm(filename)
    elif fileType.lower() in ["h5", "h5py"]:
        return h5py2psvm(filename)
    elif fileType.lower() in ["numpy", "npy"]:
        return numpy2psvm(filename)
    else:
        print "!! Only Support .xds, .cbf, .h5, .npy"
        return {}
    
    
def specialparams(notation="wtich"):
    #mT = imageTools.MaskTools()
    #mask = mT.valueLimitMask(image, vmin=0.001, vmax=100000)
    #mask = mT.circleMask(size, rmin=40, rmax=None, center=None)
    if notation == "wtich":
        psvmParms = {}
        mask = np.ones((2527,2463)).astype(int)
        mask[1255:1300+2,1235:2463] = 0
        mask[1255:1305+2,1735:2000] = 0
        mask[1255:1310+2,2000:2463] = 0
        psvmParms["mask"] = mask.T
        psvmParms["firMask"] = mask.T
        return psvmParms
    elif notation == "old":
        psvmParms = {}
        mask = np.ones((2527,2463)).astype(int)
        mask[1255:1300,1235:2463] = 0
        mask[1255:1305,1735:2000] = 0
        mask[1255:1310,2000:2463] = 0
        psvmParms["mask"] = mask.T
        psvmParms["firMask"] = mask.T
        return psvmParms
    else:
        print "!! No Special Params"
        return {}

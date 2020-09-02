"""
Angstrom (A): wavelength, crystal lattice
"""
import json
import numpy as np 
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.crystal
import scripts.fsystem
import scripts.image


# "../DIALS/G150T-2/dials.report.html"
def dials_report(fname):
    import codecs
    f=codecs.open(fname, 'r')
    content = f.readlines()

    counter = 0
    for line in content:
        if "smoothly-varying scaling term" in line:
            counter += 1
            if counter == 2:
                break
    content = None
    startx = line.index('"x": [')
    stopx = line.index('], "xaxis"')

    xvalue = []
    for data in line[startx+6:stopx].split(","):
        xvalue.append(float(data))

    starty = line.index('"y": [')
    stopy = line.index('], "yaxis"')

    yvalue = []
    for data in line[starty+6:stopy].split(","):
        yvalue.append(float(data))    
    
    return xvalue,yvalue


def getscale(x,y,theta):
    if x[0] > theta:
        return y[0]
    if x[-1] < theta:
        return y[-1]
    a = np.where(np.array(x)-theta>0)[0][0] - 1
    b = a+1 
    return (y[b] - y[a]) / (x[b] - x[a]) * (theta - x[a]) + y[a]



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
    image, header = scripts.fsystem.CBFmanager.reader(fileName)

    psvmParms = {}
    psvmParms["startAngle"] = header['phi']
    psvmParms["currentAngle"] = header['start_angle']   
    psvmParms["angleStep"] = header['angle_increment'] 
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
    return psvmParms


def expt2psvm(fexpt):
    """
    The unit of lattice constant, invAmat is A
    The unit of angles is degree
    The unit of pixelSize is meter
    The unit of detectorDistance is meter
    The unit of waveLength is A
    """
    def length(arr):
        return np.linalg.norm(arr)
    def angle(x,y):
        return np.arccos(x.dot(y)/length(x)/length(y)) * 180. / np.pi

    psvmParms = {}

    data = json.load(open(fexpt,"r"))

    psvmParms['pixelSize']    = data["detector"][0]["panels"][0]["pixel_size"][0]
    psvmParms['polarization'] = -1.0
    psvmParms['waveLength']   = data["beam"][0]["wavelength"]
    psvmParms['angleStep']    = data["scan"][0]["oscillation"][1]
    psvmParms['detectorDistance'] = - float(data["detector"][0]["panels"][0]["origin"][2])
    psvmParms['detectorCenter']   =  (-float(data["detector"][0]["panels"][0]["origin"][0])/psvmParms['pixelSize'], float(data["detector"][0]["panels"][0]["origin"][1])/psvmParms['pixelSize'])


    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat = np.array([data["crystal"][0]["real_space_a"],data["crystal"][0]["real_space_b"],data["crystal"][0]["real_space_c"]])
    Amat = np.linalg.inv(invAmat)

    ## calculate B matrix
    ## Bmat is the user's defined coordinates
    (a,b,c,alpha,beta,gamma) = [length(invAmat[0]), length(invAmat[1]), length(invAmat[2]), angle(invAmat[1], invAmat[2]), angle(invAmat[0], invAmat[2]), angle(invAmat[0], invAmat[1])]
    latticeConstant = np.array([a,b,c,alpha,beta,gamma])
    Bmat = crystalTools.standardBmat(latticceConstant=(a,b,c,alpha,beta,gamma))
    
    psvmParms["latticeConstant"] = latticeConstant
    psvmParms["Amat"] = Amat
    psvmParms["Bmat"] = Bmat
    
    return psvmParms


def h5py2psvm(fileName):
    psvmParams = scripts.fsystem.PVmanager.reader(fileName)
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
    elif fileType.lower() in ["expt"]:
        return expt2psvm(filename)
    else:
        print "!! Only Support .xds, .cbf, .h5, .npy"
        return {}

    
def special_params(notation="wtich"):
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


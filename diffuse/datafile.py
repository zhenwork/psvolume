import json
import os,sys
import numpy as np
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
import diffuse.image as imagetbx
import diffuse.crystal as crystaltbx
import core.filesystem as fsystemtbx


def dials_report_to_psvm(fname):
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
    
    return {"dials_scaling":(np.array(xvalue), np.array(yvalue))}


def getscale(x,y,theta):
    if x[0] > theta:
        return y[0]
    if x[-1] < theta:
        return y[-1]
    a = np.where(np.array(x)-theta>0)[0][0] - 1
    b = a+1 
    return (y[b] - y[a]) / (x[b] - x[a]) * (theta - x[a]) + y[a]


def gxparms_to_psvm(fileGXPARMS):
    """
    The unit of lattice constant, invAmat is A
    The unit of angles is degree
    The unit of pixelSize is mmeter
    The unit of detectorDistance is mmeter
    The unit of waveLength is A
    """
    psvmParms = {}
    with open(fileGXPARMS) as f:
        content = f.readlines()

    psvmParms['pixel_size_mm'] = float(content[7].split()[3])
    psvmParms['polarization_fr'] = -1.0
    psvmParms['wavelength_A'] = float(content[2].split()[0])
    psvmParms['angle_step_deg'] = float(content[1].split()[2])
    psvmParms['detector_distance_mm'] = float(content[8].split()[2])
    psvmParms['detector_center_px'] = (float(content[8].split()[0]), float(content[8].split()[1]))
    (a,b,c,alpha,beta,gamma) = [float(each) for each in content[3].split()[1:]]
    psvmParms["lattice_constant_A_deg"] = np.array([a,b,c,alpha,beta,gamma]) 

    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat_A = np.zeros((3,3))
    for i in range(4,7):
        for j in range(3):
            invAmat_A[i-4,j] = float(content[i].split()[j])
    Amat_invA = np.linalg.inv(invAmat_A)
    psvmParms["Amat_invA"] = Amat_invA
    content = None
    return psvmParms


def cbf_to_psvm(fileName):
    image, header = fsystemtbx.CBFmanager.reader(fileName)
    psvmParms = {}
    psvmParms["phi_angle_deg"] = header['phi']
    psvmParms["start_angle_deg"] = header['start_angle']  
    psvmParms["angle_step_deg"] = header['angle_increment']
    psvmParms["rotate_angle_deg"] = psvmParms["start_angle_deg"] - psvmParms["phi_angle_deg"]
    psvmParms["exposure_time_s"] = header['exposure_time']
    psvmParms["wavelength_A"] = header['wavelength']
    psvmParms["pixel_size_mm"] = header['x_pixel_size']
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


def expt_to_psvm(fexpt):
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

    psvmParms['pixel_size_mm'] = data["detector"][0]["panels"][0]["pixel_size"][0]
    psvmParms['polarization_fr'] = -1.0
    psvmParms['wavelength_A']   = data["beam"][0]["wavelength"]
    psvmParms['angle_step_deg']    = data["scan"][0]["oscillation"][1]
    psvmParms['detector_distance_mm'] = - float(data["detector"][0]["panels"][0]["origin"][2])
    psvmParms['detector_center_px']   =(- float(data["detector"][0]["panels"][0]["origin"][0])/psvmParms['pixelSize'], float(data["detector"][0]["panels"][0]["origin"][1])/psvmParms['pixelSize'])

    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat_A = np.array([data["crystal"][0]["real_space_a"],data["crystal"][0]["real_space_b"],data["crystal"][0]["real_space_c"]])
    Amat_invA = np.linalg.inv(invAmat_A)

    ## calculate B matrix
    ## Bmat is the user's defined coordinates
    (a,b,c,alpha,beta,gamma) = [length(invAmat_A[0]), length(invAmat_A[1]), length(invAmat_A[2]), \
                    angle(invAmat_A[1], invAmat_A[2]), angle(invAmat_A[0], invAmat_A[2]), angle(invAmat_A[0], invAmat_A[1])]
    psvmParms["lattice_constant_A_deg"] = np.array([a,b,c,alpha,beta,gamma]) 
    psvmParms["Amat_invA"] = Amat_invA
    return psvmParms


def h5py_to_psvm(fileName):
    return fsystemtbx.PVmanager.reader(fileName)


def loadfile(filename, fileType=None):
    if filename is None:
        print "!! Only Support .xds, .cbf, .h5, .npy"
        return {}
    if fileType is None:
        fileType = fsystemtbx.Fsystem.filetype(filename)
    if not isinstance(fileType, str):
        return {}

    if fileType.lower() == "xds":
        return gxparms_to_psvm(filename)
    elif fileType.lower() == "cbf":
        return cbf_to_psvm(filename)
    elif fileType.lower() in ["h5", "h5py"]:
        return h5py_to_psvm(filename)
    elif fileType.lower() in ["numpy", "npy"]:
        return np.load(filename)
    elif fileType.lower() in ["expt"]:
        return expt_to_psvm(filename)
    else:
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
import json
import os,sys
import numpy as np
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import diffuse.crystal
import core.fsystem


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
    
    return {"dials_scaling_angle_deg":np.array(xvalue), "dials_scaling_multiply_scaler":np.array(yvalue)}


def get_scale(angle_deg,scaler,theta):
    x = angle_deg
    y = scaler
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


def cbf_to_psvm(file_name):
    image, header = core.fsystem.CBFmanager.reader(file_name)
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


def dials_expt_to_psvm(fexpt):
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


def h5py_to_psvm(file_name):
    return core.fsystem.PVmanager.reader(file_name)

def numpy_to_psvm(file_name,key="data"):
    return {key:np.load(file_name)}


def load_file_single(file_name, file_type=None, keep_keys=None, reject_keys=None):
    if file_name is None:
        return {}
    if file_type is None:
        file_type = core.fsystem.Fsystem.filetype(file_name)
    if not isinstance(file_type, str):
        return {}

    if file_type.lower() in ["xds","gxparms"]:
        data = gxparms_to_psvm(file_name)
    elif file_type.lower() in ["cbf"]:
        data = cbf_to_psvm(file_name)
    elif file_type.lower() in ["h5", "h5py"]:
        data = h5py_to_psvm(file_name)
    elif file_type.lower() in ["numpy", "npy"]:
        data = numpy_to_psvm(file_name)
    elif file_type.lower() in ["dials_expt","expt"]:
        data = dials_expt_to_psvm(file_name)
    elif file_type.lower() in ["dials_report","report"]:
        data = dials_report_to_psvm(file_name)
    else:
        return {}

    if reject_keys is not None:
        for reject_key in reject_keys:
            if reject_key in data:
                data.pop(reject_key)
    if keep_keys is not None:
        for key in data:
            if key not in keep_keys:
                data.pop(key)
    return data

def file_keys_single(file_name, file_type=None):
    if file_name is None:
        return []
    if file_type is None:
        file_type = core.fsystem.Fsystem.filetype(file_name)
    if not isinstance(file_type, str):
        return []

    if file_type.lower() in ["xds","gxparms"]:
        return []
    elif file_type.lower() in ["cbf"]:
        return []
    elif file_type.lower() in ["h5", "h5py"]:
        return core.fsystem.H5manager.dnames(file_name)
    elif file_type.lower() in ["numpy", "npy"]:
        return ["npy_data"]
    elif file_type.lower() in ["dials_expt","expt"]:
        return []
    elif file_type.lower() in ["dials_report","report"]:
        return []
    else:
        return []

class Fmanager:
    @staticmethod
    def load_file(file_name, file_type=None, keep_keys=None, reject_keys=None):
        data = {}
        if "*" in file_name:
            file_name_list = core.fsystem.Fsystem.file_with_pattern(path=None, pattern=file_name)
        else:
            file_name_list = [file_name]

        for idx, file_name_single in enumerate(file_name_list):
            if idx == 0:
                data = load_file_single(file_name_single, file_type, keep_keys, reject_keys)
                continue
            data = dict_append(data, load_file_single(file_name_single, file_type, keep_keys, reject_keys))
        return data
    @staticmethod
    def file_keys(file_name, file_type=None):
        file_name_single = file_name
        if "*" in file_name:
            file_name_pattern = core.fsystem.Fsystem.file_with_pattern(path=None, pattern=file_name)
            if file_name_pattern in [None,[]]:
                return []
            file_name_single = file_name_pattern[0]
        return file_keys_single(file_name_single,file_type=file_type)

    
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
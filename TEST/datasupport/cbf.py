import cbf
import numpy as np

def getData(fname):
    """
    Data is float numpy array 
    """
    content = cbf.read(fname)
    data = np.array(content.data).astype(float)
    return data


def getHeader(fname):
    """
    Header is python dict {}
    """
    content = cbf.read(fname, metadata=True, parse_miniheader=True)
    header = content.miniheader
    return header


def getDataHeader(fname):
    """
    Data is float numpy array 
    Header is python dict {}
    """
    content = cbf.read(fname, metadata=True, parse_miniheader=True)
    data = np.array(content.data).astype(float)
    header = content.miniheader
    return (data, header)


def load(fname):
    image, header = getDataHeader(fname)
    #number = float(fname.split("/")[-1].split(".")[0].split("_")[-1])
    psvmParms = {}
    psvmParms["start_angle_deg"] = header['phi']
    psvmParms["end_angle_deg"] = header['start_angle']
    psvmParms["angle_increment_deg"] = header['angle_increment']
    psvmParms["phi_deg"] = psvmParms["end_angle_deg"] - psvmParms["start_angle_deg"]
    psvmParms["exposure_time_sec"] = header['exposure_time']
    psvmParms["wavelength_A"] = header['wavelength']
    psvmParms["x_pixel_size_mm"] = header['x_pixel_size'] * 1.0e3
    psvmParms["y_pixel_size_mm"] = header['y_pixel_size'] * 1.0e3
    psvmParms["detector_distance_cm"] = header['detector_distance'] * 1.0e2

    nx = header['pixels_in_x']
    ny = header['pixels_in_y']

    ## Detector is flipped
    if not image.shape == (nx, ny):
        print "## flip x/y of %s" % fname
        image=image.T
        if not image.shape == (nx, ny):
            raise Exception("!! Image shape doesn't fit")
    psvmParms["image"] = image 
    return psvmParms


def save(fname):
    raise Exception, "# Saving as cbf is not supported"


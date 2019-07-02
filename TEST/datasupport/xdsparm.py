import numpy as np
import crystal.tools as crystalTools

def load(fname):
    """
    The unit of lattice constant, invAmat is A
    The unit of angles is degree
    The unit of pixelSize is meter
    The unit of detectorDistance is meter
    The unit of waveLength is A
    """
    psvmParms = {}
    with open(fname, "r") as f:
        content = f.readlines()

    psvmParms['x_pixel_size_mm'] = float(content[7].split()[3])
    psvmParms['y_pixel_size_mm'] = float(content[7].split()[4])
    psvmParms['polarization'] = -1.0
    psvmParms['wavelength_A'] = float(content[2].split()[0])
    psvmParms['angle_increment_deg'] = float(content[1].split()[2])
    psvmParms['detector_distance_cm'] = float(content[8].split()[2])/10.0
    psvmParms['beam_x_pix'] = float(content[8].split()[0])
    psvmParms['beam_y_pix'] = float(content[8].split()[1])

    ## calculate the Amat matrix
    ## GXPARMS saves real space vecA, vecB, vecC, which is invAmat
    invAmat_nm = np.zeros((3,3))
    for i in range(4,7):
        for j in range(3):
            invAmat[i-4,j] = float(content[i].split()[j])
    Amat_invnm = np.linalg.inv(invAmat_nm)

    ## calculate B matrix
    ## Bmat is the user's defined coordinates
    lattice_constant_nm_deg = np.array([float(each) for each in content[3].split()[1:]])
    Bmat_invnm = crystalTools.defaultBmat(lattice_constant_nm_deg=lattice_constant_nm_deg)
    
    psvmParms["lattice_constant_nm_deg"] = lattice_constant_nm_deg
    psvmParms["Amat_invnm"] = Amat_invnm
    psvmParms["Bmat_invnm"] = Bmat_invnm
    
    return psvmParms


def save(fname):
    raise Exception, "Not implemented"
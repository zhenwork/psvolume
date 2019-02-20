"""
All unit should be Angstrom (A): wavelength, crystal lattice 
"""

import numpy as np

## convert lattice constants (a,b,c,al,be,ga) into real-space and reciprocal-space vectors
def Lattice2vector(a,b,c,ag1,ag2,ag3):
    """
    convert lattice constants into x,y,z vectors and h,k,l reciprocal vectors.
    alpha, beta, gamma are in angle (0-180)
    a,b,c are in A
    """
    alpha = ag1/180.*np.pi;
    beta  = ag2/180.*np.pi;
    gamma = ag3/180.*np.pi;
    vecx = a*np.array([1., 0., 0.])
    vecy = b*np.array([np.cos(gamma), np.sin(gamma), 0])
    vecz = c*np.array([np.cos(beta), (np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma), np.sqrt(1.+2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)/np.sin(gamma)])
    recH = np.cross(vecy, vecz)/vecx.dot(np.cross(vecy, vecz))
    recK = np.cross(vecz, vecx)/vecy.dot(np.cross(vecz, vecx))
    recL = np.cross(vecx, vecy)/vecz.dot(np.cross(vecx, vecy))

    return (vecx, vecy, vecz, recH, recK, recL)


## load the indexing infomation from xds outputs
def load_GXPARM_XDS(xds_file):

    f = open(xds_file)
    content = f.readlines()
    f.close()

    Geo = {}
    Geo['pixelSize'] = float(content[7].split()[3])
    Geo['detDistance'] = float(content[8].split()[2])
    Geo['polarization'] = -1.0
    Geo['wavelength'] = float(content[2].split()[0])
    Geo['shape']  = (int(content[7].split()[1]),   int(content[7].split()[2]))
    Geo['center'] = (float(content[8].split()[0]), float(content[8].split()[1]))
    Geo['increment'] = float(content[1].split()[2])

    ## calculate the invAmat matrix
    invAmat = np.zeros((3,3))
    for i in range(4,7):
        for j in range(3):
            invAmat[i-4,j] = float(content[i].split()[j])

    ## calculate B matrix from lattice constants
    (a,b,c,alpha,beta,gamma) = [float(each) for each in content[3].split()[1:]]
    lattice = np.array((a,b,c,alpha,beta,gamma))

    if abs(alpha-90.)<0.5: alpha = 90.
    if abs(beta -90.)<0.5: beta=90.
    if abs(gamma-90.)<0.5: gamma=90.

    (vecx, vecy, vecz, recH, recK, recL) = Lattice2vector(a,b,c,alpha,beta,gamma)
    Bmat = np.array([recH, recK, recL]).T 
    invBmat = np.linalg.inv(Bmat)

    return [Geo, Bmat, invBmat, invAmat, lattice]


## load diffraction patterns
def load_image(filename):
    ## load cbf file
    if filename.endswith('.cbf'):
        content = cbf.read(filename)
        image = np.array(content.data).astype(float)
        ## Sometimes the image is transposed in the cbf file
        ## It affects the polarization correction
        image = image.T
        return image

    ## load .img file
    if filename.endswith('.img'):
        f = open(filename,"rb")
        raw = f.read()
        h = raw[0:511]
        d = raw[512:]
        f.close()
        flat_image = np.frombuffer(d,dtype=np.int16)
        image = np.reshape(flat_image, ((1024,1024)) ).astype(float)

        ## This transpose operation is compatible with x_vectors
        image = image.T
        return image

    ## load numpy arrays
    if filename.endswith(".npy"):
        return np.load(filename)

    ## other formats are not supported
    raise Exception("## This file format is not supported")
    return None


# How to define a users mask
def get_users_mask(Geo, imageType="PILATUS"):

    mask = np.ones(Geo["shape"]).astype(int)

    ## CCD detector
    if imageType.upper()=="CCD":

        ## This is for SNC data
        radius = make_radius(Geo["shape"], center=Geo['center'])
        index = np.where(radius<10)
        mask[index] = 0
        mask[506:557, 471:517] = 0
        return mask

    ## PILATUS detector
    if imageType.upper()=="PILATUS":

        ## This is for ICH data
        # mask[1260:1300,1235:2463] = 0
        mask[1235:2463, 1255:1300] = 0
        mask[1735:2000, 1255:1305] = 0
        mask[2000:2463, 1255:1310] = 0

        radius = make_radius(Geo["shape"], center=Geo['center'])
        index = np.where(radius<40)
        mask[index] = 0
        return mask


def angle2quaternion(angle, axis='y'):
    ## angle is original angle without making it half
    ## angle = idx*increment*np.pi/180.
    if axis.upper()=='X' or axis.upper()=='A':
        return (np.cos(angle/2.), np.sin(angle/2.), 0., 0.)
    elif axis.upper()=="Y" or axis.upper()=='B':
        return (np.cos(angle/2.), 0., np.sin(angle/2.), 0.)
    elif axis.upper()=="Z" or axis.upper()=='C':
        return (np.cos(angle/2.), 0., 0., np.sin(angle/2.))
    else:
        return None


def get_tmpMask(image, vmin=0.0, vmax=1.0e7):
    ## mask is int number
    tmpMask = np.ones(image.shape).astype(int)
    tmpMask[np.where(image<vmin)] = 0
    tmpMask[np.where(image>vmax)] = 0
    return tmpMask


def save_image(fsave, image, Geo):
    zf.h5writer(fsave, 'readout', 'image')
    zf.h5modify(fsave, 'image', image)
    zf.h5modify(fsave, 'center', Geo['center'])
    zf.h5modify(fsave, 'exp', False)
    zf.h5modify(fsave, 'run', False)
    zf.h5modify(fsave, 'event', False)
    zf.h5modify(fsave, 'waveLength', Geo['wavelength'])
    zf.h5modify(fsave, 'detDistance', Geo['detDistance'])
    zf.h5modify(fsave, 'pixelSize', Geo['pixelSize'])
    zf.h5modify(fsave, 'polarization', Geo['polarization'])
    zf.h5modify(fsave, 'rot', 'matrix')



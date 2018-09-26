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
    Geo['polarization'] = 1.0 #float(content[1].split()[3])
    Geo['wavelength'] = float(content[2].split()[0])
    Geo['center'] = (float(content[8].split()[1]), float(content[8].split()[0]))
    Geo['Angle_increment'] = float(content[1].split()[2])

    ## calculate the invAmat matrix
    invAmat = np.zeros((3,3));
    for i in range(4,7):
        for j in range(3):
            invAmat[i-4,j] = float(content[i].split()[j])
    # if invAmat is not None:
    #     invAmat[1,:] = -invAmat[1,:].copy()
    #     tmp = invAmat[:,0].copy()
    #     invAmat[:,0] = invAmat[:,1].copy()
    #     invAmat[:,1] = tmp.copy()

    ## calculate B matrix from lattice constants
    (a,b,c,alpha,beta,gamma) = [float(each) for each in content[3].split()[1:]]

    if abs(alpha-90.)<0.5: alpha = 90.
    if abs(beta -90.)<0.5: beta=90.
    if abs(gamma-90.)<0.5: gamma=90.

    (vecx, vecy, vecz, recH, recK, recL) = Lattice2vector(a,b,c,alpha,beta,gamma)
    Bmat = np.array([recH, recK, recL]).T 
    invBmat = np.linalg.inv(Bmat)

    return [Geo, Bmat, invBmat, invAmat]



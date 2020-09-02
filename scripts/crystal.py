import numpy as np 
import scripts.utils 

def lattice2vector(a, b, c, alpha, beta, gamma):
    """
    convert lattice constants into x,y,z vectors and h,k,l reciprocal vectors.
    alpha, beta, gamma are in angle (0-180)
    a,b,c are in A
    """
    alpha = alpha/180.*np.pi
    beta  = beta/180.*np.pi
    gamma = gamma/180.*np.pi
    vecx = a*np.array([1., 0., 0.])
    vecy = b*np.array([np.cos(gamma), np.sin(gamma), 0]);
    vecz = c*np.array([np.cos(beta), (np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma), np.sqrt(1.+2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)/np.sin(gamma)])
    recH = np.cross(vecy, vecz)/vecx.dot(np.cross(vecy, vecz))
    recK = np.cross(vecz, vecx)/vecy.dot(np.cross(vecz, vecx))
    recL = np.cross(vecx, vecy)/vecz.dot(np.cross(vecx, vecy))
    return (vecx, vecy, vecz, recH, recK, recL)


def standardBmat(Amat=None, latticceConstant=None):
    if latticceConstant is not None:
        (a,b,c,alpha,beta,gamma) = latticceConstant
        (vecx, vecy, vecz, recH, recK, recL) = lattice2vector(a,b,c,alpha,beta,gamma)
        standardBmat = np.array([recH, recK, recL]).T
        return standardBmat
    elif Amat is not None:
        invAmat = np.linalg.inv(Amat)
        (vecA,vecB,vecC) = invAmat 
        a = scripts.utils.length(vecA)
        b = scripts.utils.length(vecB)
        c = scripts.utils.length(vecC)
        ## scripts.utils.angle return degree unit
        alpha = scripts.utils.angle(vecB, vecC)
        beta = scripts.utils.angle(vecA, vecC)
        gamma = scripts.utils.angle(vecB, vecA)
        return standardBmat(latticceConstant=(a,b,c,alpha,beta,gamma))
    else:
        raise Exception("!! Can't make standard Bmat")
        return None


import numpy as np 
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
import core.utils

def lattice_to_vector(a, b, c, alpha, beta, gamma):
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


def standard_bmat(Amat_invA=None, latticce_constant_A_deg=None):
    if latticce_constant_A_deg is not None:
        (a,b,c,alpha,beta,gamma) = latticce_constant_A_deg
        (vecx_A, vecy_A, vecz_A, recH_invA, recK_invA, recL_invA) = lattice_to_vector(a,b,c,alpha,beta,gamma)
        standardBmat = np.array([recH_invA, recK_invA, recL_invA]).T
        return standardBmat
    elif Amat_invA is not None:
        invAmat_A = np.linalg.inv(Amat_invA)
        (vecA_A,vecB_A,vecC_A) = invAmat_A 
        a = core.utils.length(vecA_A)
        b = core.utils.length(vecB_A)
        c = core.utils.length(vecC_A)
        ## mathTools.angle return degree unit
        alpha = core.utils.angle(vecB_A, vecC_A)
        beta  = core.utils.angle(vecA_A, vecC_A)
        gamma = core.utils.angle(vecB_A, vecA_A)
        return standard_bmat(latticce_constant_A_deg=(a,b,c,alpha,beta,gamma))
    else:
        print("!! can't make standard Bmat")
        return None
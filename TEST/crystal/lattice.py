import numpy as np 
import utils.math as mathTools

def lattice2vector(la_nm, lb_nm, lc_nm, alpha_deg, beta_deg, gamma_deg):
    """
    convert lattice constants into x,y,z vectors and h,k,l reciprocal vectors.
    alpha, beta, gamma are in angle (0-180)
    a,b,c are in nm
    """
    alpha_rad = alpha_deg/180.*np.pi
    beta_rad  = beta_deg /180.*np.pi
    gamma_rad = gamma_deg/180.*np.pi
    vecx_nm = la_nm*np.array([1., 0., 0.])
    vecy_nm = lb_nm*np.array([np.cos(gamma_rad), np.sin(gamma_rad), 0])
    vecz_nm = lc_nm*np.array([np.cos(beta_rad), \
        (np.cos(alpha_rad)-np.cos(gamma_rad)*np.cos(beta_rad))/np.sin(gamma_rad), \
        np.sqrt(1.+2.*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad)-np.cos(alpha_rad)**2-np.cos(beta_rad)**2-np.cos(gamma_rad)**2)/np.sin(gamma_rad)])
    recH_invnm = np.cross(vecy_nm, vecz_nm)/vecx.dot(np.cross(vecy_nm, vecz_nm))
    recK_invnm = np.cross(vecz_nm, vecx_nm)/vecy.dot(np.cross(vecz_nm, vecx_nm))
    recL_invnm = np.cross(vecx_nm, vecy_nm)/vecz.dot(np.cross(vecx_nm, vecy_nm))
    return (vecx_nm, vecy_nm, vecz_nm, recH_invnm, recK_invnm, recL_invnm)


def defaultBmat(Amat_invnm=None, lattice_constant_nm_deg=None):
    if lattice_constant_nm_deg is not None:
        (la_nm, lb_nm, lc_nm, alpha_deg, beta_deg, gamma_deg) = lattice_constant_nm_deg
        (vecx_nm, vecy_nm, vecz_nm, recH_invnm, recK_invnm, recL_invnm) = lattice2vector(la_nm, lb_nm, lc_nm, alpha_deg, beta_deg, gamma_deg)
        Bmat_invnm = np.array([recH_invnm, recK_invnm, recL_invnm]).T
        return Bmat_invnm
    elif Amat_invnm is not None:
        invAmat_nm = np.linalg.inv(Amat_invnm)
        (vecA_nm, vecB_nm, vecC_nm) = invAmat_nm
        la_nm = mathTools.length(vecA_nm)
        lb_nm = mathTools.length(vecB_nm)
        lc_nm = mathTools.length(vecC_nm)
        ## mathTools.angle return degree unit
        alpha_deg = mathTools.angle(vecB_nm, vecC_nm)
        beta_deg = mathTools.angle(vecA_nm, vecC_nm)
        gamma_deg = mathTools.angle(vecB_nm, vecA_nm)
        return defaultBmat(lattice_constant_nm_deg=(la_nm,lb_nm,lc_nm,alpha_deg,beta_deg,gamma_deg))
    else:
        raise Exception("!! Can't make default Bmat")
        return None


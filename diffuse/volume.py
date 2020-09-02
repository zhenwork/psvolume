import os,sys
import numpy as np 
from numba import jit
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import diffuse.utils as dutils
import diffuse.crystal as crystaltbx


@jit
def volume_symmetrize(volume_3d, volume_mask_3d=None, threshold_keep=(-100,1000), symmetry="P1211"):
    volume = volume_3d.copy()
    if volume_mask_3d is None:
        volumeMask = np.ones(volume_3d.shape)
    else:
        volumeMask = volume_mask_3d.copy()

    if threshold_keep is not None:
        volumeMask[(volume<threshold_keep[0]) | (volume>threshold_keep[1])] = 0.

    volume *= volumeMask
    
    volumeSym = volume.copy()
    weightSym = volumeMask.copy()
    if symmetry in ["P1211","p1211"]:
        volumeSym +=     volume[::-1,::-1,::-1]
        weightSym += volumeMask[::-1,::-1,::-1]

        volumeSym +=     volume[::-1,:,::-1]
        weightSym += volumeMask[::-1,:,::-1]

        volumeSym +=     volume[:,::-1,:]
        weightSym += volumeMask[:,::-1,:]

        index = np.where(weightSym>0.5)
        volumeSym[index] /= weightSym[index]
        return volumeSym, weightSym

    elif symmetry.lower() == "friedel":
        volumeSym +=     volume[::-1,::-1,::-1]
        weightSym += volumeMask[::-1,::-1,::-1]

        index = np.where(weightSym>0.5)
        volumeSym[index] /= weightSym[index]
        return volumeSym, weightSym
    else:
        return None 


@jit
def volume_hkl_reshape(volume,center,astar,bstar,cstar,volnew_size,volnew_center,\
                volume_mask=None,voxel_new_size=1.,threshold_keep=(-100,1000)):
    cx,cy,cz = center
    nx,ny,nz = volume.shape
    
    if volume_mask is not None:
        mask = volume_mask.copy().astype(int)
    else:
        mask = np.ones(volume.shape).astype(int)

    volnew  = np.zeros(volnew_size)
    weight  = np.zeros(volnew_size)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i,j,k] == 0:
                    continue
                if volume[i,j,k] < threshold_keep[0]:
                    continue
                if volume[i,j,k] > threshold_keep[1]:
                    continue

                hkl = float(i-cx)*astar+float(j-cy)*bstar+float(k-cz)*cstar 

                tx = hkl[0] + volnew_center[0]
                ty = hkl[1] + volnew_center[1]
                tz = hkl[2] + volnew_center[2]

                x = int(tx)
                y = int(ty)
                z = int(tz)

                # throw one line more
                if (tx < 0) or x > (nx-1) or (ty < 0) or y > (ny-1) or (tz < 0) or z > (nz-1): 
                    continue

                fx = tx - x
                fy = ty - y
                fz = tz - z
                cx = 1. - fx
                cy = 1. - fy
                cz = 1. - fz

                # Correct for solid angle and polarization
                w = volume[i,j,k]

                # save to the 3D volume
                f = cx*cy*cz 
                weight[x, y, z] += f 
                volnew[x, y, z] += f * w 

                f = cx*cy*fz 
                weight[x, y, z+1] += f 
                volnew[x, y, z+1] += f * w 

                f = cx*fy*cz 
                weight[x, y+1, z] += f 
                volnew[x, y+1, z] += f * w 

                f = cx*fy*fz 
                weight[x, y+1, z+1] += f 
                volnew[x, y+1, z+1] += f * w 

                f = fx*cy*cz 
                weight[x+1, y, z] += f 
                volnew[x+1, y, z] += f * w

                f = fx*cy*fz 
                weight[x+1, y, z+1] += f 
                volnew[x+1, y, z+1] += f * w 

                f = fx*fy*cz 
                weight[x+1, y+1, z] += f
                volnew[x+1, y+1, z] += f * w 

                f = fx*fy*fz 
                weight[x+1, y+1, z+1] += f 
                volnew[x+1, y+1, z+1] += f * w 

    index = np.where(weight>0.5)
    volnew[index] /= weight[index]
    index = np.where(weight<=0.5)
    volnew[index] = -1024
    return volnew, weight


@jit
def volume_radial_profile(volume_3d, volume_center, volume_mask_3d=None, volume_center=None, threshold_keep=(-100,1000),\
                    contribute_window=1, expand_scale=1, basis_by_column=None):
    
    volume = volume_3d.copy()
    if volume_mask_3d is None:
        volumeMask = np.ones(volume.shape).astype(int)
    else:
        volumeMask = volume_mask_3d.copy()

    if threshold_keep is not None:
        volumeMask *= (volume>=threshold_keep[0]) * (volume<=threshold_keep[1])

    volume *= volumeMask
    nx, ny, nz = volume.shape
    x = np.arange(nx) - volume_center[0]
    y = np.arange(ny) - volume_center[1]
    z = np.arange(nz) - volume_center[2]
    xaxis,yaxis,zaxis = np.meshgrid(x,y,z,indexing="ij")

    if basis_by_column is not None:
        vecA = basis_by_column[:,0].reshape((1,3))
        vecB = basis_by_column[:,1].reshape((1,3))
        vecC = basis_by_column[:,2].reshape((1,3))

        xaxis = xaxis.reshape((nx,ny,nz,1)) * vecA
        yaxis = yaxis.reshape((nx,ny,nz,1)) * vecB
        zaxis = zaxis.reshape((nx,ny,nz,1)) * vecC

        sumArray = xaxis + yaxis + zaxis
        radius = np.sqrt(np.sum(sumArray**2, axis=3)) * expand_scale
    else:
        radius = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2) * expand_scale

    radius_assign = np.around(radius).astype(int)   # integers
    max_radius = int(np.amax(radius_assign))
    radius_1d = np.zeros(max_radius+1)
    weight_1d = np.zeros(radius1d.shape)
    half_window = int((contribute_window-1.)/2.)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = radius[i,j,k]
                min_ridx = int(math.ceil( r-half_window))
                max_ridx = int(math.floor(r+half_window))
                for ridx in range(min_ridx, max_ridx+1):
                    if ridx >= 0 and ridx<=max_radius:
                        radius_1d[ridx] += volume[i,j,k]
                        weight_1d[ridx] += volumeMask[i,j,k]

    index = np.where(weight_1d > 0)
    radius_1d[index] /= weight_1d[index]
    _radialBackground = radius_1d[radius_assign]
    return _radialBackground



def datalist_resolution(HKLI_1, lattice_constant_A_deg):
    if len(HKLI_1) < 10 or HKLI_1.shape[1] != 4:
        print "Bad data"
        return None 

    a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg = lattice_constant_A_deg
    (vecx_A, vecy_A, vecz_A, recH_Ainv, recK_Ainv, recL_Ainv) = \
            crystaltbx.lattice_to_vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)

    Amat_Ainv = np.array([recH_Ainv, recK_Ainv, recL_Ainv]).T

    reciprocal_Ainv = HKLI_1[:,:3].dot(Amat_Ainv.T)   ## N*3  3*3

    radius_Ainv = np.sqrt(np.sum(reciprocal_Ainv**2, axis=1))  # N,

    res_high_A = 1.0 / np.amax(radius_Ainv)
    
    if np.amin(radius_Ainv) == 0:
        res_low_A = 1.0e9
    else:    
        res_low_A = 1.0 / np.amin(radius_Ainv)

    return radius_Ainv, res_high_A, res_low_A



def uniform_sampling_ref(lattice_constant_A_deg, res_cut_off_low_Ainv, res_cut_off_high_Ainv, num_res_shell):

    a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg = lattice_constant_A_deg
    (vecx_A, vecy_A, vecz_A, recH_Ainv, recK_Ainv, recL_Ainv) = \
            crystaltbx.lattice_to_vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)
    
    unit_volume_Ainv3 = np.cross(recH_Ainv, recK_Ainv).dot(recL_Ainv)
    tot_V_Ainv3 = res_cut_off_high_Ainv**3 - res_cut_off_low_Ainv**3
    shell_V_Ainv3 = tot_V_Ainv3 * 1.0 / num_res_shell
    linspace_Ainv = [(res_cut_off_low_Ainv**3 + idx * shell_V_Ainv3)**(1./3.) for idx in range(num_res_shell)]
    linspace_Ainv.append(res_cut_off_high_Ainv)
    assert len(linspace_Ainv) == (num_res_shell+1)
    return np.array(linspace_Ainv)


"""
def completeness(HKLI_1, lattice_constant_A_deg=None, res_shell_step_Ainv=None, \
                num_res_shell=20, num_ref_uniform=True, res_cut_off_high_A=0, res_cut_off_low_A=1e9):
    assert lattice_constant_A_deg is not None
    radius_Ainv, res_high_A, res_low_A = resolution(HKLI_1, lattice_constant_A_deg)
    res_cut_off_high_A = max(res_cut_off_high_A, res_high_A)  # high_A = 1.0A
    res_cut_off_low_A = min(res_cut_off_low_A, res_low_A)     # low_A  = 100A
    res_cut_off_high_Ainv = 1.0 / res_cut_off_high_A
    res_cut_off_low_Ainv = 1.0 / res_cut_off_low_A
    if res_cut_off_low_Ainv < 1e-8:
        res_cut_off_low_Ainv = 0.0
    if res_shell_step_Ainv is not None:
        linspace_Ainv = np.arange(res_cut_off_low_Ainv, res_cut_off_high_Ainv, res_shell_step_Ainv)
        if linspace_Ainv[-1] < res_cut_off_high_Ainv:
            linspace_Ainv = np.append(linspace_Ainv, linspace_Ainv[-1] + res_shell_step_Ainv)
    elif num_ref_uniform is False:
        linspace_Ainv = np.linspace(res_cut_off_low_Ainv, res_cut_off_high_Ainv, num_res_shell + 1)
    else:
        ## uniform sampling of number of reflection
        linspace_Ainv = uniform_sampling_ref(lattice_constant_A_deg, res_cut_off_low_Ainv, res_cut_off_high_Ainv, num_res_shell)
    return 
"""

def correlation(HKLI_1, HKLI_2, center, lattice_constant_A_deg=None, res_shell_step_Ainv=None, \
                num_res_shell=20, res_cut_off_high_A=1.5, res_cut_off_low_A=50, num_ref_uniform=True, \
                vmin=-100, vmax=1000):
    ## num_res_shell > res_shell_step_Ainv
    ## res_cut_off_low_A has larger nunber
    ## res_cut_off_high_A has smaller number
    ## only count values inside [vmin, vmax)

    assert lattice_constant_A_deg is not None

    a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg = lattice_constant_A_deg
    (vecx_A, vecy_A, vecz_A, recH_Ainv, recK_Ainv, recL_Ainv) = \
            crystaltbx.lattice_to_vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)

    res_cut_off_high_Ainv = 1.0 / res_cut_off_high_A
    res_cut_off_low_Ainv = 1.0 / res_cut_off_low_A

    if res_shell_step_Ainv is not None:
        linspace_Ainv = np.arange(res_cut_off_low_Ainv, res_cut_off_high_Ainv, res_shell_step_Ainv)
        if linspace_Ainv[-1] < res_cut_off_high_Ainv:
            linspace_Ainv = np.append(linspace_Ainv, linspace_Ainv[-1] + res_shell_step_Ainv)
    elif num_ref_uniform is False:
        linspace_Ainv = np.linspace(res_cut_off_low_Ainv, res_cut_off_high_Ainv, num_res_shell + 1)
        linspace_Ainv[-1] += 1e-5
    else:
        ## uniform sampling of number of reflection
        linspace_Ainv = uniform_sampling_ref(lattice_constant_A_deg, res_cut_off_low_Ainv, res_cut_off_high_Ainv, num_res_shell)
        linspace_Ainv[-1] += 1e-5

    #nx,ny,nz = HKLI_1.shape
    nx,ny,nz = HKLI_1.shape
    cx,cy,cz = center

    h = np.arange(nx)-cx
    k = np.arange(ny)-cy
    l = np.arange(nz)-cz 
    
    H, K, L = np.meshgrid(h,k,l,indexing="ij")
    
    R_Ainv = np.sqrt((H*recH_Ainv[0] + K*recK_Ainv[0] + L*recL_Ainv[0])**2 + \
                     (H*recH_Ainv[1] + K*recK_Ainv[1] + L*recL_Ainv[1])**2 + \
                     (H*recH_Ainv[2] + K*recK_Ainv[2] + L*recL_Ainv[2])**2 )

    real_num_shell = len(linspace_Ainv)-1
    floor_res_Ainv = linspace_Ainv[:real_num_shell]
    upper_res_Ainv = linspace_Ainv[1:]
    center_Ainv = (upper_res_Ainv+floor_res_Ainv)/2.
    number_value = []
    corr = []
    for idx in range(real_num_shell):
        r1 = floor_res_Ainv[idx]
        r2 = upper_res_Ainv[idx]
        index = np.where( (R_Ainv>=r1) & (R_Ainv<r2) & (HKLI_1>=vmin) & (HKLI_1<vmax) & (HKLI_2>=vmin) & (HKLI_2<vmax) )
        number_value.append(len(index[0]))

        if len(index[0]) > 8:
            corr.append(np.corrcoef(HKLI_1[index], HKLI_2[index])[0,1])
        else:
            corr.append(0)
            
    floor_res_Ainv = np.append(floor_res_Ainv, floor_res_Ainv[0])
    upper_res_Ainv = np.append(upper_res_Ainv, upper_res_Ainv[-1])
    
    r1 = floor_res_Ainv[-1]
    r2 = upper_res_Ainv[-1]
    index = np.where( (R_Ainv>=r1) & (R_Ainv<r2) & (HKLI_1>=vmin) & (HKLI_1<vmax) & (HKLI_2>=vmin) & (HKLI_2<vmax) )
    number_value.append(len(index[0]))

    if len(index[0]) > 8:
        corr.append(np.corrcoef(HKLI_1[index], HKLI_2[index])[0,1])
    else:
        corr.append(0)

    return 1./np.array(upper_res_Ainv), 1./np.array(floor_res_Ainv), np.array(corr), np.array(number_value)



def volume_to_txt(volume,volume_center,fsave="tmp.txt", vmask=None, \
                    threshold_keep=(-100,1000), lift=0, headers=None):
    if vmask is None:
        vMask = np.ones(volume.shape).astype(int)
    else:
        vMask = vmask.copy()

    if threshold_keep is not None:
        vMask *= (volume>=threshold_keep[0]) * (volume<=threshold_keep[1])

    lift = max(0,lift)
    (nx, ny, nz) = volume.shape
    (cx, cy, cz) = volume_center

    with open(fsave, "w") as fw:
        if headers is not None:
            for line in headers:
                if line.endswith("\n"):
                    fw.write(line)
                else:
                    fw.write(line+"\n")

        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if vMask[x,y,z]==0:
                        continue

                    h = x-cx
                    k = y-cy
                    l = z-cz 
                    val = round(volume[x,y,z]+lift, 3)
                    string = str(h).rjust(4)+str(k).rjust(4)+str(l).rjust(4)+str(val).rjust(10)+"\n"
                    fw.write(string)


def volume_to_phenix(volume,volume_center,fsave="tmp.txt", vmask=None, \
                    threshold_keep=(-50,1000), lift=10, headers=None):
    if vmask is None:
        vMask = np.ones(volume.shape).astype(int)
    else:
        vMask = vmask.copy()

    if threshold_keep is not None:
        vMask *= (volume>=threshold_keep[0]) * (volume<=threshold_keep[1])

    lift = max(0,lift)
    vmin = np.amin(volume * vMask)
    (nx, ny, nz) = volume.shape
    (cx, cy, cz) = volume_center

    with open(fsave, "w") as fw:
        if headers is not None:
            for line in headers:
                if line.endswith("\n"):
                    fw.write(line)
                else:
                    fw.write(line+"\n")

        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if vMask[x,y,z]==0:
                        continue

                    h = x-cx
                    k = y-cy
                    l = z-cz 
                    val = round(volume[x,y,z]-vmin+lift, 3)
                    sqr = round(np.sqrt(val), 3)

                    string = str(h).rjust(4)+str(k).rjust(4)+str(l).rjust(4)+str(val).rjust(8)+str(sqr).rjust(8)+"\n"

                    fw.write(string)



@jit
def pViewer(_volume, vector, center=None, directions=None, voxelsize=1, \
        vmin=None, vmax=None, rmax=None, rmin=None, depth=1, stretch=1.0, standardCut=True):
    """
    volume is a 3D matrix, whose three directions are x,y,z respectively; 
    vector is the viewing direction like (1,2,1), it doesn't have to be an unit vector; 
    center is the origin of volume, if not set, it will be the center of volume matrix;
    thr is the low/high threshold like thr=(0,10), other values are transparent; 
    depth indicates the viewing transparency.
    columnMatrix: meaning the real directions of x,y,z, for example, three columns are vh, vk, vl. The position of each voxel (h,k,l)
            should be h*vh + k*vk + l*vl
    """
    volume = _volume.copy()
    (nx, ny, nz) = volume.shape
    if center is not None:
        (cx, cy, cz) = center
    else:
        (cx, cy, cz) = np.array(volume.shape)*0.5-0.5

    h = np.arange(nx)-cx*1.0
    k = np.arange(ny)-cy*1.0
    l = np.arange(nz)-cz*1.0
    (haxis, kaxis, laxis) = np.meshgrid(h,k,l,indexing='ij')

    ## convert raw 3D matrix (h,k,l) to real directions vx,vy,vz
    if directions is not None:
        Hreal = directions[:,0]
        Kreal = directions[:,1]
        Lreal = directions[:,2]
    else:
        Hreal = np.array([1., 0, 0])
        Kreal = np.array([0, 1., 0])
        Lreal = np.array([0, 0, 1.])

    xaxis = haxis * Hreal[0] + kaxis * Kreal[0] + laxis * Lreal[0]
    yaxis = haxis * Hreal[1] + kaxis * Kreal[1] + laxis * Lreal[1]
    zaxis = haxis * Hreal[2] + kaxis * Kreal[2] + laxis * Lreal[2]
    radius = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    
    minval = np.amin(volume)
    maxval = np.amax(volume)
    if vmin is None: vmin=minval-1
    if vmax is None: vmax=maxval+1
    
    if rmin is None: rmin=np.amin(radius)-1
    if rmax is None: rmax=np.amax(radius)+1
    volume[(radius<rmin)|(radius>=rmax)] = vmin-1
    volume[(radius<rmin)] = vmax-1                ## pending 
    
    if standardCut is True:
        volume[(xaxis>0)&(zaxis>0)] = vmin-1
        volume[(xaxis<=0)&(yaxis>0)&(zaxis>0)] = vmin-1
        
    xaxis /= voxelsize
    yaxis /= voxelsize
    zaxis /= voxelsize

    ## get projection direction (unit vectors)
    vec = np.array(vector)
    if vec[0]==0 and vec[1]==0:
        px = np.array((1.,0.,0.))
        py = np.array((0.,1.,0.))
        pz = np.array((0.,0.,1.))
    else:
        pz = vec/np.sqrt(vec.dot(vec))
        pzh = np.array((pz[0], pz[1], 0))
        lpzh = np.sqrt(pzh.dot(pzh))
        py = -pz[2]*pzh/lpzh + lpzh*np.array((0.,0.,1.))
        px = np.cross(py, pz)*stretch
        py = py*stretch

    ## monitor the projection axis
    proj = np.zeros((3, 2))
    proj[0,0] = np.array([1.,0,0]).dot(px)
    proj[0,1] = np.array([1.,0,0]).dot(py)
    proj[1,0] = np.array([0,1.,0]).dot(px)
    proj[1,1] = np.array([0,1.,0]).dot(py)
    proj[2,0] = np.array([0,0,1.]).dot(px)
    proj[2,1] = np.array([0,0,1.]).dot(py)


    # project to new x,y,z axis
    volx = xaxis*px[0] + yaxis*px[1] + zaxis*px[2]
    voly = xaxis*py[0] + yaxis*py[1] + zaxis*py[2]
    volz = xaxis*pz[0] + yaxis*pz[1] + zaxis*pz[2]

    volx = np.around(volx).astype(int)
    voly = np.around(voly).astype(int)
    volz = np.around(volz)

    xmin = np.amin(volx)
    xmax = np.amax(volx)
    ymin = np.amin(voly)
    ymax = np.amax(voly)

    detvalue =   np.zeros((xmax-xmin+1, ymax-ymin+1))
    detpolar =   np.zeros((xmax-xmin+1, ymax-ymin+1)) + np.amin(volz) - 1024
    detcount =   np.zeros((xmax-xmin+1, ymax-ymin+1))
    resolution = np.zeros((xmax-xmin+1, ymax-ymin+1))

    cenX = 0 - xmin
    cenY = 0 - ymin

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value = volume[i,j,k]
                if value<vmin or value>=vmax:
                    continue
                dx = volx[i,j,k]-xmin
                dy = voly[i,j,k]-ymin 
                
                if volz[i,j,k]>detpolar[dx,dy]:
                    detpolar[dx,dy] = volz[i,j,k]
                    resolution[dx,dy] = radius[i,j,k]

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value = volume[i,j,k]
                if value<vmin or value>=vmax:
                    continue
                dx = volx[i,j,k]-xmin
                dy = voly[i,j,k]-ymin 
                
                if volz[i,j,k]>=detpolar[dx,dy]-depth:
                    detvalue[dx,dy]  = value+detvalue[dx,dy]*detcount[dx,dy]
                    detvalue[dx,dy] /= detcount[dx,dy]+1
                    detcount[dx,dy] += 1
    index = np.where(detcount==0)
    detvalue[index] = -1024
    return detvalue, detcount, detpolar, resolution, cenX, cenY, proj
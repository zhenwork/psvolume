from numba import jit
import numpy as np 
import scripts.utils
import scripts.crystal

@jit
def volumeSymmetrize_alg1(_volume, _volumeMask = None, _threshold=(-100,1000), symmetry="P1211"):
    volume = _volume.copy()
    weight = np.zeros(volume.shape)
    nx, ny, nz = volume.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                mi = nx-1-i
                mj = ny-1-j
                mk = nz-1-k
                
                if symmetry.lower() == "p1211":
                    pairs = np.array([ volume[i,j,k], volume[mi,mj,mk], volume[mi,j,mk], volume[i,mj,k] ]) #P1211
                elif symmetry.lower() == "friedel":
                    # FIXME: only applicable to friedel symmetry
                    pairs = np.array([ volume[i,j,k], volume[mi,mj,mk] ]) #P1211
                elif symmetry.lower() == "snc":
                    # FIXME: only applicable for snc data
                    pairs = np.array([ volume[i,j,k], volume[mi,mj,k], volume[mj,i,k], volume[j,mi,k], \
                                       volume[mi,mj,mk], volume[i,j,mk], volume[j,mi,mk], volume[mj,i,mk]])
                else:
                    pairs = None
                
                ori = pairs.copy()
                pairs = pairs[np.where(pairs>_threshold[0])].copy()
                pairs = pairs[np.where(pairs<_threshold[1])].copy()
                weight[i,j,k] = len(pairs)

                if len(pairs) == 0: 
                    volume[i,j,k] = np.amin(ori)
                else: 
                    volume[i,j,k] = np.mean(pairs)

    return volume, weight    


@jit
def volumeSymmetrize(_volume, _volumeMask = None, _threshold=(-100,1000), symmetry="P1211"):
    volume = _volume.copy()
    if _volumeMask is None:
        volumeMask = np.ones(volume.shape)
    else:
        volumeMask = _volumeMask.copy()

    if _threshold is not None:
        volumeMask[(volume<_threshold[0]) | (volume>_threshold[1])] = 0.

    volume *= volumeMask
    
    if symmetry.lower() == "p1211":
        volume2 = volume[::-1,::-1,::-1].copy()
        volumeMask2 = volumeMask[::-1,::-1,::-1].copy()
        volume3 = volume[::-1,:,::-1].copy()
        volumeMask3 = volumeMask[::-1,:,::-1].copy()
        volume4 = volume[:,::-1,:].copy()
        volumeMask4 = volumeMask[:,::-1,:].copy()

        volumeSym = (volume + volume2 + volume3 + volume4)
        weightSym = (volumeMask + volumeMask2 + volumeMask3 + volumeMask4)

        index = np.where(weightSym>0.5)
        volumeSym[index] /= weightSym[index]
        return volumeSym, weightSym

    elif symmetry.lower() == "friedel":
        volume2 = volume[::-1,::-1,::-1].copy()
        volumeMask2 = volumeMask[::-1,::-1,::-1].copy()

        volumeSym = (volume + volume2)
        weightSym = (volumeMask + volumeMask2)

        index = np.where(weightSym>0.5)
        volumeSym[index] /= weightSym[index]
        return volumeSym, weightSym

    elif symmetry.lower() == "snc":
        return None
    else:
        return None 


@jit
def hkl2volume(volume, astar, bstar, cstar, _volumeMask = None, ithreshold=(-100,1000)):
    idata = volume.copy()
    num_samp = idata.shape[0]
    icen = (num_samp-1)/2
    center = np.array([icen]*3).astype(float)
    size = num_samp
    model3d = np.zeros((num_samp, num_samp, num_samp))
    weight = np.zeros((num_samp, num_samp, num_samp))

    if _volumeMask is not None:
        volumeMask = _volumeMask.copy()
    else:
        volumeMask = np.ones(idata.shape)

    if ithreshold is not None:
        volumeMask[(idata<ithreshold[0]) | (idata>ithreshold[1])] = 0

    volumeMask = volumeMask.astype(int)

    idata[volumeMask==0] = -1024
    
    for i in range(-icen, icen+1):
        for j in range(-icen, icen+1):
            for k in range(-icen, icen+1):

                if idata[i+icen, j+icen, k+icen] < ithreshold[0]: continue
                if idata[i+icen, j+icen, k+icen] > ithreshold[1]: continue

                hkl = float(i)*astar+float(j)*bstar+float(k)*cstar
                pos = hkl+center

                tx = pos[0]
                ty = pos[1]
                tz = pos[2]

                x = int(tx)
                y = int(ty)
                z = int(tz)

                # throw one line more
                if (tx < 0) or x > (num_samp-1) or (ty < 0) or y > (num_samp-1) or (tz < 0) or z > (num_samp-1): continue

                fx = tx - x
                fy = ty - y
                fz = tz - z
                cx = 1. - fx
                cy = 1. - fy
                cz = 1. - fz

                # Correct for solid angle and polarization
                w = idata[i+icen,j+icen,k+icen]

                # save to the 3D volume
                f = cx*cy*cz 
                weight[x, y, z] += f 
                model3d[x, y, z] += f * w 

                f = cx*cy*fz 
                weight[x, y, ((z+1)%size)] += f 
                model3d[x, y, ((z+1)%size)] += f * w 

                f = cx*fy*cz 
                weight[x, ((y+1)%size), z] += f 
                model3d[x, ((y+1)%size), z] += f * w 

                f = cx*fy*fz 
                weight[x, ((y+1)%size), ((z+1)%size)] += f 
                model3d[x, ((y+1)%size), ((z+1)%size)] += f * w 

                f = fx*cy*cz 
                weight[((x+1)%size), y, z] += f 
                model3d[((x+1)%size), y, z] += f * w

                f = fx*cy*fz 
                weight[((x+1)%size), y, ((z+1)%size)] += f 
                model3d[((x+1)%size), y, ((z+1)%size)] += f * w 

                f = fx*fy*cz 
                weight[((x+1)%size), ((y+1)%size), z] += f
                model3d[((x+1)%size), ((y+1)%size), z] += f * w 

                f = fx*fy*fz 
                weight[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f 
                model3d[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f * w 

    index = np.where(weight>1.e-2)
    model3d[index] /= weight[index]
    index = np.where(weight<=1.e-2)
    model3d[index] = -1024
    return model3d, weight

@jit
def distri(idata, astar, bstar, cstar, ithreshold=(-100,1000), iscale=1, iwindow=5):

    la = scripts.utils.length(astar)
    lb = scripts.utils.length(bstar)
    lc = scripts.utils.length(cstar)
    num_samp = idata.shape[0]
    center = (num_samp - 1.)/2.
    ir = (int(center*np.sqrt(3.)*max(la/lb, lc/lb))+20)*iscale
    #print 'ir='+str(ir)
    distri = np.zeros(ir)
    weight = np.zeros(ir)
    Rmodel = np.zeros((num_samp, num_samp, num_samp)).astype(int)
    
    for i in range(num_samp):
        for j in range(num_samp):
            for k in range(num_samp):
                if idata[i,j,k]<ithreshold[0] or idata[i,j,k]>ithreshold[1]: continue
                ii = i-center
                jj = j-center
                kk = k-center
                r = float(ii)*astar+float(jj)*bstar+float(kk)*cstar
                r = np.sqrt(np.sum(r**2))*float(iscale)
                intr = int(round(r))
                Rmodel[i,j,k] = intr
                
                isize = (iwindow-1)/2
                for delta in range(-isize, isize+1):
                    if (intr+delta)>=0 and (intr+delta)<len(distri):
                        distri[intr+delta] += idata[i,j,k]
                        weight[intr+delta] += 1.

    index = np.where(weight>0)
    distri[index] = distri[index]/weight[index]
    return [distri, Rmodel]

@jit
def radialBackground(_volume, _volumeMask=None, volumeCenter=None, threshold=(-100,1000), window=1, scale=1, Basis=None):
    volume = _volume.copy()

    if _volumeMask is None:
        volumeMask = np.ones(volume.shape).astype(int)
    else:
        volumeMask = _volumeMask.copy()

    if threshold is not None:
        volumeMask[(volume<threshold[0]) | (volume>threshold[1])] = 0

    volume *= volumeMask

    nx, ny, nz = volume.shape
    xaxis, yaxis, zaxis = scripts.utils.meshgrid3D(volume.shape, center=volumeCenter)

    if Basis is not None:
        vecA = Basis[:,0].reshape((1,3))
        vecB = Basis[:,1].reshape((1,3))
        vecC = Basis[:,2].reshape((1,3))

        xaxis = xaxis.reshape((nx,ny,nz,1)) * vecA
        yaxis = yaxis.reshape((nx,ny,nz,1)) * vecB
        zaxis = zaxis.reshape((nx,ny,nz,1)) * vecC

        sumArray = xaxis + yaxis + zaxis

        radius = np.sqrt(np.sum(sumArray**2, axis=3)) * scale
        radius = np.around(radius).astype(int)
    else:
        radius = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
        radius = np.around(radius).astype(int)

    radius1d = np.zeros(int(np.amax(radius))+1)
    weight1d = np.zeros(radius1d.shape)

    hwindow = int((window-1.)/2.)
    rmax = len(radius1d)-1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = int(radius[i,j,k])
                for h in range(-hwindow, hwindow+1):
                    if r+h >= 0 and r+h<=rmax:
                        radius1d[r+h] += volume[i,j,k]
                        weight1d[r+h] += volumeMask[i,j,k]

    index = np.where(weight1d>0)
    radius1d[index] /= weight1d[index]

    _radialBackground = np.zeros(volume.shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = int(radius[i,j,k])
                _radialBackground[i,j,k] = radius1d[r]

    return _radialBackground



def stats(HKLI_1, lattice_constant_A_deg):
    if len(HKLI_1) < 10 or HKLI_1.shape[1] != 4:
        print "Bad data"
        return None 

    a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg = lattice_constant_A_deg
    (vecx_A, vecy_A, vecz_A, recH_Ainv, recK_Ainv, recL_Ainv) = \
            scripts.crystal.lattice2vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)

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
            scripts.crystal.lattice2vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)
    
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

def correlation(HKLI_1, HKLI_2, lattice_constant_A_deg=None, res_shell_step_Ainv=None, \
                num_res_shell=20, res_cut_off_high_A=1.5, res_cut_off_low_A=50, num_ref_uniform=True, \
                vmin = -1000, vmax=1020):
    ## num_res_shell > res_shell_step_Ainv
    ## res_cut_off_low_A has larger nunber
    ## res_cut_off_high_A has smaller number
    ## only count values inside [vmin, vmax)

    assert lattice_constant_A_deg is not None

    a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg = lattice_constant_A_deg
    (vecx_A, vecy_A, vecz_A, recH_Ainv, recK_Ainv, recL_Ainv) = \
            scripts.crystal.lattice2vector(a_A, b_A, c_A, alpha_deg, beta_deg, gamma_deg)

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
    cx = int((nx-1.0)/2.0)
    cy = int((ny-1.0)/2.0)
    cz = int((nz-1.0)/2.0)

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



@jit
def meanf(idata, _scale = 3, clim=(0,50)):
    delta = (_scale - 1)/2
    idata = idata.astype(float)
    newList = idata.copy()
    for i in range(len(idata)):
        istart = i-delta
        iend = i+delta
        if istart < 0: istart = 0
        if iend > len(idata)-1: iend = len(idata)+1
        Temp = idata[istart: iend+1].copy()
        Temp = Temp[np.where(Temp>clim[0])].copy()
        Temp = Temp[np.where(Temp<clim[1])].copy()
        if Temp.shape[0] == 0: continue
        newList[i] = np.nanmean(Temp)
    return newList


def volume2txt(volume, fsave="tmp.txt", _vMask=None, vmin=-100, vmax=1000, headers=None):
    if _vMask is None:
        vMask = np.ones(volume.shape).astype(int)
    else:
        vMask = _vMask.copy()

    if vmin is None:
        vmin = np.amin(volume)-1
    if vmax is None:
        vmax = np.amax(volume)+1

    (nx, ny, nz) = volume.shape
    cx = (nx-1)/2
    cy = (ny-1)/2
    cz = (nz-1)/2

    fw = open(fsave, "w")
    if headers is not None:
        for line in headers:
            if line.endswith("\n"):
                fw.write(line)
            else:
                fw.write(line+"\n")

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                h = x-cx
                k = y-cy
                l = z-cz
                val = volume[x,y,z]

                if vMask[x,y,z]==0:
                    continue
                if val<vmin or val>vmax:
                    continue

                val = round(val, 3)

                string = str(h).rjust(4)+str(k).rjust(4)+str(l).rjust(4)+str(val).rjust(10)+"\n"

                fw.write(string)

    fw.close()


def volume2Phenix(volume, fsave="tmp.txt", _vMask=None, vmin=-50, vmax=1000, headers=None):
    if _vMask is None:
        vMask = np.ones(volume.shape).astype(int)
    else:
        vMask = _vMask.copy()

    if vmin is None:
        vmin = np.amin(volume)-1
    if vmax is None:
        vmax = np.amax(volume)+1

    (nx, ny, nz) = volume.shape
    cx = (nx-1)/2
    cy = (ny-1)/2
    cz = (nz-1)/2

    fw = open(fsave, "w")
    if headers is not None:
        for line in headers:
            if line.endswith("\n"):
                fw.write(line)
            else:
                fw.write(line+"\n")

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                h = x-cx
                k = y-cy
                l = z-cz
                val = volume[x,y,z]

                if vMask[x,y,z]==0:
                    continue
                if val<vmin or val>vmax:
                    continue

                val = round(val-vmin, 3)
                sqr = round(np.sqrt(val), 3)

                string = str(h).rjust(4)+str(k).rjust(4)+str(l).rjust(4)+str(val).rjust(8)+str(sqr).rjust(8)+"\n"

                fw.write(string)

    fw.close()



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


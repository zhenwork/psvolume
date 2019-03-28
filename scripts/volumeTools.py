from numba import jit
import numpy as np 
import mathTools


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
    model3d[index] = 0
    return model3d, weight

@jit
def distri(idata, astar, bstar, cstar, ithreshold=(-100,1000), iscale=1, iwindow=5):

    la = mathTools.length(astar)
    lb = mathTools.length(bstar)
    lc = mathTools.length(cstar)
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
    xaxis, yaxis, zaxis = mathTools.meshgrid3D(volume.shape, center=volumeCenter)

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

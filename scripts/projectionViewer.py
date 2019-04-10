import numpy as np
from numba import jit
    
@jit
def pViewer(volume, vector, center=None, thr=None, depth=None, stretch=1.0):
    """
    volume is a 3D matrix, whose three directions are x,y,z respectively; 
    vector is the viewing direction like (1,2,1), it doesn't have to be an unit vector; 
    center is the origin of volume, if not set, it will be the center of volume matrix;
    thr is the low/high threshold like thr=(0,10), other values are transparent; 
    depth indicates the viewing transparency.
    """
    (nx, ny, nz) = volume.shape
    if center is not None: (cx, cy, cz) = center
    else: (cx, cy, cz) = np.array(volume.shape)*0.5-0.5
    x = np.arange(nx)-cx*1.0
    y = np.arange(ny)-cy*1.0
    z = np.arange(nz)-cz*1.0
    (xaxis, yaxis, zaxis) = np.meshgrid(x,y,z,indexing='ij')

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
    
    volx = xaxis*px[0] + yaxis*px[1] + zaxis*px[2]
    voly = xaxis*py[0] + yaxis*py[1] + zaxis*py[2]
    volz = xaxis*pz[0] + yaxis*pz[1] + zaxis*pz[2]

    volx = np.around(volx).astype(int)
    voly = np.around(voly).astype(int)
    if depth is not None: volz = np.around(volz/depth).astype(int)

    xmin = np.amin(volx)
    xmax = np.amax(volx)
    ymin = np.amin(voly)
    ymax = np.amax(voly)

    vmin = np.amin(volume)
    vmax = np.amax(volume)
    
    detvalue = np.ones((xmax-xmin+1, ymax-ymin+1))*(vmin-10)
    detpolar = np.ones((xmax-xmin+1, ymax-ymin+1))*(np.amin(volz)-1)
    detcount = np.zeros((xmax-xmin+1, ymax-ymin+1))

    if thr[0] is None: thr[0]=vmin-1
    if thr[1] is None: thr[1]=vmax+1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value = volume[i,j,k]
                if value<thr[0] or value>thr[1]:
                    continue
                dx = volx[i,j,k]-xmin
                dy = voly[i,j,k]-ymin
                if volz[i,j,k]<detpolar[dx,dy]: 
                    continue
                elif volz[i,j,k]>detpolar[dx,dy]:
                    detvalue[dx,dy] = value
                    detpolar[dx,dy] = volz[i,j,k]
                    detcount[dx,dy] = 1
                else:
                    detvalue[dx,dy]  = value+detvalue[dx,dy]*detcount[dx,dy]
                    detvalue[dx,dy] /= detcount[dx,dy]+1
                    detcount[dx,dy] += 1
    return detvalue

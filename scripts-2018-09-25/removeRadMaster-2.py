from fileManager import *
from removeRadClient import *
from mpi4py import MPI
from numba import jit
from shutil import copyfile

import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="image folder", default=".", type=str)
parser.add_argument("-o","--o", help="subImage", default="subImage", type=str)
parser.add_argument("-vmin","--vmin", help="minimum value", default=0.0, type=float)
parser.add_argument("-nmax","--nmax", help="maximum value", default=-1, type=int)
args = parser.parse_args()


zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[nmax, allFile] = zio.counterFile(folder_i, title='.slice')
if args.nmax == -1: 
    args.nmax = int(nmax)
folder_o = path_i + "/" + args.o


if comm_rank == 0:
    print '### Path  : ', path_i
    print '### Folder: ', folder_i  
    print '### save Folder: ', folder_o  
    if not os.path.exists(folder_o ): 
        os.mkdir(folder_o )
    print '### save folder: '+ folder_o 
else:
    while not os.path.isdir(folder_o ): pass


@jit
def getBack(image=None, mask=None, center=None, radius = None):
    nx, ny = image.shape
    cx, cy = center

    if radius is None:
        x = np.arange(nx) - cx
        y = np.arange(ny) - cy

        xaxis, yaxis = np.meshgrid(x,y,indexing="ij")
        radius = np.sqrt(xaxis**2+yaxis**2)
        radius = np.around(radius/3.).astype(int) * 3
        radius = radius.astype(int)


    val = np.zeros(np.amax(radius) + 1)
    wei = np.zeros(np.amax(radius) + 1)

    for i in range(nx):
        for j in range(ny):
            r = radius[i,j]
            val[r] += mask[i,j] * image[i,j]
            wei[r] += mask[i,j]

    index = np.where(wei>0)
    val[index] /= wei[index]

    back = np.zeros(image.shape)

    for i in range(nx):
        for j in range(ny):
            r = radius[i,j]
            back[i,j] = val[r]

    return back




####################################################
idx = 0
fname = folder_i + "/" + str(idx).zfill(5)+'.slice'
Geo = zio.get_image_info(fname)
image = zf.h5reader(fname, 'image')

nx, ny = image.shape
cx, cy = Geo["center"]

x = np.arange(nx) - cx
y = np.arange(ny) - cy

xaxis, yaxis = np.meshgrid(x,y,indexing="ij")
radius = np.sqrt(xaxis**2+yaxis**2)
radius = np.around(radius/3.).astype(int) * 3
radius = radius.astype(int)
#####################################################



sep = np.linspace(0, args.nmax, comm_size+1).astype('int')

for idx in range(sep[comm_rank], sep[comm_rank+1]):
    fname = folder_i + "/" + str(idx).zfill(5)+'.slice'
    Geo = zio.get_image_info(fname)
    
    image = zf.h5reader(fname, 'image')
    mask = np.zeros(image.shape)
    mask[image > args.vmin] = 1.

    radBack = getBack(image=image, mask=mask, center=Geo["center"], radius=radius)

    image = image - radBack

    index = np.where(mask < 0.5)
    image[index] = -1024

    fsave = folder_o  + '/' + str(idx).zfill(5)+'.slice'

    copyfile(fname, fsave)
    zf.h5modify(fsave, "image", image)
    
    print '### Rank: '+str(comm_rank).rjust(3)+' finished image: '+str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1]) 
import time
from fileManager import *
from loadDataTools import *
from imageProcessTools import *
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default=None)
parser.add_argument("-o","--o", help="save folder", default=None)
parser.add_argument("-xds","--xds", help="xds file", default=None)
parser.add_argument("-nmin","--nmin", help="smallest number", default=0, type=int)
parser.add_argument("-nmax","--nmax", help="largest number", default=-1, type=int)
parser.add_argument("-imageType","--imageType", help="image type", default="PILATUS", type=str)
args = parser.parse_args()


zf = iFile()
zio = IOsystem()
[path_o, folder_o] = zio.get_path_folder(args.o)
if args.fname is not None:
    [path_i, folder_i] = zio.get_path_folder(args.fname)
    suffix_i = zio.get_suffix(args.fname)
    [counter, selectFile] = zio.counterFile(path_i, title=suffix_i)
    print "## Total number in %s is:%5d" % (path_i, counter)

if args.nmax==-1: args.nmax = counter+args.nmin

## print some parameters
if comm_rank == 0:
    print "## path_i: %s"%path_i
    print "## folder_i: %s"%folder_i
    print "## path_o: %s"%path_o
    print "## folder_o: %s"%folder_o
    print "## Process %5d - %5d"%(args.nmin, args.nmax)
    if not os.path.isdir(folder_o):
        os.makedirs(folder_o)
else:
    while not os.path.isdir(folder_o): 
        time.sleep(5)


# load xds indexing files
if args.xds is not None:
    print "### load xds file: %s"%args.xds
    [Geo, Bmat, invBmat, invAmat, lattice] = load_GXPARM_XDS(args.xds)
else: 
    raise Exception('### Not a valid xds file')


## make the general mask
mask = get_users_mask(Geo, imageType="PILATUS")
Mask = expand_mask(mask, cwin=(1,1), value=0)


## save mask and lattice to image.process file
if comm_rank == 0:
    Filename = path_o+'/image.process'
    zf.h5writer(Filename, 'mask', mask)
    zf.h5modify(Filename, 'Mask', Mask)
    zf.h5modify(Filename, 'Bmat', Bmat)
    zf.h5modify(Filename, 'invBmat', invBmat)
    zf.h5modify(Filename, 'lattice', lattice)

    ## Test the Geometry especially about the shape
    ## FIXME: what if the image one doesn't exist
    fileOne = args.fname.replace('#####', str(1).zfill(5))
    imageOne = load_image(fileOne)
    assert imageOne.shape[0]==int(Geo['shape'][0])
    assert imageOne.shape[1]==int(Geo['shape'][1])


## save image to the rawImage folder
sep = np.linspace(args.nmin, args.nmax, comm_size+1).astype('int')
print "## Rank:%3d/%3d processes: %4d - %4d"%(comm_rank,comm_size,sep[comm_rank],sep[comm_rank+1])

for idx in range(sep[comm_rank], sep[comm_rank+1]):

    if args.fname is not None:
        fname = args.fname.replace('#####', str(idx).zfill(5))

    ## mask out bad pixels
    image = load_image(fname)
    tmpMask=get_tmpMask(image, vmin=0.001, vmax=100000)
    index = np.where(tmpMask*mask==0)
    image[index] = -1

    ## get the quaternion
    quaternion = angle2quaternion(idx*Geo['increment'], axis='y')
    R1 = Quat2Rotation(quaternion)

    # FIXME: calculate the relations between A,U,B
    ## matrix = invBmat.dot(invUmat.dot(R1))
    matrix = invAmat.dot(R1)

    ## save images
    fsave = folder_o+'/'+str(idx).zfill(5)+'.slice'
    save_image(fsave,image,Geo)
    zf.h5modify(fsave, 'rotation', matrix)
    zf.h5modify(fsave, 'scale', 1.0)

    print '### Rank:'+str(comm_rank).rjust(3)+' ::::: '+str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1])


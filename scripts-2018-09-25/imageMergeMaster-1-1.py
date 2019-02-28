import os
import numpy as np 
from mpidata import *
from fileManager import *
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-mode","--mode", help="matrix", default="hkl", type=str)
parser.add_argument("-peak","--peak", help="keep the bragg peak or not", default=0, type=int)
parser.add_argument("-vSampling","--vSampling", help="num of images to process", default=1, type=int)
# parser.add_argument("-vCenter","--vCenter", help="num of images to process", default=60, type=int)
parser.add_argument("-nmin","--nmin", help="minimum image number", default=0, type=int)
parser.add_argument("-nmax","--nmax", help="maximum image number", default=-1, type=int)
parser.add_argument("-thrmin","--thrmin", help="minimum pixel value", default=0.0, type=float)
parser.add_argument("-voxel","--voxel", help="voxel exist or not", default=".", type=str)
parser.add_argument("-choice","--choice", help="select specific images", default="all", type=str)
parser.add_argument("-list","--list", help="merge list", default=".", type=str)
parser.add_argument("-cc12","--cc12", help="calculate the CC1/2", default=0, type=int)
parser.add_argument("-mask","--mask", help="numpy array", default=None, type=str)
args = parser.parse_args()


zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[nmax, allFile] = zio.counterFile(folder_i, title='.slice')
if args.nmax == -1: args.nmax = int(nmax)

Vol = {}
Vol['volumeSampling'] = int(args.vSampling)
Vol['volumeCenter'] = int(args.vSampling)*60
Vol['volumeSize'] = 2*Vol['volumeCenter']+1

model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)

volume = np.zeros([Vol['volumeSize']]*3)
vMask  = np.zeros([Vol['volumeSize']]*3)

volumeRaw = np.zeros([Vol['volumeSize']]*3)
vMaskRaw  = np.zeros([Vol['volumeSize']]*3)

if args.cc12 != 0:
    model3d_1 = np.zeros([Vol['volumeSize']]*3)
    weight_1  = np.zeros([Vol['volumeSize']]*3)

    model3d_2 = np.zeros([Vol['volumeSize']]*3)
    weight_2  = np.zeros([Vol['volumeSize']]*3)
    
    model3d = np.array([model3d_1, model3d_2])
    weight = np.array([weight_1, weight_2])


# FIXME: This is specific for the snc dataset
#########################
voxel = None
if args.voxel != ".":
    print "## voxel exist: %s" % args.voxel
    voxel = np.load(args.voxel)
    voxel = voxel.T
    print "## Loaded the voxel file ... "
#########################


## If the merge list exists:
if args.list == ".":
    mergeList = np.arange(args.nmin, args.nmax)
    print "### Merge All from %d to %d " % (args.nmin, args.nmax)
else:
    print "### Merge List exists"
    mergeList = np.load(args.list)
    print "### Loaded the merge list"
if args.mask is not None:
    print "### loading mask"
    mask = np.load(args.mask)
    

x = np.arange(121) - 60
y = np.arange(121) - 60
z = np.arange(121) - 60

xaxis, yaxis, zaxis = np.meshgrid(x,y,z,indexing="ij")
RR = np.sqrt(xaxis**2+yaxis**2+zaxis**2)


if comm_rank == 0:
    folder_o = zio.makeFolder(path_i, title='sr')

    print '### read from Path  : ', path_i
    print '### read from Folder: ', folder_i
    print "### save to Folder : ", folder_o
    print "### Images:  [", args.nmin, args.nmax, ')'
    print "### Mode  : ", args.mode
    print "### Volume: ", model3d.shape
    print "### Center: ", Vol['volumeCenter']
    print "### Sampling: ", Vol['volumeSampling']

    for idx in range(args.nmin, args.nmax):

        fname = folder_i+'/'+str(idx).zfill(5)+'.slice'
        image = zf.h5reader(fname, 'image')
        Geo = zio.get_image_info(fname)
        image = image * 1. 

        if args.mask is not None:
            image[mask==0] = -1024
        
        sumIntens = round(np.sum(image[image>0]), 8)
        
        model3dRaw = np.zeros([Vol['volumeSize']]*3)
        weightRaw  = np.zeros([Vol['volumeSize']]*3)

        [model3dRaw, weightRaw] = ImageMerge_HKL(model3dRaw, weightRaw, image, Geo, Vol, Kpeak=args.peak, thrmin=args.thrmin)          

        model3d = model3dRaw.copy()
        weight = weightRaw.copy()

        model3d = (model3d + model3d[::-1, ::-1, ::-1] + model3d[::-1, :, ::-1] + model3d[:, ::-1, :]) 
        weight = (weight + weight[::-1, ::-1, ::-1] + weight[::-1, :, ::-1] + weight[:, ::-1, :]) 

        if idx == args.nmin:
            volume = model3d.copy() 
            vMask = weight.copy() 
            volumeRaw = model3dRaw.copy()
            vMaskRaw = weightRaw.copy()
            print "### adding first pattern", idx
        else:
            index = np.where( (vMask>0) & (weight>0) & (RR<35) )
            old = volume[index]/vMask[index]
            new = model3d[index]/weight[index]

            scale = np.dot(old, new)/np.dot(new, new)

            volume += scale * model3d
            vMask += weight

            volumeRaw += scale * model3dRaw
            vMaskRaw += weightRaw

            old = None
            new = None

            print "### adding to existing volume", len(index[0]), scale, np.sum(vMaskRaw)

            # FIXME: This is specific for snc dataset:
            # [model3d, weight] = ImageMerge_HKL_VOXEL(model3d, weight, image, Geo, Vol, Kpeak=args.peak, voxel=voxel, idx=idx, thrmin = args.thrmin)
        
        print '### rank ' + str(comm_rank).rjust(3) + ' is processing file: '+str(args.nmin)+'/'+str(idx)+'/'+str(args.nmax) +'  sumIntens: '+str(sumIntens).ljust(10)


    model3d = volumeRaw.copy()
    weight = vMaskRaw.copy()

    model3d = ModelScaling(model3d, weight)
    pathIntens = folder_o+'/merge.volume'
    Smat = zf.h5reader(folder_i+'/00000.slice', 'Smat')

    
    print "### saving File: ", pathIntens
    ThisFile = zf.readtxt(os.path.realpath(__file__))
    zf.h5writer(pathIntens, 'execute', ThisFile)
    zf.h5modify(pathIntens, 'sampling', Vol['volumeSampling'])
    chunks = list(model3d.shape[1:])
    chunks.insert(0,1)
    zf.h5modify(pathIntens, 'intens', model3d, chunks=tuple(chunks), opts=7)
    zf.h5modify(pathIntens, 'weight', weight,  chunks=tuple(chunks), opts=7)
    zf.h5modify(pathIntens, 'Smat', Smat)


"""
1. Merge without background subtraction 
2. Scale based on overall intensity of crystal diffraction
"""

import os,sys
import numpy as np
sys.path.append('/reg/neh/home5/zhensu/Develop/psvolume')
import scripts.expAgent as expAgent
from scripts.mpidata import *
import scripts.fileManager as fileManager
from numba import jit

PsvolumeManager = fileManager.PsvolumeManager()
FileSystem = fileManager.FileSystem()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fsave","--fsave", help="save folder", default=None, type=str)
parser.add_argument("-fname","--fname", help="input files", default=None, type=str)
parser.add_argument("-fback","--fback", help="input files", default=None, type=str)
parser.add_argument("-xds","--xds", help="xds file", default=None, type=str)
parser.add_argument("-label","--label", help="xds file", default=None, type=str)


parser.add_argument("-special","--special", help="special params", default="wtich", type=str)
parser.add_argument("-firMask","--firMask", help="first mask?", default=0, type=int)
parser.add_argument("-expMask","--expMask", help="expand mask?", default=1, type=int)
parser.add_argument("-scaling","--scaling", help="scaling method", default="rad", type=str)
args = parser.parse_args()



"""
self.removeBadPixels()
# 2. expand mask (True is default)
if expMask:
    self.expandMask()
# 3. deep remove bad pixels
self.deepRemoveBads() 
# 4. polarization correction
self.polarizationCorrection()
# 5. solid angle correction
self.solidAngleCorrection()
# 6. remove Bragg peaks
self.removeBragg()
"""








####
files = FileSystem.listFileWithFind(args.fname.replace("#####", "*"))
nmax = len(files)

assign = np.linspace(1, nmax+1, comm_size+1).astype(int)
print ">>>> %3d process [ %4d, %4d ) in %4d"%(comm_rank, assign[comm_rank], assign[comm_rank+1], nmax)




## reference pattern ############################
refile = args.fname.replace("#####", "00001")
imageAgent = expAgent.ImageAgent()
imageAgent.loadImage(refile)
imageAgent.loadImage(args.xds)


if args.fback is not None:
    reback = args.fback.replace("#####", "00001")
    back = imageAgent.readfile(filename=reback)["image"]
    imageAgent.image = (imageAgent.image - 0.3*back) * imageAgent.mask

imageAgent.preprocess(expMask=args.expMask, notation=args.special)
imageAgent.radprofile()
refData = imageAgent.todict()
imageAgent = None
## reference pattern ############################


for idx in range(assign[comm_rank], assign[comm_rank+1]):
    

    filename = args.fname.replace("#####", "%.5d"%idx)
    imageAgent = expAgent.ImageAgent()
    imageAgent.loadImage(filename)
    imageAgent.loadImage(args.xds) 
    print "Loading: ", filename

    if args.fback is not None:
        fileback = args.fback.replace("#####", "%.5d"%idx)
        back = imageAgent.readfile(filename=fileback)["image"]
        imageAgent.image = (imageAgent.image - 0.3*back) * imageAgent.mask


    if args.firMask:
        imageAgent.mask *= refData["firMask"]
        imageAgent.image *= refData["firMask"]


    imageAgent.preprocess(expMask=args.expMask, notation=args.special)
    imageAgent.radprofile()


    if args.scaling.lower() == "sum":
        imageAgent.scaling(reference = refData, rmin=160, rmax=400)
    elif args.scaling.lower() == "ave":
        imageAgent.scaling(reference = refData, mode="ave", rmin=160, rmax=400)
    elif args.scaling.lower() == "rad":
        imageAgent.scaling(reference = refData, mode="rad", rmin=160, rmax=400)
    elif args.scaling.lower() == "overall":
        imageAgent.scaling(reference = refData, rmin=0, rmax=100000)
    else:
        raise Exception("!! ERROR")

    ## save file
    fsave = args.fsave.replace("#####", "%.5d"%idx)
    PsvolumeManager.psvm2h5py(imageAgent.todict(), fsave)
    imageAgent = None


## Finish: 
if comm_rank != 0:
    md=mpidata()
    md.small.rank = comm_rank
    md.small.finish = True
    md.send()
    md = None

else:
    for nrank in range(comm_size-1):
        md=mpidata()
        md.recv()
        recvRank = md.small.rank
        md = None
    flock = os.path.dirname(args.fsave) + "/.lock"
    with open(flock, "w") as f:
        f.write("finish")


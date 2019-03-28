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
parser.add_argument("-xds","--xds", help="xds file", default=".", type=str)
parser.add_argument("-label","--label", help="xds file", default=".", type=str)
args = parser.parse_args()

files = FileSystem.listFileWithFind(args.fname.replace("#####", "*"))
nmax = len(files)

assign = np.linspace(1, nmax+1, comm_size+1).astype(int)
print ">>>> %3d process [ %4d, %4d ) in %4d"%(comm_rank, assign[comm_rank], assign[comm_rank+1], nmax)




## reference pattern ############################
refile = args.fname.replace("#####", "00001")
reback = args.fback.replace("#####", "00001")

imageAgent = expAgent.ImageAgent()
imageAgent.loadImage(refile)
imageAgent.loadImage(args.xds)
back = imageAgent.readfile(filename=reback)["image"]
imageAgent.image = (imageAgent.image - 0.3*back) * imageAgent.mask
imageAgent.preprocess()
refData = imageAgent.todict()
imageAgent = None
## reference pattern ############################


for idx in range(assign[comm_rank], assign[comm_rank+1]):
    
    filename = args.fname.replace("#####", "%.5d"%idx)
    fileback = args.fback.replace("#####", "%.5d"%idx)
    fsave = args.fsave.replace("#####", "%.5d"%idx)
    print "Loading: ", filename
    

    imageAgent = expAgent.ImageAgent()
    imageAgent.loadImage(filename)
    imageAgent.loadImage(args.xds) 
    back = imageAgent.readfile(filename=fileback)["image"]
    imageAgent.image = (imageAgent.image - 0.3*back) * imageAgent.mask
    #########
    # imageAgent.mask *= refData["mask"]
    # imageAgent.image *= refData["mask"]
    #########
    imageAgent.preprocess()
    imageAgent.scaling(reference = refData, rmin=160, rmax=400)
    

    PsvolumeManager.psvm2h5py(imageAgent.todict(), fsave)
    imageAgent = None

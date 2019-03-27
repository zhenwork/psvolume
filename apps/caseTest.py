"""
Merge without background subtraction based on overall intensity of crystal diffraction
"""

import os,sys
import numpy as np
sys.path.append('/reg/data/ana04/users/zhensu/Software/psvolume')
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
parser.add_argument("-xds","--xds", help="xds file", default=".", type=str)
args = parser.parse_args()

files = FileSystem.listFileWithFind(args.fname.replace("#####", "*"))
nmax = len(files)

assign = np.linspace(1, nmax+1, comm_size+1).astype(int)
print ">>>> %3d process [ %4d, %4d ) in %4d"%(comm_rank, assign[comm_rank], assign[comm_rank+1], nmax)




## reference pattern
refile = args.fname.replace("#####", "00001")
imageAgent = expAgent.Diffraction()
imageAgent.loadImage(refile)
imageAgent.loadImage(args.xds)

imageAgent.preprocess()

refData = imageAgent.todict()
imageAgent = None



for idx in range(assign[comm_rank], assign[comm_rank+1]):
    
    filename = args.fname.replace("#####", "%.5d"%idx)
    fsave = args.fsave.replace("#####", "%.5d"%idx)
    print "Loading: ", filename
    
    imageAgent = expAgent.Diffraction()
    imageAgent.loadImage(filename)
    imageAgent.loadImage(args.xds)
    imageAgent.mask *= refData["mask"]
    imageAgent.image *= refData["mask"]
    imageAgent.preprocess()
    imageAgent.scaling(reference = refData, rmin=160, rmax=400)
    
    PsvolumeManager.psvm2h5py(imageAgent.todict(), fsave)
    imageAgent = None
    
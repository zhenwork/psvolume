import os,sys
import numpy as np
sys.path.append('/reg/neh/home5/zhensu/Develop/psvolume')
import scripts.fileManager as fileManager
from numba import jit
import scripts.mathTools as mathTools
import scripts.volumeTools as volumeTools

H5FileManager = fileManager.H5FileManager()
PsvolumeManager = fileManager.PsvolumeManager()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default=None, type=str) 
parser.add_argument("-cutoff","--cutoff", help="input files", default=4, type=int) 
args = parser.parse_args()


psvm = PsvolumeManager.h5py2psvm(args.fname)

volume = psvm["volume"]
weight = psvm["weight"]
Amat = psvm["Amat"]
Bmat = psvm["Bmat"]
vMask = (weight>=args.cutoff).astype(int)

index = np.where(weight<args.cutoff)
volume[index] = -1024
weight[index] = 0
vMask[index] = 0



## symmetry
volumeSym, weightSym = volumeTools.volumeSymmetrize(volume, _volumeMask=vMask, _threshold=(-100,1000), symmetry="P1211")
volumeSym[weightSym==0] = -1024

H5FileManager.h5modify(args.fname, "volumeSym", volumeSym)


## background
symBack = volumeTools.radialBackground(volumeSym, _volumeMask=(weightSym>0).astype(int), threshold=(-100,1000), window=5, \
                                       Basis=Bmat/mathTools.length(Bmat[:,1]), scale=4)
rawBack = volumeTools.radialBackground(volume, _volumeMask=vMask, threshold=(-100,1000), window=5, \
                                       Basis=Bmat/mathTools.length(Bmat[:,1]), scale=4)
volumeSymSub = volumeSym - symBack
volumeSub = volume - rawBack
volumeSymSub[weightSym==0] = -1024
volumeSub[weight<1] = -1024


H5FileManager.h5modify(args.fname, "volumeSymSub", volumeSymSub)
H5FileManager.h5modify(args.fname, "volumeSub", volumeSub)
H5FileManager.h5modify(args.fname, "symBack", symBack)
H5FileManager.h5modify(args.fname, "rawBack", rawBack)


## convert to good
a = Bmat[:,0].copy()
b = Bmat[:,1].copy()
c = Bmat[:,2].copy()

astar = a / mathTools.length(b)
bstar = b / mathTools.length(b)
cstar = c / mathTools.length(b)

volumeXYZ, _ = volumeTools.hkl2volume(volume, astar, bstar, cstar, _volumeMask = vMask, ithreshold=(-100,1000))

volumeSubXYZ, _ = volumeTools.hkl2volume(volumeSub, astar, bstar, cstar, _volumeMask = vMask, ithreshold=(-100,1000))

volumeSymSubXYZ, _ = volumeTools.hkl2volume(volumeSymSub, astar, bstar, cstar, _volumeMask = (weightSym>0).astype(int), ithreshold=(-100,1000))
volumeSymXYZ, _ = volumeTools.hkl2volume(volumeSym, astar, bstar, cstar, _volumeMask = (weightSym>0).astype(int), ithreshold=(-100,1000))


H5FileManager.h5modify(args.fname, "volumeSymSubXYZ", volumeSymSubXYZ)
H5FileManager.h5modify(args.fname, "volumeSymXYZ", volumeSymXYZ)
H5FileManager.h5modify(args.fname, "volumeSubXYZ", volumeSubXYZ)
H5FileManager.h5modify(args.fname, "volumeXYZ", volumeXYZ)
H5FileManager.h5modify(args.fname, "weightSym", weightSym)


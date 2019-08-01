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
import scripts.mergeTools as mergeTools

PsvolumeManager = fileManager.PsvolumeManager()
FileSystem = fileManager.FileSystem()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fsave","--fsave", help="save folder", default=None, type=str)
parser.add_argument("-fname","--fname", help="input files", default=None, type=str)
parser.add_argument("-xds","--xds", help="xds file", default=None, type=str)
parser.add_argument("-label","--label", help="xds file", default=None, type=str)

parser.add_argument("-startidx","--startidx", help="1+startidx, can be -1 if you start from 0", default=0, type=int)
parser.add_argument("-special","--special", help="special params", default="wtich", type=str)
parser.add_argument("-firMask","--firMask", help="first mask?", default=0, type=int)
parser.add_argument("-expMask","--expMask", help="expand mask?", default=1, type=int)
parser.add_argument("-scaling","--scaling", help="scaling method", default="rad", type=str)

parser.add_argument("-rmin","--rmin", help="scaling rmin", default=100, type=int)
parser.add_argument("-rmax","--rmax", help="scaling rmax", default=1000, type=int)
parser.add_argument("-vmin","--vmin", help="scaling vmin", default=20, type=int)
parser.add_argument("-bmin","--bmin", help="scaling box size", default=0.0, type=float)
parser.add_argument("-bmax","--bmax", help="scaling box maximum size", default=0.15, type=float)
args = parser.parse_args()




####
files = FileSystem.listFileWithFind(args.fname.replace("#####", "*"))
nmax = len(files)

startidx = 1+args.startidx
assign = np.linspace(1+args.startidx, nmax+1+args.startidx, comm_size+1).astype(int)
print ">>>> %3d process [ %4d, %4d ) in %4d"%(comm_rank, assign[comm_rank], assign[comm_rank+1], nmax)
print ">>>> start from image: %.5d"%startidx



sumPeak = np.zeros(nmax)
avePeak = np.zeros(nmax)
cntPeak = np.zeros(nmax)


for idx in range(assign[comm_rank], assign[comm_rank+1]):
    
    refidx = idx - startidx

    filename = args.fname.replace("#####", "%.5d"%idx )
    print "Loading image: ", refidx, filename
    imageAgent = expAgent.ImageAgent()
    imageAgent.loadImage(filename)
    imageAgent.loadImage(args.xds) 

    ## remove Bad pixels
    imageAgent.removeBadPixels(notation=args.special, vmin=0.001, vmax=100000, rmin=40, rmax=None)

    ## expand pixel mask
    if args.expMask is True or int(args.expMask) == 1:
        imageAgent.expandMask()


    ## corrections
    imageAgent.polarizationCorrection()
    imageAgent.solidAngleCorrection()

    image = imageAgent.image.copy()

    ## peak, radial Mask
    mask = imageAgent.mask.copy()


    mask *= 1 - imageAgent.buildPeakMask(bmin=args.bmin, bmax=args.bmax)
    median_backg = imageAgent.medianBack(window=(11,11))
    mask *= ((image - median_backg)>args.vmin)
    mask *= imageAgent.circleMask(rmin=args.rmin, rmax=args.rmax)


    sumPeak[refidx] = np.sum((image-median_backg) * mask)
    cntPeak[refidx] = np.sum(mask)
    avePeak[refidx] = sumPeak[refidx] * 1.0 / cntPeak[refidx]

    fsave = args.fsave.replace("#####", "%.5d"%idx)
    ## save original image, overall mask
    print "## Image %.5d ==> min=%5.1f, max=%5.1f"%(idx, np.amin(imageAgent.image), np.amax(imageAgent.image))
    zf.h5writer(fsave, "image", image)
    zf.h5modify(fsave, "median", median_backg)
    zf.h5modify(fsave, "mask", mask)
    imageAgent = None
    median_backg = None
    image = None
    


## Finish: 

if comm_rank == 0:
    for i in range(comm_size-1):
        md=mpidata()
        md.recv()
        sumPeak += md.sumPeak
        avePeak += md.avePeak
        cntPeak += md.cntPeak
        recvRank = md.small.rank
        md = None
        print '### received file from ' + str(recvRank).rjust(3)

    fsave = args.fsave.replace("#####", "final")
    zf.h5writer(fsave, "sumPeak", sumPeak)
    zf.h5modify(fsave, "avePeak", avePeak)
    zf.h5modify(fsave, "cntPeak", cntPeak)
else:
    md=mpidata()
    md.addarray('sumPeak', sumPeak)
    md.addarray('avePeak', avePeak)
    md.addarray('cntPeak', cntPeak)
    md.small.rank = comm_rank
    md.send()
    md = None
    print '### Rank ' + str(comm_rank).rjust(4) + ' is sending file ... '
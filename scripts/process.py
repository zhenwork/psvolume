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
import scripts.utils as utils
from numba import jit

PsvolumeManager = fileManager.PsvolumeManager()
FileSystem = fileManager.FileSystem()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname",      help="input files", default=None, type=str)
parser.add_argument("-fback","--fback",      help="backg files", default=None, type=str) 
parser.add_argument("-fsave","--fsave",      help="save folder", default=None, type=str)

parser.add_argument("-fxds","--fxds",        help="xds file",    default=None, type=str)
parser.add_argument("-fdials","--fdials",    help="dials file",  default=None, type=str)

parser.add_argument("-event","--event",      help="1-20,30,40", default=None, type=str) 
parser.add_argument("-refidx","--refidx",    help="reference idx", default=1, type=int) 

parser.add_argument("-scaling","--scaling",  help="overall/sum/ave/rad/dials",default="rad",type=str) 
parser.add_argument("-expmask","--expmask",  help="expand mask?",default=0,   type=int) 

parser.add_argument("-rmin","--rmin",        help="scaling rmin",default=160, type=int) 
parser.add_argument("-rmax","--rmax",        help="scaling rmax",default=500, type=int) 

parser.add_argument("-special","--special",  help="special params",default="wtich", type=str) 
args = parser.parse_args()


def process(event, args):
    filename = args.fname.replace("#####", "%.5d"%event )
    print "Loading image: ", filename 

    imageAgent = expAgent.ImageAgent()  
    imageAgent.loadImage(filename)   
    imageAgent.loadImage(args.fxds)   
    imageAgent.loadImage(args.fdials) 
    imageAgent.removeBadPixels(notation=args.special, vmin=None, vmax=100000, rmin=40, rmax=None)
    imageAgent.expandMask(1)

    if args.fback is not None:
        fileback = args.fback.replace("#####", "%.5d"%event ) 
        print "Loading backg: ", fileback

        backgAgent = expAgent.ImageAgent()
        backgAgent.loadImage(fileback)
        backgAgent.removeBadPixels(notation=args.special, vmin=None, vmax=100000, rmin=40, rmax=None)
        backgAgent.expandMask(1)

        imageAgent.image = (imageAgent.image - 0.3*backgAgent.image) * imageAgent.mask * backgAgent.mask
        imageAgent.mask *= backgAgent.mask
        backgAgent = None

    imageAgent.mask  *= (imageAgent.image > 0)
    imageAgent.image *= imageAgent.mask

    imageAgent.preprocess()
    imageAgent.radprofile()

    if args.scaling.lower() == "sum":
        imageAgent.scaling(reference = refData, mode="sum", rmin=args.rmin, rmax=args.rmax)
    elif args.scaling.lower() == "ave":
        imageAgent.scaling(reference = refData, mode="ave", rmin=args.rmin, rmax=args.rmax)
    elif args.scaling.lower() == "rad":
        imageAgent.scaling(reference = refData, mode="rad", rmin=args.rmin, rmax=args.rmax)
    elif args.scaling.lower() == "overall":
        imageAgent.scaling(reference = refData, mode="ave", rmin=0, rmax=100000)
    elif args.scaling.lower() == "dials":
        imageAgent.scaling(reference = refData, mode="dials", fdials=args.dials)

    return imageAgent





#### Get files
evtidx = utils.getevents(args.event)
idx = np.linspace(0,len(evtidx)+1,comm_size+1).astype(int)
assign = evtidx[idx[comm_rank]:idx[comm_rank+1]]
print "rank %d process [%d , %d], total=%d"%(comm_rank, assign[0], assign[-1], len(assign))   

#### create folder
if comm_rank == 0:
    fbase = os.path.dirname(args.fsave)
    if not os.path.isdir(fbase):
        os.makedirs(fbase)


#### process the reference file
imageAgent = process(args.refidx, args)
refData = imageAgent.todict()
imageAgent = None


#### process the pipeline
for idx,event in enumerate(evtidx):
    if event not in assign:
        continue

    imageAgent = process(event, args)
    print "## Image %.5d ==> min=%5.1f, max=%5.1f"%(idx, np.amin(imageAgent.image), np.amax(imageAgent.image))
    
    ## save file
    fsave = args.fsave.replace("#####", "%.5d"%idx)
    PsvolumeManager.psvm2h5py(imageAgent.todict(), fsave)
    imageAgent = None


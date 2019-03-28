import os,sys
import numpy as np
sys.path.append('/reg/neh/home5/zhensu/Develop/psvolume')
import scripts.expAgent as expAgent
from scripts.mpidata import *
import scripts.fileManager as fileManager
from numba import jit
import scripts.mathTools as mathTools

FileSystem = fileManager.FileSystem()
H5FileManager = fileManager.H5FileManager()
PsvolumeManager = fileManager.PsvolumeManager()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fsave","--fsave", help="save folder", default=None, type=str)
parser.add_argument("-fname","--fname", help="input files", default=None, type=str) 
args = parser.parse_args()

files = FileSystem.listFileWithFind(args.fname.replace("#####", "*"))
number = len(files)

assign = np.linspace(0, number, comm_size).astype(int)
volume = np.zeros((121,121,121))
weight = np.zeros((121,121,121))

if comm_rank!=0:
    print ">>>> %3d process [ %4d, %4d ) in %4d"%(comm_rank, assign[comm_rank-1], assign[comm_rank], number)
    
    mergeAgent = expAgent.MergeAgent()
    for idx in range(assign[comm_rank-1], assign[comm_rank]):
        print "add file: ", files[idx]
        mergeAgent.addfile(files[idx])
        
    mergeAgent.merge()
    
    md=mpidata()
    md.addarray('volume', mergeAgent.volume)
    md.addarray('weight', mergeAgent.weight)
    md.small.rank = comm_rank
    md.small.num = assign[comm_rank]-assign[comm_rank-1]
    md.send()
    md = None
    
else:
    for nrank in range(comm_size-1):
        md=mpidata()
        md.recv()
        volume += md.volume
        weight += md.weight
        recvRank = md.small.rank
        md = None
        print '### received file from ' + str(recvRank).rjust(2) + '/' + str(comm_size)
    
    
    index = np.where(weight>=4)
    volume[index] /= weight[index]
    index = np.where(weight<4)
    volume[index] = 0
    weight[index] = 0

    psvm = PsvolumeManager.h5py2psvm(files[0])
    Amat = psvm["Amat"]
    Bmat = psvm["Bmat"]
    H5FileManager.h5writer(args.fsave, 'volume', volume, chunks=(1,121,121), opts=7)
    H5FileManager.h5modify(args.fsave, 'weight', weight,  chunks=(1,121,121), opts=7)
    H5FileManager.h5modify(args.fsave, 'Amat', Amat)
    H5FileManager.h5modify(args.fsave, 'Bmat', Bmat)
    H5FileManager.h5modify(args.fsave, 'Smat', Bmat/mathTools.length(Bmat[:,1]))
    


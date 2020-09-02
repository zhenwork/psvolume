import os,sys
import numpy as np
from numba import jit

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.utils
import scripts.manager
import scripts.fsystem
import scripts.datafile 
from scripts.mpidata import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default=None, type=str)
parser.add_argument("-fsave","--fsave", help="save folder", default=None, type=str)
parser.add_argument("-save_dname","--save_dname", help="save folder", default="merged_volume", type=str)
parser.add_argument("-read_dname","--read_dname", help="merge dname in the fname", default="processed_image", type=str)
parser.add_argument("-select_idx","--select_idx", help="select idx of images", default=None, type=str)
parser.add_argument("--merge_by_averaging",help="cases: None, [], [1,2,3]",default=None,nargs="*")
args = parser.parse_args()


#### 
args.fsave = args.fsave or args.fname
if comm_rank == 0:
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)


#### load data from file
import_data = utils.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname])
num_images = len(import_data.get(args.read_dname))

select_idx = scripts.utils.get_array_list(args.select_idx) or range(num_images)
assign_idx = select_idx[0::comm_size]


#### process to do 
args.merge_by_averaging = scripts.utils.get_process_params(args.merge_by_averaging)


if comm_rank != 0:
    mergeAgent = scripts.manager.MergeAgent()
    for image_idx in assign_idx:
        file_name = import_data.get(args.read_dname)[image_idx]
        print "#### Rank %3d/%3d adds file: %s"%(comm_rank, comm_size, file_name) 
        mergeAgent.addfile(file_name)
    
    if args.merge_by_averaging.status:
        volume,weight = mergeAgent.merge_by_averaging(**args.merge_by_averaging.params)
    
    md=mpidata()
    md.addarray('volume', mergeAgent.volume)
    md.addarray('weight', mergeAgent.weight)
    md.small.rank = comm_rank
    md.small.num_merge = len(assign_idx)
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
        print '#### Rank 0 received file from Rank ' + str(recvRank).rjust(2) 
    
    index = np.where(weight>=4)
    volume[index] /= weight[index]
    index = np.where(weight<4)
    volume[index] = -1024
    weight[index] = 0

comm.Barrier()

if comm_rank == 0:
    volume_dir = os.path.dirname(import_data.get(args.merge_dname))
    volume_file = os.path.join(volume_dir,"%s.h5"%args.save_dname)
    data_save = {args.save_dname:volume_file}

    history = scripts.fsystem.H5manager.reader(args.fsave, "history")
    from sys import argv
    if not history:
        history = [" ".join(argv)]
    else:
        history.append(" ".join(argv))
    data_save["history"] = history
    scripts.fsystem.PVmanager.modify(data_save, args.fsave)
    
    ## write volume into files
    psvm = scripts.datafile.loadfile(file_name,"h5")
    Amat = psvm["Amat"]
    Bmat = psvm["Bmat"]

    scripts.fsystem.H5manager.modify(volume_file, 'volume', volume, chunks=(1,args.Vsize,args.Vsize), compression="gzip", compression_opts=7)
    scripts.fsystem.H5manager.modify(volume_file, 'weight', weight, chunks=(1,args.Vsize,args.Vsize), compression="gzip", compression_opts=7)
    scripts.fsystem.H5manager.modify(volume_file, 'Amat', Amat)
    scripts.fsystem.H5manager.modify(volume_file, 'Bmat', Bmat)
    scripts.fsystem.H5manager.modify(volume_file, 'Smat', Bmat/scripts.utils.length(Bmat[:,1]))



    
    


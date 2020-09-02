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
parser.add_argument("-fname","--fname",help="input files", default="./data.diffuse", type=str) 
parser.add_argument("-fsave","--fsave",help="save new files", default=None, type=str) 
parser.add_argument("-read_dname","--read_dname",help="save new files", default="processed_image", type=str) 
parser.add_argument("-save_dname","--save_dname",help="save new files", default="per_image_multiply_scale", type=str) 

parser.add_argument("--scale_by_radial_profile",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_overall_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_average_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--scale_by_dials_bragg",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
args = parser.parse_args() 


#### create folders
args.fsave = args.fsave or args.fname
if comm_rank == 0:
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)

#### load data from file
import_data = utils.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname,"dials_report_file"])
num_images = len(import_data.get(args.read_dname))


#### get process params
args.scale_by_dials_bragg = scripts.utils.get_process_params(args.scale_by_dials_bragg)
args.scale_by_radial_profile = scripts.utils.get_process_params(args.scale_by_radial_profile)
args.scale_by_overall_intensity = scripts.utils.get_process_params(args.scale_by_overall_intensity)
args.scale_by_average_intensity = scripts.utils.get_process_params(args.scale_by_average_intensity)

per_image_multiply_scale = np.zeros(num_images)
reference_obj = None

#### process the pipeline
for image_idx in range(num_images):
    if idx%comm_size != comm_size:
        continue

    file_name = import_data.get(args.read_dname)[image_idx]
    image_obj = scripts.manager.ImageAgent()
    image_obj.loadImage(file_name)

    # calculate overall intensity 
    if args.scale_by_dials_bragg.status:
        if not reference_obj:
            reference_idx = args.scale_by_dials_bragg.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx])
        per_image_multiply_scale[image_obj] = image_obj.scale_by_dials_bragg(**args.scale_by_dials_bragg.params)

    # calculate average intensity 
    elif args.scale_by_radial_profile.status:
        if not reference_obj:
            reference_idx = args.scale_by_radial_profile.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx])
        per_image_multiply_scale[image_obj] = image_obj.scale_by_radial_profile(**args.scale_by_radial_profile.params)

    # calculate average intensity 
    elif args.scale_by_overall_intensity.status:
        if not reference_obj:
            reference_idx = args.scale_by_overall_intensity.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx])
        per_image_multiply_scale[image_obj] = image_obj.scale_by_overall_intensity(**args.scale_by_overall_intensity.params)

    # calculate average intensity 
    elif args.scale_by_average_intensity.status:
        if not reference_obj:
            reference_idx = args.scale_by_average_intensity.params.get("reference_idx") or 0
            reference_obj = scripts.manager.ImageAgent()
            reference_obj.loadImage(import_data.get(args.read_dname)[reference_idx])
        per_image_multiply_scale[image_obj] = image_obj.scale_by_average_intensity(**args.scale_by_average_intensity.params)

    image_obj = None
reference_obj = None

if comm_rank != 0:
    md=mpidata()
    md.addarray('per_image_multiply_scale', per_image_multiply_scale) 
    md.small.rank = comm_rank 
    md.send()
    md = None
else:
    for nrank in range(comm_size-1):
        md=mpidata()
        md.recv()
        per_image_multiply_scale += md.per_image_multiply_scale 
        recvRank = md.small.rank
        md = None
        print '#### Rank 0 received file from Rank ' + str(recvRank).rjust(2) 
comm.Barrier()

if comm_rank==0:
    scale_dir = os.path.dirname(file_name)
    scale_file = os.path.join(scale_dir,"%s.npy"%args.save_dname)
    np.save(scale_file, per_image_multiply_scale)

    data_save = {args.save_dname:scale_file}
    history = scripts.fsystem.H5manager.reader(args.fsave, "history")
    from sys import argv
    if not history:
        history = [" ".join(argv)]
    else:
        history.append(" ".join(argv))
    data_save["history"] = history

    scripts.fsystem.PVmanager.modify(data_save, args.fsave)
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
from scripts.mpidata import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname",help="input files", default="./data.diffuse", type=str) 
parser.add_argument("-fsave","--fsave",help="save new files", default=None, type=str) 
parser.add_argument("-save_dir","--save_dir",help="save folder", default="./processed_data", type=str) 

parser.add_argument("-read_dname","--read_dname",help="save folder", default="image_file", type=str) 
parser.add_argument("-save_dname","--save_dname",help="save folder", default="processed_image", type=str) 

parser.add_argument("--apply_detector_mask",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--subtract_background",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--remove_bad_pixels",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--parallax_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--polarization_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--solid_angle_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--detector_absorption_correction",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--remove_bragg_peaks",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--calculate_radial_profile",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--calculate_overall_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--calculate_average_intensity",help="cases: None, [], [1,2,3]",default=None,nargs="*") 
args = parser.parse_args()

#### create folders
args.fsave = args.fsave or args.fname
if comm_rank == 0:
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)


#### load data from file
import_data = utils.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname, "backg_file", \
                                                "gxparms_file","detector_mask_file"])
num_images = len(import_data.get("image_file"))


#### get process params
args.apply_detector_mask = scripts.utils.get_process_params(args.apply_detector_mask)
args.remove_bad_pixels = scripts.utils.get_process_params(args.remove_bad_pixels)
args.subtract_background = scripts.utils.get_process_params(args.subtract_background)
args.parallax_correction = scripts.utils.get_process_params(args.parallax_correction)
args.polarization_correction = scripts.utils.get_process_params(args.polarization_correction)
args.solid_angle_correction = scripts.utils.get_process_params(args.solid_angle_correction)
args.detector_absorption_correction = scripts.utils.get_process_params(args.detector_absorption_correction)
args.remove_bragg_peaks = scripts.utils.get_process_params(args.remove_bragg_peaks)
args.calculate_radial_profile = scripts.utils.get_process_params(args.calculate_radial_profile)
args.calculate_overall_intensity = scripts.utils.get_process_params(args.calculate_overall_intensity)
args.calculate_average_intensity = scripts.utils.get_process_params(args.calculate_average_intensity)


#### process the pipeline
for image_idx in range(num_images):
    if idx%comm_size != comm_size:
        continue

    image_obj = scripts.manager.ImageAgent()
    image_obj.loadImage(import_data.get("image_file")[image_idx])
    image_obj.loadImage(import_data.get("gxparms_file"))
    image_obj.__dict__.update(import_data)

    if args.subtract_background.status:
        backg_obj = scripts.manager.ImageAgent()
        backg_obj.loadImage(import_data.get("backg_file")[image_idx])
        backg_obj.loadImage(import_data.get("gxparms_file"))
        backg_obj.__dict__.update(import_data)

    if args.apply_detector_mask.status:
        image_obj.apply_detector_mask(**args.apply_detector_mask.params)
        if args.subtract_background.status:
            backg_obj.apply_detector_mask(**args.apply_detector_mask.params)

    # remove bad pixels
    if args.remove_bad_pixels.status:
        image_obj.remove_bad_pixels(**args.remove_bad_pixels.params)
        if args.subtract_background.status:
            backg_obj.remove_bad_pixels(**args.remove_bad_pixels.params)

    # subtract background
    if args.subtract_background.status:
        image_obj.subtract_background(backg=backg_obj,**args.subtract_background.params)

    # parallax correction 
    if args.parallax_correction.status:
        image_obj.parallax_correction(**args.parallax_correction.params)

    # polarization correction 
    if args.polarization_correction.status:
        image_obj.polarization_correction(**args.polarization_correction.params)

    # solid angle correction 
    if args.solid_angle_correction.status:
        image_obj.solid_angle_correction(**args.solid_angle_correction.params)

    # detector absorption correction
    if args.detector_absorption_correction.status:
        image_obj.detector_absorption_correction(**args.detector_absorption_correction.params)

    # remove bragg peaks
    if args.remove_bragg_peak.status:
        image_obj.remove_bragg_peak(**args.remove_bragg_peak.params)

    # calculate radial profile
    if args.calculate_radial_profile.status:
        image_obj.calculate_radial_profile(**args.calculate_radial_profile.params)

    # calculate overall intensity 
    if args.calculate_overall_intensity.status:
        image_obj.calculate_overall_intensity(**args.calculate_overall_intensity.params)

    # calculate average intensity 
    if args.calculate_average_intensity.status:
        image_obj.calculate_average_intensity(**args.calculate_average_intensity.params)

    image_obj.phi = image_obj.angleStep * 1.0 * image_idx
    fsave_image = os.path.join(args.save_dir,"%s_%.5d.h5"%(args.save_dname,image_idx))
    scripts.fsystem.PVmanager.modify(fsave_image,image_obj.__dict__)
    print("#### Rank %3d/%3d has processed image %5d"%(comm_rank,comm_size,idx))



if comm_rank==0:
    processed_file = [os.path.realpath(os.path.join(args.data_dir,"%s_%.5d.h5"%(args.save_dname,image_idx))) for image_idx in range(num_images)]
    data_save = {args.save_dname : processed_file}
    history = scripts.fsystem.H5manager.reader(args.fsave, "history")
    
    from sys import argv
    if not history:
        history = [" ".join(argv)]
    else:
        history.append(" ".join(argv))
    data_save["history"] = history

    scripts.fsystem.PVmanager.modify(data_save, args.fsave)


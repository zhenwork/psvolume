import os,sys
import numpy as np
from numba import jit
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

from core.mpidata import *
import workflow.manager 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--from_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                    default=None,nargs="*") 
parser.add_argument("--into_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                    default=None,nargs="*") 

parser.add_argument("--apply_detector_mask",help="cases: None, [], [1,2,3]",default=None,type=str) 
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


## load data for arguments
ipm = workflow.manager.ImageProcessMaster(args=args.__dict__)
ipm.prepare_required_params()

for image_idx in range(ipm.num_images):
    if image_idx % comm_size != comm_rank:
        continue

    image_obj = ipm.request_image_data(idx=image_idx)
    backg_obj = ipm.request_backg_data(idx=image_idx)

    # apply detector mask
    if ipm.apply_detector_mask.status:
        image_obj.apply_detector_mask(**ipm.apply_detector_mask.params)
        backg_obj.apply_detector_mask(**ipm.apply_detector_mask.params)

    # remove bad pixels
    if ipm.remove_bad_pixels.status:
        image_obj.remove_bad_pixels(**ipm.remove_bad_pixels.params)
        backg_obj.remove_bad_pixels(**ipm.remove_bad_pixels.params)

    # subtract background
    if ipm.subtract_background.status:
        image_obj.subtract_background(backg=backg_obj,**ipm.subtract_background.params)

    # parallax correction 
    if ipm.parallax_correction.status:
        image_obj.parallax_correction(**ipm.parallax_correction.params)

    # polarization correction 
    if ipm.polarization_correction.status:
        image_obj.polarization_correction(**ipm.polarization_correction.params)

    # solid angle correction 
    if ipm.solid_angle_correction.status:
        image_obj.solid_angle_correction(**ipm.solid_angle_correction.params)

    # detector absorption correction
    if ipm.detector_absorption_correction.status:
        image_obj.detector_absorption_correction(**ipm.detector_absorption_correction.params)

    # remove bragg peaks
    if ipm.remove_bragg_peak.status:
        image_obj.remove_bragg_peak(**ipm.remove_bragg_peak.params)

    # calculate radial profile
    if ipm.calculate_radial_profile.status:
        image_obj.calculate_radial_profile(**ipm.calculate_radial_profile.params)

    # calculate overall intensity 
    if ipm.calculate_overall_intensity.status:
        image_obj.calculate_overall_intensity(**ipm.calculate_overall_intensity.params)

    # calculate average intensity 
    if ipm.calculate_average_intensity.status:
        image_obj.calculate_average_intensity(**ipm.calculate_average_intensity.params)

    ipm.update_result(idx=image_idx, image_data=image_obj.changed_params())


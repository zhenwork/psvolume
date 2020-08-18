import os,sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="psvolume data",default="./diffuse.data",type=str) 
parser.add_argument("--image", help="data/image",default="data/image",type=str) 
parser.add_argument("--detector_mask", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--remove_bad_pixels", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--polarization_correction", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--solid_angle_correction", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--detector_absorption_correction", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--parabollox_correction", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--bragg_peak_mask", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--subtract_background", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--radial_profile", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--overall_intensity", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
parser.add_argument("--average_intensity", help="cases: None, [], [1,2,3]",default=None,nargs="*") 
args = parser.parse_args()



## detector_mask
if 

"""
1. Merge without background subtraction 
2. Scale based on overall intensity of crystal diffraction
"""

import os,sys
import numpy as np
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import scripts.fsystem
import scripts.utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-image_file","--image_file", help="input files", default=None, nargs="*")
parser.add_argument("-backg_file","--backg_file", help="backg files", default=None, nargs="*")
parser.add_argument("-fsave","--fsave",help="save file", default="./data.diffuse", type=str) 
parser.add_argument("-pdb_file","--pdb_file", help="expand mask?",default=None,type=str) 
parser.add_argument("-gxparms_file","--gxparms_file", help="xds file",default=None, type=str)
parser.add_argument("-dials_expt_file","--dials_expt_file",help="dials file",default=None, type=str)
parser.add_argument("-dials_report_file","--dials_report_file",help="dials file",default=None, type=str)
parser.add_argument("-detector_mask_file","--detector_mask_file", help="numpy array",default=None,type=str) 
args = parser.parse_args()


#### create folder
folder = os.path.dirname(os.path.realpath(args.fsave))
if not os.path.isdir(folder):
    os.makedirs(folder)


#### save the diffraction image
data_save = {}
if args.image_file:
    data_save["image_file"] = args.image_file 
if args.backg_file:
    data_save["backg_file"] = args.backg_file 
    assert len(args.image_file)==len(args.backg_file)


#### process the pipeline
if args.gxparms_file:
    data_save["gxparms_file"] = gxparms_file
if args.dials_expt_file:
    data_save["dials_expt_file"] = dials_expt_file
if args.dials_report_file:
    data_save["dials_report_file"] = dials_report_file
if args.detector_mask_file:
    data_save["detector_mask_file"] = detector_mask_file
if args.pdb_file:
    data_save["pdb_file"] = pdb_file

if os.path.isfile(args.fsave):
    history = scripts.fsystem.H5manager.reader(args.fsave, "history")
    from sys import argv
    if not history:
        history = [" ".join(argv)]
    else:
        history.append(" ".join(argv))
    data_save["history"] = history

scripts.fsystem.PVmanager.modify(data_save, args.fsave) 
print ("Import Done")

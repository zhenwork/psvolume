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
import scripts.dataExtract 
import scripts.fileManager 
import scripts.utils

PsvolumeManager = scripts.fileManager.PsvolumeManager()
FileSystem = scripts.fileManager.FileSystem()
H5Manager = scripts.fileManager.H5FileManager()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-image_file","--image_file", help="input files", default=None, nargs="*")
parser.add_argument("-backg_file","--backg_file", help="backg files", default=None, nargs="*")
parser.add_argument("-fsave","--fsave",help="save file", default=None, type=str)

parser.add_argument("-gxparms","--gxparms", help="xds file",default=None, type=str)
parser.add_argument("-dials_expt","--dials_expt",help="dials file",default=None, type=str)
parser.add_argument("-dials_report","--dials_report",help="dials file",default=None, type=str)

parser.add_argument("-exp_mask","--exp_mask", help="expand mask?",default=0,type=int) 
args = parser.parse_args()


#### create folder
folder = os.path.dirname(args.fsave)
if not os.path.isdir(folder):
    os.makedirs(folder)


#### save the diffraction image
data_save = {}
if args.image_file:
    data_save["image_file"] = args.image_file
if args.backg_file:
    data_save["backg_file"] = args.backg_file 
    assert len(args.image_file)==len(backg_file)


#### process the pipeline
if args.gxparms:
    scripts.utils.dict_merge(data_save, scripts.dataExtract.xds2psvm(args.gxparms))
if args.dials_expt:
    scripts.utils.dict_merge(data_save, scripts.dataExtract.expt2psvm(args.dials_expt))
if args.dials_report:
    scripts.utils.dict_merge(data_save, scripts.dataExtract.dials_report(args.dials_report))
if args.exp_mask:
    scripts.utils.dict_merge(data_save, np.load(args.exp_mask))

if os.path.isfile(args.fsave):
    history = H5Manager.h5reader(args.fsave, "history")
    from sys import argv
    if not history:
        history = [" ".join(argv)]
    else:
        history.append(" ".join(argv))
    data_save["history"] = history

PsvolumeManager.psvm2h5py(data_save, args.fsave) 
print ("Import Done")

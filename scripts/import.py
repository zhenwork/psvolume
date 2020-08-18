# python --from image=file1.cbf,file2.cbf,file3.cbf dials=file4 --to diffuse.data

import os,sys
import core.filesystem
import diffuse.datafile

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_file", help="cbf",default=[],nargs="*") 
parser.add_argument("--backg_file", help="cbf",default=[],nargs="*") 
parser.add_argument("--into", help="diffuse.data",default="./diffuse.data",type=str)
parser.add_argument("--dials_report_file", help="file",default=None,type=str) 
parser.add_argument("--dials_expt_file", help="file",default=None,type=str) 
parser.add_argument("--gxparms_file", help="file",default=None,type=str) 
args = parser.parse_args() 


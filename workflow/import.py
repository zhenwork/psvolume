# python --from image=file1.cbf,file2.cbf,file3.cbf dials=file4 --to diffuse.data

import os,sys
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
    
import workflow.manager

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_file", help="cbf",default=None,nargs="*") 
parser.add_argument("--backg_file", help="cbf",default=None,nargs="*") 
parser.add_argument("--dials_report_file", help="file",default=None,type=str) 
parser.add_argument("--dials_expt_file", help="file",default=None,type=str) 
parser.add_argument("--gxparms_file", help="file",default=None,type=str) 
parser.add_argument("--from_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                    default=None,nargs="*") 
parser.add_argument("--into_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                    default="./diffuse.dat",nargs="*") 
args = parser.parse_args() 


# analyze arguments 
state = diffuse.master.ImportMaster(args=args.__dict__)
state.prepare_required_params()
state.start_process()
state.update_result()
state.free_memory()
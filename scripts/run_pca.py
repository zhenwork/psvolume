import h5py
import os,sys
import numpy as np
from scipy import signal  
from sklearn.decomposition import PCA
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
from scripts.mpidata import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname",help="input files", default="./data.diffuse", type=str) 
parser.add_argument("-fsave","--fsave",help="save new files", default=None, type=str) 
parser.add_argument("-read_dname","--read_dname",help="save new files", default="processed_image", type=str) 
parser.add_argument("-save_dname","--save_dname",help="save new files", default="pca_image", type=str) 
parser.add_argument("-scaling","--scaling",help="scaling factors", default="per_image_multiply_scale", type=str) 
args = parser.parse_args() 

#### make files
args.fsave = args.fsave or args.fname
if comm_rank == 0:
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    if args.fsave != args.fname:
        scripts.fsystem.Fsystem.copyfile(src=args.fname,dst=args.fsave)

#### load data from file
import_data = utils.fsystem.PVmanager.reader(args.fname,keep_keys=[args.read_dname,args.scaling])
num_images = len(import_data.get(args.read_dname))
per_image_multiply_scale = np.load(import_data.get(args.scaling))


def process_one_single_pca(radial_profiles):
    pca = PCA()
    pca.fit(radial_profiles)
    fitted = pca.transform(radial_profiles)

    tmp_data = fitted.copy()
    tmp_data[:,1:] = 0 
    backg_profiles = tmp_data.dot(pca.components_)
    tmp_data = None

    for idx in range(len(files)):
        firstpos = np.argmax(radial_profiles[idx]>0)
        backg[idx,:firstpos] = 0
        backg[idx,firstpos:(firstpos+5)] = backg_profiles[idx,(firstpos+5)]
    return 

def process_pca(radial_profiles, num_pca=3):
    return 

## calculate the radial profile
if comm_rank == 0:
    radial_profiles = [] 
    for image_idx in range(num_images):
        fname = import_data.get(args.read_dname)[image_idx]
        with h5py.File(fname, "r") as f:
            profile = f["radprofile"].value 
        radial_profiles.append(profile * per_image_multiply_scale[image_idx]) 
    radial_profiles = np.array(radial_profiles) 

    
    
print"stage 1 done"

comm.Barrier()

## smooth the profile





if show_plot and comm_rank==0:
    # first 4 PCA components
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))

    plt.subplot(1,3,1)
    for profile in radpr:
        plt.plot(profile[:1500])
    plt.title("raw profile",fontsize=20)


    plt.subplot(1,3,2)
    for profile in backg:
        plt.plot(profile[:1500])
    plt.title("PCA background",fontsize=20)


    plt.subplot(1,3,3)
    for idx in range(len(files)):
        plt.plot(radpr[idx,:1500] - backg[idx,:1500])
    plt.title("residue profile",fontsize=20)

    plt.tight_layout()
    plt.show()

    

###  
with h5py.File(files[0], "r") as f:
    nx, ny = f["image"].value.shape
    cx, cy = f["detectorCenter"].value

x = np.arange(nx) - cx
y = np.arange(ny) - cy
xaxis, yaxis = np.meshgrid(x, y, indexing="ij")

r = np.around(np.sqrt(xaxis**2 + yaxis**2)).astype(int)

for idx,fname in enumerate(files):
    if idx % comm_size != comm_rank:
        continue 
        
    with h5py.File(fname, "r") as f:
        image = f["image"].value * f["scale"].value
        mask  = f["mask"].value
    
    mask *= (image>0) 
    
    sub = backg[idx][r]
    image -= sub  

    index = np.where(mask==0)
    image[index] = 0
    mask[index] = 0 
    
    with h5py.File(fname, "r+") as f:
        f["image"][...] = image 
        f["mask"][...]  = mask 
        f["scale"][...] = 1.0 
        
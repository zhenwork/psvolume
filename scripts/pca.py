import h5py
import os,sys
import numpy as np
from scipy import signal
import glob
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


from sys import argv
path_h5py = argv[1]


### load the radial background
path_h5py = path_h5py.replace("#####","*")

files = sorted(glob.glob(path_h5py))
print len(files)
print files[0]
print files[1]

## calculate the radial profile
radpr = []
scale = []
for fname in files:
    with h5py.File(fname, "r") as f:
        sl = f["scale"].value
        pr = f["radprofile"].value
        radpr.append(pr * sl)
        scale.append(sl)
radpr = np.array(radpr)

## smooth the profile
smooth = []
for profile in radpr:
    smooth.append(signal.convolve(profile,np.ones(3),"same")/3.)
radpr = np.array(smooth)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(radpr)
fitted = pca.transform(radpr)

tmp_data = fitted.copy()
tmp_data[:,3:] = 0 
backg = tmp_data.dot(pca.components_)
tmp_data = None

for idx in range(len(files)):
    firstpos = np.argmax(radpr[idx]>0)
    backg[idx,:firstpos] = 0
    backg[idx,firstpos:(firstpos+5)] = backg[idx,(firstpos+5)]
    
print"stage 1 done"

show_plot = False
if show_plot:
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
    if idx % comm_rank != 0:
        continue 
        
    with h5py.File(fname, "r") as f:
        image = f["image"].value * f["scale"].value
        mask  = f["mask"].value
    
    mask *= (image>0) 
    
    backg = radpr[idx][r]
    image -= backg 

    index = np.where(mask==0)
    image[index] = 0
    mask[index] = 0 
    
    with h5py.File(fname, "r+") as f:
        f["image"][...] = image 
        f["mask"][...]  = mask 
        f["scale"][...] = 1.0 
        
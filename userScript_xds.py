"""
start point of the merging process
 
All unit should be Angstrom (A): wavelength, crystal lattice 
"""

import numpy as np 
from imageMergeClient import *
import h5py
import cbf
import os


## read the xds indexing file
user_get_xds(filename)


# How to read the idx image, return a 2d matrix
def user_get_image(idx):
	fname = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'+'/wtich_274_10_1_'+str(idx+1).zfill(5)+'.cbf'
	content = cbf.read(fname)
	image = np.array(content.data).astype(float)
	return image




# How to get the idx center, return a turple (cx,cy)
def user_get_center(idx):
	return (1265.838623, 1228.777588)  




# How to get the idx quaternion, return a quaternion (q1,q2,q3,q4)
def user_get_orientation(idx):
	return (np.cos(idx*0.1*np.pi/2./180.), 0., np.sin(idx*0.1*np.pi/2./180.), 0.)




# How to get the scale factor, return a number (if not using, return None)
def user_get_scalingFactor(idx):
	#return 1.
	f = h5py.File('/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/data-ana/scalesMike.h5')
	scale = f[f.keys()[0]].value
	f.close()
	return scale[idx]
	return None



# How to define a users mask 
def user_get_mask():
	path = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'
	fname = os.path.join(path, 'wtich_274_10_1_'+str(1).zfill(5)+'.cbf')
	content = cbf.read(fname)
	data = np.array(content.data).astype(float)
	mask = np.ones(data.shape).astype(int)
	index = np.where(data > 100000)
	mask[index] = 0
	mask[1260:1300,1235:2463] = 0
	radius = make_radius(mask.shape, center=(1265.33488372, 1228.00813953))
	index = np.where(radius<25)
	mask[index] = 0
	return mask



## Tools:
def user_get_xds(filename):
	f = open(filename)
	content = f.readlines()
	f.close()

	Geo = {}
	Geo['pixelSize'] = float(content[7].split()[3])
	Geo['detDistance'] = float(content[8].split()[2])
	Geo['polarization'] = 'y' if float(content[1].split()[3])>0.9 else 'x'
	Geo['wavelength'] = float(content[2].split()[0])

	## calculate the invAmat matrix
	invAmat = np.zeros((3,3));
	for i in range(4,7):
	    for j in range(3):
	        invAmat[i-4,j] = float(content[i].split()[j])
	if invAmat is not None:
	    invAmat[1,:] = -invAmat[1,:].copy()
	    tmp = invAmat[:,0].copy()
	    invAmat[:,0] = invAmat[:,1].copy()
	    invAmat[:,1] = tmp.copy()

	## calculate B matrix from lattice constants
	Bmat = np.zeros((3,3));
	a = float(content[3].split()[1]); 
	b = float(content[3].split()[2]); 
	c = float(content[3].split()[3]); 
	alpha = float(content[3].split()[4]); 
	beta  = float(content[3].split()[5]); 
	gamma = float(content[3].split()[6]); 
	(vecx, vecy, vecz, recH, recK, recL) = Lattice2vector(a,b,c,ag1,ag2,ag3)
	Bmat = np.array([recH, recK, recL]).T 
	invBmat = np.linalg.inv(Bmat)
	return [Bmat, invBmat]
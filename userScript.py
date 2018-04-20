import numpy as np 
from imageMergeClient import *
import h5py
import cbf
import os

## basic parameters
Geo = {}
Geo['pixelSize'] = 172.0
Geo['detDistance'] = 200147.4
Geo['beamStop'] = 50
Geo['polarization'] = 'y'
Geo['wavelength'] = 0.082653

## rotation matrix of the first image
rot_two = np.array([[-0.2438,  0.9655,  -0.0919],
                    [-0.8608, -0.2591,  -0.4381],
                    [-0.4468, -0.0277,   0.8942]])


## crystal lattice
sima = np.array([ 0.007369 ,   0.017496 ,   -0.000000])
simb = np.array([-0.000000 ,   0.000000 ,    0.017263])
simc = np.array([ 0.015730 ,   0.000000,     0.000000])


# How to read the idx image, return a 2d matrix
def user_get_image(idx):
	fname = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'+'/wtich_274_10_1_'+str(idx+1).zfill(5)+'.cbf'
	content = cbf.read(fname)
	image = np.array(content.data).astype(float)
	return image


# How to get the idx center, return a turple (cx,cy)
def user_get_center(idx):
	return (1265.33488372, 1228.00813953)


# How to get the idx quaternion, return a quaternion (q1,q2,q3,q4)
def user_get_orientation(idx):
	return (np.cos(idx*0.1*np.pi/2./180.), 0., np.sin(idx*0.1*np.pi/2./180.), 0.)


# How to get the scale factor, return a number (if not using, return None)
def user_get_scalingFactor(idx):
	f = h5py.File('/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/data-ana/scalesMike.h5')
	scale = f[f.keys()[0]].value
	scale = 1./scale
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



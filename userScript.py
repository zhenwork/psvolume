"""
start point of the merging process
 
All unit should be Angstrom (A): wavelength, crystal lattice 

If you have input the xds indexing file, you don't have to input basic parameters here
"""

import numpy as np 
from imageMergeClient import *
import h5py
import cbf
import os

# How to read the idx image, return a 2d matrix
def user_get_image(idx):
	#fname = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'+'/wtich_274_10_1_'+str(idx+1).zfill(5)+'.cbf'
	#fname = '/reg/data/ana13/xpp/xppk7915/res/vdbedem/ICHwt_cryo_2'+'/ICHwt_cryo_2_1_'+str(idx+1).zfill(5)+'.cbf'
	#fname = '/reg/data/ana04/users/zhensu/xpptut/volume/ICH_wt_cds4/crystal/rawcbf/ICH_wt_cds4_1_'+str(idx+1).zfill(5)+'.cbf'   ## blank image
	fname = '/reg/data/ana04/users/zhensu/xpptut/volume/ICH_wt_cds4/crystal/rawcbf/ICH_wt_cds4_blk_1_'+str(idx+1).zfill(5)+'.cbf'
	content = cbf.read(fname)
	image = np.array(content.data).astype(float)
	return image


# How to get the idx quaternion, return a quaternion (q1,q2,q3,q4)
def user_get_orientation(idx):
	return (np.cos(idx*0.15*np.pi/2./180.), 0., np.sin(idx*0.15*np.pi/2./180.), 0.)


# How to get the scale factor, return a number (if not using, return None)
def user_get_scalingFactor(idx):
	return 1.
	f = h5py.File('/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/data-ana/scalesMike.h5')
	scale = f[f.keys()[0]].value
	f.close()
	return scale[idx]


# How to define a users mask
def user_get_mask(Geo):
	data = user_get_image(0)
	mask = np.ones(data.shape).astype(int)
	index = np.where(data > 10000)
	mask[index] = 0
	mask[1260:1300,1235:2463] = 0
	radius = make_radius(mask.shape, center=Geo['center'])
	index = np.where(radius<25)
	mask[index] = 0
	return mask



## basic parameters
Geo = {}
Geo['pixelSize'] = 172.0
Geo['detDistance'] = 200147.4
Geo['polarization'] = 'y'
Geo['wavelength'] = 0.82653
Geo['center'] = (1265.33488372, 1228.00813953)


## inverse U matrix is usually unknown in cctbx result
invAmat = None
invUmat = np.array([[-0.2438,  0.9655,  -0.0919],
					[-0.8608, -0.2591,  -0.4381],
					[-0.4468, -0.0277,   0.8942]])


## B matrix (crystal lattice) in the unit of A-1
Bmat = np.array([[ 0.007369 ,   0.017496 ,   -0.000000],
				 [ 0.000000 ,   0.000000 ,    0.017263],
				 [ 0.015730 ,   0.000000,     0.000000]]).T
sima = Bmat[:,0].copy()
simb = Bmat[:,1].copy()
simc = Bmat[:,2].copy()
lsima = np.sqrt(np.sum(sima**2))
lsimb = np.sqrt(np.sum(simb**2))
lsimc = np.sqrt(np.sum(simc**2))
Kac = np.arccos(np.dot(sima, simc)/lsima/lsimc)
Kbc = np.arccos(np.dot(simb, simc)/lsimb/lsimc)
Kab = np.arccos(np.dot(sima, simb)/lsima/lsimb)
sima = lsima * np.array([np.sin(Kac), 0., np.cos(Kac)])
simb = lsimb * np.array([0., 1., 0.])
simc = lsimc * np.array([0., 0., 1.])
Bmat = np.array([sima,simb,simc]).T
invBmat = np.linalg.inv(Bmat)
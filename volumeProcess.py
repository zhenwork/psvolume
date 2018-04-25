from numba import jit
import os, sys
import numpy as np  
import h5py
from function_v02 import *
from scipy.signal import medfilt
import scipy.ndimage

num_samp=1000
path_data = '0005/result.h5'
tag='-scale-test-0005'
thr = (-100, 1000)

# making detector and quaternion
def perpixel(lam=3.0, wavelength=1.3, pixsize=89.0, detd=53300.0):
	return lam * float(pixsize) / float(wavelength) / float(detd)

rot2 = np.array([[-0.2438,  0.9655,  -0.0919],
				 [-0.8608, -0.2591,  -0.4381],
				 [-0.4468, -0.0277,   0.8942]])

sima = np.array([ 0.007369 ,   0.017496 ,   -0.000000])
simb = np.array([-0.000000 ,   0.000000 ,	0.017263])
simc = np.array([ 0.015730 ,   0.000000,	 0.000000])
lsima = np.sqrt(np.sum(sima**2))
lsimb = np.sqrt(np.sum(simb**2))
lsimc = np.sqrt(np.sum(simc**2))
Kac = np.arccos(np.dot(sima, simc)/lsima/lsimc)
Kbc = np.arccos(np.dot(simb, simc)/lsimb/lsimc)
Kab = np.arccos(np.dot(sima, simb)/lsima/lsimb)

lscale = perpixel(lam=1.6, wavelength=0.82653, pixsize=172.0, detd=200147.4)
print 'sima, simb, simc, angle = ', lsima, lsimb, lsimc, Kab, Kac, Kbc
print 'sima, simb, simc = ', lsima/lscale, lsimb/lscale, lsimc/lscale

sima = lsima/lscale * np.array([np.sin(Kac), 0., np.cos(Kac)])
simb = lsimb/lscale * np.array([0., 1., 0.])
simc = lsimc/lscale * np.array([0., 0., 1.])

imat_inv = np.linalg.inv(np.transpose(np.array([sima,simb,simc])))

astar = sima.copy()
bstar = simb.copy()
cstar = simc.copy()

print 'astar = ', astar
print 'bstar = ', bstar
print 'cstar = ', cstar

print 'reading data ... '
f = h5py.File(path_data,'r')
data = np.array(f[f.keys()[0]]).astype(float)
f.close()

print '### extracting data ...'
hkl = hklExtraction(data, astar, bstar, cstar, icen=60, ithreshold=(-100,1000), iradius=(32, 600), outer=(5,5,5), inner=None )
#hkl = data.copy()
print 'hkl max/min: ', np.amin(hkl), np.amax(hkl), hkl.shape

print '### symmetrizing data ...'
symhkl = lauesym(hkl, ithreshold=thr)
print 'symhkl max/min: ', np.amin(symhkl), np.amax(symhkl), symhkl.shape

la = lens(astar)
lb = lens(bstar)
lc = lens(cstar)

astar = la/lb*np.array([1., 0, 0])
bstar = np.array([0, 1., 0])
cstar = lc/lb*np.array([np.cos(Kac), 0, np.sin(Kac)])

print '### distribuction calculation ... '
[backg, radius] = distri(symhkl, astar, bstar, cstar, ithreshold=thr, iscale=1, iwindow=5)
print 'backg max/min: ', np.amin(backg), np.amax(backg), radius.shape

#backg = meanf(backg, _scale=5, clim=(0.1, 50))

symhkl_sub = symhkl.copy()
[nnxx, nnyy, nnzz] = symhkl.shape
for i in range(nnxx):
	for j in range(nnyy):
		for k in range(nnzz):
			intr = radius[i,j,k]
			symhkl_sub[i,j,k] -= backg[intr]

print 'aniso max/min: ', np.amin(symhkl_sub), np.amax(symhkl_sub)


print '### volumelist calculation ... '
volumelist = hkl2volume(symhkl_sub, astar, bstar, cstar, ithreshold=thr)

print '### saving data ... '
if __name__ == "__main__":  
	
	print ('start saving files... ')  
	f = h5py.File('result'+tag,'w')
	datawr = f.create_dataset('hkl', hkl.shape)
	datawr[...] = hkl
	datawr = f.create_dataset('distri', backg.shape)
	datawr[...] = backg
	datawr = f.create_dataset('volume', volumelist.shape)
	datawr[...] = volumelist
	datawr = f.create_dataset('symhkl', symhkl.shape)
	datawr[...] = symhkl
	datawr = f.create_dataset('anisohkl', symhkl_sub.shape)
	datawr[...] = symhkl_sub
	f.close() 
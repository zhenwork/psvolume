import os, sys
import numpy as np
from volumeTools import *
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-thrmin","--thrmin", help="min value", default=-100, type=int)
parser.add_argument("-thrmax","--thrmax", help="max value", default=1000, type=int)
parser.add_argument("-name","--name", help="name", default="volume", type=str)
args = parser.parse_args()
thr = (args.thrmin, args.thrmax)

# read volume geometry
Umatrix = zf.h5reader(args.i, 'Umatrix')
astar = Umatrix[0,:].copy()
bstar = Umatrix[1,:].copy()
cstar = Umatrix[2,:].copy()
print '### astar = ', astar
print '### bstar = ', bstar
print '### cstar = ', cstar
print '### threshold: ', thr

print '### reading volume ... '
rawData = zf.h5reader(args.i, 'intens')
print '### max/min: ', np.amin(rawData), np.amax(rawData)

print '### symmetrizing data ...'
symData = lauesym(rawData, ithreshold=thr)
print '### max/min: ', np.amin(symData), np.amax(symData)

print '### distribuction calculation ... '
[backg, radius] = distri(symhkl, astar, bstar, cstar, ithreshold=thr, iscale=1, iwindow=5)
print '### bgd max/min: ', np.amin(backg), np.amax(backg)

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
import os, sys
import numpy as np
from volumeTools import *
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-thrmin","--thrmin", help="min value", default=-100, type=int)
parser.add_argument("-thrmax","--thrmax", help="max value", default=1000, type=int)
parser.add_argument("-name","--name", help="name", default=".", type=str)
args = parser.parse_args()
thr = (args.thrmin, args.thrmax)
zf = iFile()

# read volume geometry
Umatrix = zf.h5reader(args.i, 'Umatrix')
astar = Umatrix[:,0].copy()
bstar = Umatrix[:,1].copy()
cstar = Umatrix[:,2].copy()
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

print '### background calculation ... '
[backg, radius] = distri(symData, astar, bstar, cstar, ithreshold=thr, iscale=2, iwindow=4)
print '### bgd max/min: ', np.amin(backg), np.amax(backg)

#backg = meanf(backg, _scale=5, clim=(0.1, 50))

print '### subtracting background ... '
subData = symData.copy()
[nnxx, nnyy, nnzz] = symData.shape
for i in range(nnxx):
	for j in range(nnyy):
		for k in range(nnzz):
			intr = radius[i,j,k]
			subData[i,j,k] -= backg[intr]
print '### max/min: ', np.amin(subData), np.amax(subData)


print ('### start saving files... ')  
zf.h5modify(args.i, name+'symData', symData)
zf.h5modify(args.i, name+'anisoData', subData)
zf.h5modify(args.i, name+'backg', backg)
import os, sys
import numpy as np
from volumeTools import *
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-thrmin","--thrmin", help="min value", default=-100, type=int)
parser.add_argument("-thrmax","--thrmax", help="max value", default=1000, type=int)
parser.add_argument("-name","--name", help="name", default="", type=str)
args = parser.parse_args()
thr = (args.thrmin, args.thrmax)
zf = iFile()

# read volume geometry
Smat = zf.h5reader(args.i, 'Smat')
astar = Smat[:,0].copy()
bstar = Smat[:,1].copy()
cstar = Smat[:,2].copy()
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
[backg, radius] = distri_alg2(symData, astar, bstar, cstar, ithreshold=thr, iscale=4, iwindow=5)
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

if np.amax( np.abs( Smat-np.eye(3) ) )>1e-3:
	print "### converting hkl to xyz coordinate"
	anisoData = hkl2volume(subData, astar, bstar, cstar, ithreshold=thr)
	print '### max/min: ', np.amin(anisoData), np.amax(anisoData)
else: 
	print "### This is already the best coordinate ... "
	anisoData = subData.copy()

print ('### start saving files... ')  
zf.h5modify(args.i, args.name+'symData', symData)
zf.h5modify(args.i, args.name+'subData', subData)
zf.h5modify(args.i, args.name+'anisoData', anisoData)
zf.h5modify(args.i, args.name+'backg', backg)
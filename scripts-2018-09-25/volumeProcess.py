import os, sys
import numpy as np
from volumeTools import *
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-thrmin","--thrmin", help="min value", default=-100, type=float)
parser.add_argument("-thrmax","--thrmax", help="max value", default=1000, type=float)
parser.add_argument("-name","--name", help="name", default="", type=str)
parser.add_argument("-sym","--sym", help="process", default="laue", type=str)  #symmetry can be "laue","inv"
parser.add_argument("-sub","--sub", help="process", default=1, type=int)
parser.add_argument("-hkl2xyz","--hkl2xyz", help="process", default=1, type=int)
parser.add_argument("-rescale","--rescale", help="if rescale=0.5, then voxel value will be smaller", default=1.0, type=float)
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
print '### symmetrize: ', args.sym
print '### background: ', bool(args.sub)
print '### hkl TO xyz: ', bool(args.hkl2xyz)
print "### rescale factor: ", args.rescale


### Only when we need one of the processes, we start to read the file
if (args.sym == "laue") or (args.sym == "inv") or (args.sub==1) or (args.hkl2xyz==1):
	print '### reading volume ... '
	rawData = zf.h5reader(args.i, 'intens') 
	index = np.where(rawData < -1000)
	rawData *= args.rescale
	rawData[index] = -1024
	print '### max/min: ', np.amin(rawData), np.amax(rawData)


### First step is to symmetrize the data
if args.sym=="laue":
	print '### Laue symmetrizing data ...'
	[symData, symCounter] = lauesym(rawData, ithreshold=thr)
	print '### max/min: ', np.amin(symData), np.amax(symData)
	zf.h5modify(args.i, args.name+'symData', symData)
	zf.h5modify(args.i, args.name+'symCounter', symCounter)
elif args.sym=="inv":
	print '### Inv symmetrizing data ...'
	[symData, symCounter] = inversesym(rawData, ithreshold=thr)
	print '### max/min: ', np.amin(symData), np.amax(symData)
	zf.h5modify(args.i, args.name+'InvsymData', symData)
	zf.h5modify(args.i, args.name+'InvsymCounter', symCounter)
else:
	symData = rawData.copy()


### Second step is to subtract the background
if args.sub==1:
	print '### background calculation ... '
	[backg, radius] = distri(symData, astar, bstar, cstar, ithreshold=thr, iscale=4, iwindow=5)
	print '### bgd max/min: ', np.amin(backg), np.amax(backg)

	#backg = meanf(backg, _scale=5, clim=(0.1, 50))
	print '### subtracting background ... '
	subData = symData.copy()
	rawSubData = rawData.copy()
	volumeBack = np.zeros(rawData.shape)
	[nnxx, nnyy, nnzz] = rawData.shape
	for i in range(nnxx):
		for j in range(nnyy):
			for k in range(nnzz):
				intr = radius[i,j,k]
				subData[i,j,k] -= backg[intr]
				rawSubData[i,j,k] = rawData[i,j,k] - backg[intr]
				volumeBack[i,j,k] = backg[intr]
	print '### max/min: ', np.amin(subData), np.amax(subData)
	zf.h5modify(args.i, args.name+'subData', subData)
	zf.h5modify(args.i, args.name+'backg', backg)
	zf.h5modify(args.i, args.name+'volumeBack', volumeBack)
else:
	subData = symData.copy()


### Last step is to convert the hkl basis to xyz basis
if args.hkl2xyz==1:
	if np.amax( np.abs( Smat-np.eye(3) ) )>1e-3:
		print "### converting hkl to xyz coordinate"
		anisoData = hkl2volume(subData, astar, bstar, cstar, ithreshold=thr)
		print '### max/min: ', np.amin(anisoData), np.amax(anisoData)
		anisoSubRaw = hkl2volume(rawSubData, astar, bstar, cstar, ithreshold=thr)
		threeDRaw = hkl2volume(rawData, astar, bstar, cstar, ithreshold=thr)
		symDataXYZ = hkl2volume(symData, astar, bstar, cstar, ithreshold=thr)
	else: 
		print "### This is already the best coordinate ... "
		anisoData = subData.copy()

	print ('### start saving files... ')
	zf.h5modify(args.i, args.name+'anisoData', anisoData)
	zf.h5modify(args.i, args.name+'anisoSubRaw', anisoSubRaw)
	zf.h5modify(args.i, args.name+'threeDRaw', threeDRaw)
	zf.h5modify(args.i, args.name+'symDataXYZ', symDataXYZ)


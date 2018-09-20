import os, sys
import numpy as np
import h5py
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="file 1", default=".", type=str)
parser.add_argument("-j","--j", help="file 2", default=".", type=str)
parser.add_argument("-tag","--tag", help="tag", default="", type=str)
parser.add_argument("-mode","--mode", help="mode can be 'shell' or 'ball'", default="shell", type=str)
parser.add_argument("-o","--o", help="save path", default=".", type=str)
parser.add_argument("-expand","--expand", help="expand", default=1.0, type=float)
parser.add_argument("-rmin","--rmin", help="min radius", default=0, type=int)
parser.add_argument("-rmax","--rmax", help="max radius", default=-1, type=int)
parser.add_argument("-i1","--i1", help="ilim[0]", default="-100", type=str)
parser.add_argument("-i2","--i2", help="ilim[1]", default="100", type=str)
parser.add_argument("-j1","--j1", help="jlim[0]", default="-100", type=str)
parser.add_argument("-j2","--j2", help="jlim[1]", default="100", type=str)
parser.add_argument("-iname","--iname", help="ilim[0]", default="anisoData", type=str)
parser.add_argument("-jname","--jname", help="ilim[1]", default="anisoData", type=str)
parser.add_argument("-count","--count", help="symCounter", default="", type=str)
parser.add_argument("-v","--v", help="verbose", default=0, type=int)
parser.add_argument("-bins","--bins", help="bins", default=-1, type=int)
args = parser.parse_args()
if args.i1 == "." or args.i2 == ".": ilim=None
else: ilim=(float(args.i1), float(args.i2))
if args.j1 == "." or args.j2 == ".": jlim=None
else: jlim=(float(args.j1), float(args.j2))




## correlation function
def cal_correlation(list1, list2):
	X = list1.copy()
	Y = list2.copy()
	X = X.astype(float)
	Y = Y.astype(float)
	X_ave = np.sum(X)/float(X.shape[0])
	Y_ave = np.sum(Y)/float(Y.shape[0])
	S1 = np.sum((X-X_ave)*(Y-Y_ave))
	S2 = np.sqrt(np.var(X)*float(X.shape[0])) * np.sqrt(np.var(Y)*float(Y.shape[0]))
	if S2 == 0: return 0.0
	return float(S1)/float(S2)

## 3d radius matrix
def make_3d_radius(nx, ny, nz, cx, cy, cz):
	x = np.arange(nx) - cx
	y = np.arange(ny) - cy
	z = np.arange(nz) - cz
	[xaxis, yaxis, zaxis] = np.meshgrid(x,y,z,indexing='ij')
	rMatrix = np.sqrt(xaxis**2+yaxis**2+zaxis**2)
	return rMatrix

## remove bad pixels
def data_remove(list1, list2, ilim=ilim, jlim=jlim):
	if ilim is not None:
		index = np.where(((list1>=ilim[0]) * (list1<=ilim[1]))==True)
		list1 = list1[index].copy()
		list2 = list2[index].copy()
		if len(list1)==0: return (list1, list2)
	if jlim is not None:
		index = np.where(((list2>=jlim[0]) * (list2<=jlim[1]))==True)
		list1 = list1[index].copy()
		list2 = list2[index].copy()
	return (list1, list2)


## correlation calculation
def q_Shell_Corr(data_i, data_j, center=(-1,-1,-1), rmin=0, rmax=-1, expand=1, ilim=None, jlim=None, mode='shell'):
	(nx,ny,nz) = data_i.shape;
	(cx,cy,cz) = center;
	if cx==-1: cx=(nx-1.)/2.
	if cy==-1: cy=(ny-1.)/2.
	if cz==-1: cz=(nz-1.)/2.
	rMatrix = expand*1.0*make_3d_radius(nx, ny, nz, cx, cy, cz);
	rMatrix = np.around(rMatrix).astype(int)
	if rmax==-1: rmax=int(np.amax(rMatrix))+1
	qCorr = np.zeros(rmax)
	for r in range(rmin, rmax):
		if mode=='ball': index = np.where(rMatrix<=r)
		else: index = np.where(rMatrix==r)
		list_i = data_i[index].ravel()
		list_j = data_j[index].ravel()
		(list_i, list_j) = data_remove(list_i, list_j, ilim=ilim, jlim=jlim)
		commLength = len(list_i)
		if len(list_i)<8: qCorr[r] = 0.0
		else: qCorr[r] = cal_correlation(list_i, list_j);
		print '### R:'+str(r).rjust(4)+'   NUM:'+str(commLength).rjust(6)+'   qCorr:  ' + str(round(qCorr[r],5)).ljust(8)
	return qCorr



def q_Shell_Corr_Bins(data_i, data_j, center=(-1,-1,-1), rmin=args.rmin, rmax=args.rmax, bins=args.bins, ilim=ilim, jlim=jlim):
	(nx,ny,nz) = data_i.shape;
	(cx,cy,cz) = center;
	if cx==-1: cx=(nx-1.)/2.
	if cy==-1: cy=(ny-1.)/2.
	if cz==-1: cz=(nz-1.)/2.
	rMatrix = 1.0*make_3d_radius(nx, ny, nz, cx, cy, cz);
	rMatrix = np.around(rMatrix).astype(int)
	if rmax==-1: rmax=int(np.amax(rMatrix))+1

	nTotal = len(np.where(rMatrix<=rmax)[0])
	nvoxel_per_bin = int(nTotal/float(bins))+1
	qCorr = np.zeros(bins)
	r1 = 0
	n = 0
	for r in range(rmin, rmax):
		index = np.where((rMatrix>r1)*(rMatrix<=r)==True)
		if len(index[0]) < nvoxel_per_bin:
			if r < rmax - 1:
				continue
		
		list_i = data_i[index].ravel()
		list_j = data_j[index].ravel()
		(list_i, list_j) = data_remove(list_i, list_j, ilim=ilim, jlim=jlim)
		commLength = len(list_i)
		if len(list_i)<8: qCorr[n] = 0.0
		else: qCorr[n] = cal_correlation(list_i, list_j);
		print '### R: '+str(r1).rjust(4)+" -> "+str(r).rjust(4)+'   NUM:'+str(commLength).rjust(6)+'   qCorr:  ' + str(round(qCorr[r],5)).ljust(8)

		r1 = r
		n = n+1


	index = np.where((rMatrix>rmin)*(rMatrix<=rmax-1)==True)
	list_i = data_i[index].ravel()
	list_j = data_j[index].ravel()
	(list_i, list_j) = data_remove(list_i, list_j, ilim=ilim, jlim=jlim)
	commLength = len(list_i)
	if len(list_i)<8: totCorr = 0.0
	else: totCorr = cal_correlation(list_i, list_j);

	print "### TOTAL: "+str(rmin).rjust(4)+" -> "+str(rmax-1).rjust(4)+'   NUM:'+str(commLength).rjust(6)+'   qCorr:  ' + str(round(qCorr[r],5)).ljust(8)
	return qCorr, totCorr





zf = iFile()
print ('### reading dataset one ...')
data_i = zf.h5reader(args.i, args.iname)
print ('### reading dataset two ...')
data_j = zf.h5reader(args.j, args.jname)
assert data_i.shape == data_j.shape


if args.count != "":
	print ("### reading the counter file from "+args.i)
	counter = zf.h5reader(args.i, args.count )
	print ("### max/min counter = ", np.amin(counter), np.amax(counter) )
	index = np.where(counter<2.5)
	print ("### mask out ", len(index[0]))
	data_i[index] = ilim[0]-1024
	data_j[index] = jlim[0]-1024

	
if args.bins == -1:
	qCorr = q_Shell_Corr(data_i, data_j, center=(-1,-1,-1), rmin=args.rmin, rmax=args.rmax, expand=args.expand, ilim=ilim, jlim=jlim, mode=args.mode) #mode can be "ball" or "shell"
elif args.bins > 1:
	qCorr, totCorr = q_Shell_Corr_Bins(data_i, data_j, center=(-1,-1,-1), rmin=args.rmin, rmax=args.rmax, bins=int(args.bins), ilim=ilim, jlim=jlim)
else:
	raiseException("### Bins is wrong")

fsave = args.o+'/corr-sep-list.h5'+args.tag
print '### saving file: ', fsave
ThisFile = zf.readtxt(os.path.realpath(__file__))
zf.h5writer(fsave, 'execute', ThisFile)
zf.h5modify(fsave, 'qCorr', qCorr)

if args.v != 0:
	import matplotlib.pyplot as plt
	plt.figure(figsize=(10,6))
	plt.plot(qCorr)
	plt.ylim(0.5,1.1)
	plt.tight_layout()
	plt.show()


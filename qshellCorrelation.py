import os, sys
import numpy as np
import h5py
from fileManager import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="file 1", default=".", type=str)
parser.add_argument("-j","--j", help="file 2", default=".", type=str)
parser.add_argument("-tag","--tag", help="tag", default="", type=str)
parser.add_argument("-o","--o", help="save path", default=".", type=str)
parser.add_argument("-expand","--expand", help="expand", default=1.0, type=float)
parser.add_argument("-i1","--i1", help="ilim[0]", default="-100", type=str)
parser.add_argument("-i2","--i2", help="ilim[1]", default="100", type=str)
parser.add_argument("-j1","--j1", help="jlim[0]", default="-100", type=str)
parser.add_argument("-j2","--j2", help="jlim[1]", default="100", type=str)
args = parser.parse_args()
if args.i1 == "." or args.i2 == ".": ilim=None
else: ilim=(float(args.i1), float(args.i2))
if args.j1 == "." or args.j2 == ".": jlim=None
else: jlim=(float(args.j1), float(args.j2))
#path_exp = 'result-scale-test-0005'
#path_sim = '/reg/data/ana04/users/zhensu/xpptut/experiment/0023/0018/volumelist.h5'




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
		index = np.where(((list1>=ilim[0]) and (list1<=ilim[1]))==True)
		list1 = list1[index].copy()
		list2 = list2[index].copy()
		if len(list1)==0: return (list1, list2)
	if jlim is not None:
		index = np.where(((list2>=jlim[0]) and (list2<=jlim[1]))==True)
		list1 = list1[index].copy()
		list2 = list2[index].copy()
	return (list1, list2)

## correlation calculation
def q_Shell_Corr(data_i, data_j, center=(-1,-1,-1), rmin=0, rmax=-1, expand=1, ilim=None, jlim=None):
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
		print 'radius is --------------- ', r
		index = np.where(rMatrix==r)
		list_i = data_i[index].ravel()
		list_j = data_j[index].ravel()
		(list_i, list_j) = data_remove(list_i, list_j, ilim=ilim, jlim=jlim)
		print 'data size = ', len(list_i)
		if len(list_i)<8:
			print 'corrlection = ', 0.0
			qCorr[r] = 0.0
		qCorr[r] = cal_correlation(list_i, list_j);
		print 'corrlection = ', qCorr[r]
	return qCorr


zf = iFile()
print ('reading dataset one ...')
data_i = zf.h5reader(args.i, 'anisoData')
print ('reading dataset two ...')
data_j = zf.h5reader(args.j, 'anisoData')
assert data_i.shape == data_j.shape

qCorr = q_Shell_Corr(data_i, data_j, center=(-1,-1,-1), rmin=0, rmax=-1, expand=args.expand, ilim=ilim, jlim=jlim)

fsave = os.path.join(args.o, '/corr-sep-list.h5'+tag)
ThisFile = zf.readtxt(os.path.realpath(__file__))
zf.h5writer(fsave, 'execute', ThisFile)
zf.h5modify(fsave, 'qCorr', qCorr)
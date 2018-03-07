import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from FileManager import *
zf = iFile()

parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="Dmatrix file name", default="", type=str)
parser.add_argument("-x","--x", help="plot x axis", default=1, type=int)
parser.add_argument("-y","--y", help="plot y axis", default=2, type=int)
parser.add_argument("-v","--v", help="verbose", default=False)
args = parser.parse_args()

if args.i is "": raise Exception('no such file ... ')

def nearest(idata, N):
	data = idata.astype(float)
	for i in range(len(data)):
		index = np.argsort(data[i])[::-1]
		data[i, index[N:]] = 0
	data1 = data.copy()
	data2 = np.transpose(data).copy()
	num1 = data1 > 0
	num1 = num1.astype(float)
	num2 = data2 > 0
	num2 = num2.astype(float)
	data = data1+data2
	num = (num1+num2)
	index = np.where(num>0)
	data[index] /= num[index]
	return data

Dmatrix = zf.h5reader(args.i, 'dmatrix-sym')
label = zf.h5reader(args.i, 'label')

N = len(Dmatrix)

vmean = np.mean(Dmatrix)
vmax = np.amax(Dmatrix)
print 'size = ', Dmatrix.shape 

Dmatrix = np.exp( (Dmatrix-1)/1.6487 ) 

D12 = np.zeros((N,N))
M = np.zeros((N,N))
D121 = np.zeros((N,N))
alpha = 0.5

for i in range(N):
	D12[i,i] = np.sum(Dmatrix[i,:])**(-alpha)
L12 = np.dot(D12, np.dot(Dmatrix, D12))

for i in range(N):
	D121[i,i] = np.sum(L12[i,:])**(-1.)
Mmatrix = np.dot(D121, L12)

val, vec = np.linalg.eig(Mmatrix)
argval = np.argsort(val)[::-1]

args.x = argval[args.x]
args.y = argval[args.y]
print args.x, args.y, val[args.x], val[args.y]


(minx, maxx) = ( np.amin(val[args.x]*vec[:,args.x]), np.amax(val[args.x]*vec[:,args.x]) )
(miny, maxy) = ( np.amin(val[args.y]*vec[:,args.y]), np.amax(val[args.y]*vec[:,args.y]) )
gapx = maxx - minx
gapy = maxy - miny


InPlot = np.zeros((N,2))
InPlot[:,0] = val[args.x]*vec[:,args.x].copy()
InPlot[:,1] = val[args.y]*vec[:,args.y].copy()
zf.h5writer('InPlot.h5', 'InPlot', InPlot, 'float64')



fig, ax = plt.subplots(figsize=(10,10))

plt.plot(val[args.x]*vec[:,args.x].ravel(),	  val[args.y]*vec[:,args.y].ravel(), '.', ms=8, color='b')
plt.plot(val[args.x]*vec[label, args.x].ravel(), val[args.y]*vec[label, args.y].ravel(), '.', ms=8, color='r')
plt.xlim(minx-gapx*0.1, maxx+gapx*0.2)
plt.ylim(miny-gapy*0.2, maxy+gapy*0.2)

label = zf.h5reader(args.i, 'label')
if args.v:
	for i in range(N):
		#if i%10 != 0: continue
		if label[i] < 3.5: continue
		ax.text(val[args.x]*vec[i][args.x], val[args.y]*vec[i][args.y], str(i), fontsize=11)
index = np.argsort(vec[:,args.y])
print index[:100]

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.tight_layout()
plt.show()

from numba import jit
import numpy as np 
from scipy.signal import medfilt
import scipy.ndimage

def lens(idata):
	return np.sqrt(np.sum(idata**2))

def medianf (data, scale):
	return scipy.ndimage.median_filter(data, scale)

@jit
def hklExtraction(idata, astar, bstar, cstar, icen=60, ithreshold=(-100,1000), iradius=(0,1e5), outer=(5,5,5), inner=None):
	
	num_samp = idata.shape[0]
	center = (num_samp -1.)/2.
	ilen = 2*icen+1
	model3d = np.ones((ilen, ilen, ilen))*-32768
	
	for i in range(-icen, icen+1):
		for j in range(-icen, icen+1):
			for k in range(-icen, icen+1):
				pos = float(i)*astar+float(j)*bstar+float(k)*cstar
				if np.sum(pos**2) > iradius[1]**2 or np.sum(pos**2) < iradius[0]**2: continue
				hkl = np.around(pos+center).astype(int)
				x = hkl[0]
				y = hkl[1]
				z = hkl[2]

				if (x<outer[0]) or (x>(num_samp-outer[0]-1)) or (y<outer[1]) or (y>(num_samp-1-outer[1])) or (z<outer[2]) or (z>(num_samp-1-outer[2])): continue

				Temp = idata[(x-outer[0]):(x+1+outer[0]), (y-outer[1]):(y+outer[1]+1), (z-outer[2]):(z+1+outer[2])].copy()
				if inner is not None:
					Temp[(outer[0]-inner[0]):(outer[0]+inner[0]+1), (outer[1]-inner[1]):(outer[1]+inner[1]+1), (outer[2]-inner[2]):(outer[2]+inner[2]+1)] = -32768

				Temp = Temp[ np.where(Temp > ithreshold[0]) ].copy()
				Temp = Temp[ np.where(Temp < ithreshold[1]) ].copy()

				if len(Temp) < 5: continue

				imean = np.nanmean(Temp)
				istd = np.nanstd(Temp)
				Temp = Temp[np.where(Temp > (imean - 2.*istd)) ].copy()
				Temp = Temp[np.where(Temp < (imean + 2.*istd)) ].copy()
				if len(Temp) < 5: continue

				imean = np.mean(Temp)
				model3d[i+icen,j+icen,k+icen] = imean
				
	return model3d

@jit
def lauesym(idata, ithreshold=(-100,1000)):
	icopy = idata.copy()
	num_samp = idata.shape[0]
	for i in range(num_samp):
		for j in range(num_samp):
			for k in range(num_samp):
				mi = num_samp-1-i
				mj = num_samp-1-j
				mk = num_samp-1-k
				pairs = np.array([ idata[i,j,k], idata[mi,mj,mk], idata[mi,j,mk], idata[i,mj,k] ])
				ori = pairs.copy()
				pairs = pairs[np.where(pairs>ithreshold[0])].copy()
				pairs = pairs[np.where(pairs<ithreshold[1])].copy()
				if len(pairs) == 0: icopy[i,j,k] = np.mean(ori)
				else: icopy[i,j,k] = np.mean(pairs)
	return icopy
				
	
	
@jit
def hkl2volume(idata, astar, bstar, cstar, ithreshold=(-100,1000)):
	num_samp = idata.shape[0]
	icen = (num_samp-1)/2
	center = np.array([icen]*3).astype(float)
	size = num_samp
	model3d = np.zeros((num_samp, num_samp, num_samp))
	weight = np.zeros((num_samp, num_samp, num_samp))
	
	for i in range(-icen, icen+1):
		for j in range(-icen, icen+1):
			for k in range(-icen, icen+1):

				if idata[i+icen, j+icen, k+icen] < ithreshold[0]: continue
				if idata[i+icen, j+icen, k+icen] > ithreshold[1]: continue

				hkl = float(i)*astar+float(j)*bstar+float(k)*cstar
				pos = hkl+center

				tx = pos[0]
				ty = pos[1]
				tz = pos[2]

				x = int(tx)
				y = int(ty)
				z = int(tz)

				# throw one line more
				if (tx < 0) or x > (num_samp-1) or (ty < 0) or y > (num_samp-1) or (tz < 0) or z > (num_samp-1): continue

				fx = tx - x
				fy = ty - y
				fz = tz - z
				cx = 1. - fx
				cy = 1. - fy
				cz = 1. - fz

				# Correct for solid angle and polarization
				w = idata[i+icen,j+icen,k+icen]

				# save to the 3D volume
				f = cx*cy*cz 
				weight[x, y, z] += f 
				model3d[x, y, z] += f * w 

				f = cx*cy*fz 
				weight[x, y, ((z+1)%size)] += f 
				model3d[x, y, ((z+1)%size)] += f * w 

				f = cx*fy*cz 
				weight[x, ((y+1)%size), z] += f 
				model3d[x, ((y+1)%size), z] += f * w 

				f = cx*fy*fz 
				weight[x, ((y+1)%size), ((z+1)%size)] += f 
				model3d[x, ((y+1)%size), ((z+1)%size)] += f * w 

				f = fx*cy*cz 
				weight[((x+1)%size), y, z] += f 
				model3d[((x+1)%size), y, z] += f * w

				f = fx*cy*fz 
				weight[((x+1)%size), y, ((z+1)%size)] += f 
				model3d[((x+1)%size), y, ((z+1)%size)] += f * w 

				f = fx*fy*cz 
				weight[((x+1)%size), ((y+1)%size), z] += f
				model3d[((x+1)%size), ((y+1)%size), z] += f * w 

				f = fx*fy*fz 
				weight[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f 
				model3d[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f * w 

	index = np.where(weight>0)
	model3d[index] /= weight[index]
	index = np.where(weight<=0)
	model3d[index] = -32768
	return model3d

@jit
def distri(idata, astar, bstar, cstar, ithreshold=(-100,1000), iscale=1, iwindow=5):

	la = lens(astar)
	lb = lens(bstar)
	lc = lens(cstar)
	num_samp = idata.shape[0]
	center = (num_samp - 1.)/2.
	ir = (int(center*np.sqrt(3.)*max(la/lb, lc/lb))+20)*iscale
	print 'ir='+str(ir)
	distri = np.zeros(ir)
	weight = np.zeros(ir)
	Rmodel = np.zeros((num_samp, num_samp, num_samp)).astype(int)
	
	for i in range(num_samp):
		for j in range(num_samp):
			for k in range(num_samp):
				if idata[i,j,k]<ithreshold[0] or idata[i,j,k]>ithreshold[1]: continue
				ii = i-center
				jj = j-center
				kk = k-center
				r = float(ii)*astar+float(jj)*bstar+float(kk)*cstar
				r = np.sqrt(np.sum(r**2))*float(iscale)
				intr = int(round(r))
				Rmodel[i,j,k] = intr
				
				isize = (iwindow-1)/2
				for delta in range(-isize, isize+1):
					if (intr+delta)>=0 and (intr+delta)<len(distri):
						distri[intr+delta] += idata[i,j,k]
						weight[intr+delta] += 1.

	index = np.where(weight>0)
	distri[index] = distri[index]/weight[index]
	return [distri, Rmodel]

@jit
def meanf(idata, _scale = 3, clim=(0,50)):
	delta = (_scale - 1)/2
	idata = idata.astype(float)
	newList = idata.copy()
	for i in range(len(idata)):
		istart = i-delta
		iend = i+delta
		if istart < 0: istart = 0
		if iend > len(idata)-1: iend = len(idata)+1
		Temp = idata[istart: iend+1].copy()
		Temp = Temp[np.where(Temp>clim[0])].copy()
		Temp = Temp[np.where(Temp<clim[1])].copy()
		if Temp.shape[0] == 0: continue
		newList[i] = np.nanmean(Temp)
	return newList
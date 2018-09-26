import numpy as np
from PyQt4 import QtGui
import pyqtgraph as pg
import os, sys
import matplotlib.pyplot as plt


class imagePloter:
	def pgplot(self, idata, clim=None):
		dataff = idata.copy()
		if clim is None:
			print('automatic parameter finding ... ')
			datastring = dataff.copy()
			datastring.shape = np.prod(dataff.shape),
			datastring = np.sort(datastring)
			length = len(datastring)
			mind = length*0.98-1
			mind = int(mind)
			maxmax = datastring[mind]
			minmin = 0
			if maxmax <= minmin:
				minmin = np.amin(datastring)

		else:
			minmin = clim[0]
			maxmax = clim[1]  

		pg.image(dataff, title="crystal rotation series",levels=(minmin, maxmax))
		if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
			pg.QtGui.QApplication.exec_()

	def pltRing(self, arr, r = None, unit=None, center=(None,None), spacing=None, clim=None):
		(nx, ny) = arr.shape
		cx = center[0];
		cy = center[1];
		if cx is None: cx = (nx-1.)/2.
		if cy is None: cy = (ny-1.)/2.
		if spacing is None:
			spacing = max(abs(nx-cx), abs(ny-cy), abs(cx), abs(cy))/5.
			numRing = 5
		else:
			numRing = int( max(abs(nx-cx), abs(ny-cy), abs(cx), abs(cy))/spacing)

		if r is not None and unit is not None:
			r = r*1.0/unit

		if r is None:
			rad = np.linspace(0,2*np.pi, 1000)
			for i in range(numRing):
				xaxis = spacing*(i+1)*np.sin(rad)+cx
				yaxis = spacing*(i+1)*np.cos(rad)+cy
				if i==0:
					x = xaxis.copy();
					y = yaxis.copy();
				else:
					x = np.append(x, xaxis);
					y = np.append(y, yaxis);					
		else:
			rad = np.linspace(0,2*np.pi, 1000)
			spacing = r
			x = spacing*np.sin(rad)+cx
			y = spacing*np.cos(rad)+cy


		plt.figure(figsize=(8,8));
		plt.imshow(arr, clim=clim)
		plt.plot(x, y, '--', linewidth=2,color='black')
		plt.title('spacing r='+str(round(spacing, 2)))
		plt.tight_layout()
		plt.show()

	def angularDistri(self, arr, Arange=None, num=30, rmax=None, rmin=None, center=(None,None)):
		"""
		num denotes how many times you want to divide the angle
		"""
		assert len(arr.shape)==2
		(nx, ny) = arr.shape
		cx = center[0];
		cy = center[1];
		if cx is None: cx = (nx-1.)/2.
		if cy is None: cy = (ny-1.)/2.

		xaxis = np.arange(nx)-cx + 1.0e-5; 
		yaxis = np.arange(ny)-cy + 1.0e-5; 
		[x,y] = np.meshgrid(xaxis, yaxis, indexing='ij')
		r = np.sqrt(x**2+y**2)
		sinTheta = y/r;
		cosTheta = x/r; 
		angle = np.arccos(cosTheta);
		index = np.where(sinTheta<0);
		angle[index] = 2*np.pi - angle[index]
		if rmin is not None:
		    index = np.where(r<rmin);
		    angle[index] = -1
		if rmax is not None:
		    index = np.where(r>rmax);
		    angle[index] = -1
		if Arange is not None:
		    index = np.where((angle>Arange[0]*np.pi/180.)*(angle<Arange[1]*np.pi/180.)==True);
		    subData = arr[index].copy()
		    aveIntens = np.mean(subData)
		    aveAngle = (Arange[0]+Arange[1])/2.        
		    return [aveAngle, aveIntens];

		rad = np.linspace(0, 2*np.pi, num+1);
		aveIntens = np.zeros(num)
		aveAngle = np.zeros(num)
		for i in range(num):
		    index = np.where((angle>rad[i])*(angle<rad[i+1])==True);
		    subData = arr[index].copy()
		    aveIntens[i] = np.mean(subData)
		    aveAngle[i] = (rad[i]+rad[i+1])/2.
		return [aveAngle, aveIntens]


	def sliceCut(self, data, axis='x', window=5, center=None, clim=None):
		"""
		input a 3d volume, then it will output the average slice within certain range and angle
		"""
		(nx,ny,nz) = data.shape
		if center is None:
			cx = (nx-1.)/2.;
			cy = (ny-1.)/2.;
			cz = (nz-1.)/2.;
		else:
			(cx,cy,cz) = center;
		if clim is None: 
			(vmin, vmax) = (-100, 1000);
		else:
			(vmin, vmax) = clim;
			
		nhalf = (window-1)/2
		Vindex = ((data>=vmin)*(data<=vmax)).astype(float);

		if axis == 'x':
			return np.sum(data[cx-nhalf:cx+nhalf+1,:,:]*Vindex[cx-nhalf:cx+nhalf+1,:,:], axis=0)/(np.sum(Vindex[cx-nhalf:cx+nhalf+1,:,:], axis=0)+1.0e-5)
		elif axis == 'y':
			return np.sum(data[:,cx-nhalf:cx+nhalf+1,:]*Vindex[:,cx-nhalf:cx+nhalf+1,:], axis=1)/(np.sum(Vindex[:,cx-nhalf:cx+nhalf+1,:], axis=1)+1.0e-5)        
		elif axis == 'z':
			return np.sum(data[:,:,cx-nhalf:cx+nhalf+1]*Vindex[:,:,cx-nhalf:cx+nhalf+1], axis=2)/(np.sum(Vindex[:,:,cx-nhalf:cx+nhalf+1], axis=2)+1.0e-5)
		else: 
			return 0
import numpy as np
from numba import jit

def Lattice2vector(a,b,c,ag1,ag2,ag3):
	"""
	convert lattice constants into x,y,z vectors and h,k,l reciprocal vectors.
	alpha, beta, gamma are in angle (0-180)
	a,b,c are in A
	"""
	alpha = ag1/180.*np.pi;
	beta  = ag2/180.*np.pi;
	gamma = ag3/180.*np.pi;
	vecx = a*np.array([1., 0., 0.]);
	vecy = b*np.array([np.cos(gamma), np.sin(gamma), 0]);
	vecz = c*np.array([np.cos(beta), (np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma), np.sqrt(1.+2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)/np.sin(gamma)])
	recH = np.cross(vecy, vecz)/vecx.dot(np.cross(vecy, vecz));
	recK = np.cross(vecz, vecx)/vecy.dot(np.cross(vecz, vecx));
	recL = np.cross(vecx, vecy)/vecz.dot(np.cross(vecx, vecy));
	return (vecx, vecy, vecz, recH, recK, recL)

def ThreeDIndexing(a,b,c):
	nx = int(a)
	ny = int(b)
	nz = int(c)
	cx = nx/2.-1./2.
	cy = ny/2.-1./2.
	cz = nz/2.-1./2.
	x = np.arange(nx)-cx
	y = np.arange(ny)-cy
	z = np.arange(nz)-cz
	[xaxis, yaxis, zaxis] = np.meshgrid(x,y,z, indexing='ij')
	return (xaxis, yaxis, zaxis)



def Crystfel2Rotation(abc_star):
	astar = abc_star[0:3]
	bstar = abc_star[3:6]
	cstar = abc_star[6:9]
	mat = np.array([astar, bstar, cstar]).T
	rot = np.linalg.inv(mat)
	return rot

def Quat2Rotation(quaternion):
	rot = np.zeros([3,3])
	(q0, q1, q2, q3) = quaternion
	q01 = q0*q1 
	q02 = q0*q2 
	q03 = q0*q3 
	q11 = q1*q1 
	q12 = q1*q2 
	q13 = q1*q3 
	q22 = q2*q2 
	q23 = q2*q3 
	q33 = q3*q3 

	rot[0, 0] = (1. - 2.*(q22 + q33)) 
	rot[0, 1] = 2.*(q12 - q03) 
	rot[0, 2] = 2.*(q13 + q02) 
	rot[1, 0] = 2.*(q12 + q03) 
	rot[1, 1] = (1. - 2.*(q11 + q33)) 
	rot[1, 2] = 2.*(q23 - q01) 
	rot[2, 0] = 2.*(q13 - q02) 
	rot[2, 1] = 2.*(q23 + q01) 
	rot[2, 2] = (1. - 2.*(q11 + q22)) 
	return rot

def Euler2Rotation(Euler):
	return 

def RandomRotation():
	return 

def make_radius(size, center=None):
	(nx, ny) = size
	if center is None:
		cx = (nx-1.)/2.
		cy = (ny-1.)/2.
		center = (cx,cy)
	x = np.arange(nx) - center[0]
	y = np.arange(ny) - center[1]
	[xaxis, yaxis] = np.meshgrid(x, y, indexing='ij')
	radius = np.sqrt(xaxis**2 + yaxis**2)
	return radius

def Geometry(image, Geo):
	"""
	The unit of wavelength is A
	"""
	waveLength = Geo['waveLength']
	center = Geo['center']

	(nx, ny) = image.shape
	x = np.arange(nx) - center[0]
	y = np.arange(ny) - center[1]
	[xaxis, yaxis] = np.meshgrid(x, y)
	xaxis = xaxis.T.ravel()
	yaxis = yaxis.T.ravel()
	zaxis = np.ones(nx*ny)*Geo['detDistance']/Geo['pixelSize']
	norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
	## The first axis is negative
	voxel = (np.array([xaxis,yaxis,zaxis])/norm - np.array([[0.],[0.],[1.]]))/waveLength
	return voxel

@jit
def ImageMerge_HKL(model3d, weight, image, Geo, Volume, Kpeak=False):
	Vsize = Volume['volumeSize']
	Vcenter = Volume['volumeCenter']
	Vsample = Volume['volumeSampling']
	voxel = Geometry(image, Geo)

	Image = image.ravel()
	Rot = Geo['rotation']
	HKL = Vsample*(Rot.dot(voxel)).T

	for t in range(len(HKL)):

		if (Image[t] < 0): continue
		
		hkl = HKL[t] + Vcenter
		
		h = hkl[0] 
		k = hkl[1] 
		l = hkl[2] 
		
		inth = int(round(h)) 
		intk = int(round(k)) 
		intl = int(round(l)) 

		if (inth<0) or inth>(Vsize-1) or (intk<0) or intk>(Vsize-1) or (intl<0) or intl>(Vsize-1): continue
		
		hshift = abs(h/Vsample-round(h/Vsample))
		kshift = abs(k/Vsample-round(k/Vsample))
		lshift = abs(l/Vsample-round(l/Vsample))
		if (hshift<0.25) and (kshift<0.25) and (lshift<0.25) and not Kpeak: continue
		
		weight[ inth,intk,intl] += 1.
		model3d[inth,intk,intl] += Image[t] 

	return [model3d, weight]

@jit
def ImageMerge_XYZ(model3d, weight, image, Geo, Volume, Kpeak=False):
	Vsize = Volume['volumeSize']
	Vcenter = Volume['volumeCenter']
	Vsample = Volume['volumeSampling']
	voxel = Geometry(image, Geo)

	Image = image.ravel()
	Rot = Geo['rotation']
	Umatrix = Geo['Umatrix']
	HKL = (Rot.dot(voxel)).T
	XYZ = Vsample*HKL.dot(Umatrix.T)

	for t in range(len(XYZ)):

		if (Image[t] < 0): continue
		
		xyz = XYZ[t] + Vcenter
		
		x = xyz[0] 
		y = xyz[1] 
		z = xyz[2] 
		
		intx = int(round(x)) 
		inty = int(round(y)) 
		intz = int(round(z)) 

		if (intx<0) or intx>(Vsize-1) or (inty<0) or inty>(Vsize-1) or (intz<0) or intz>(Vsize-1): continue

		h = HKL[t,0] 
		k = HKL[t,1] 
		l = HKL[t,2] 		
		hshift = abs(h-round(h))
		kshift = abs(k-round(k))
		lshift = abs(l-round(l))
		if (hshift<0.25) and (kshift<0.25) and (lshift<0.25) and not Kpeak: continue
		
		weight[ intx,inty,intz] += 1.
		model3d[intx,inty,intz] += Image[t] 

	return [model3d, weight]


def ModelScaling(model3d, weight):
	index = np.where(weight>0)
	model3d[index] /= weight[index]
	index = np.where(weight<1.0e-4)
	model3d[index] = -1024
	return model3d

def expand_mask(mask, cwin=(2,2), value=1):
	"""
	cwin is the half size of window
	"""
	(nx,ny) = mask.shape
	newMask = mask.copy()
	index = np.where(mask==value)
	for i in range(-cwin[0], cwin[0]+1):
		for j in range(-cwin[1], cwin[1]+1):
			newMask[((index[0]+i)%nx, (index[1]+j)%ny)] = value
	return newMask
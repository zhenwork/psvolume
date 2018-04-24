from mpidata import *
import numpy as np
import h5py
import os
from numba import jit
#from fileManager import *
# from imageMergeClient import *
import argparse
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
args.o = '/reg/data/ana04/users/zhensu/xpptut/volume'


class iFile:
	def h5writer(self, fname, keys, data, chunks=None, opts=7):
		f = h5py.File(fname, 'w')
		if chunks is None:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
		else:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
		idatawr[...] = np.array(data)
		f.close()
		
	def h5reader(self, fname, keys=None):		
		f = h5py.File(fname, 'r')
		if keys is None: keys = f.keys()[0]
		idata = f[keys].value
		f.close()
		return idata
		
	def h5modify(self, fname, keys, data, chunks=None, opts=7):
		f = h5py.File(fname, 'r+')
		try: f.__delitem__(keys)
		except: pass
		
		if chunks is None:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
		else:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
		idatawr[...] = np.array(data)
		f.close()



	def get_image_info(self, path):
		f = h5py.File(path, 'r')
		Info = {}
		Info['readout'] = f['readout'].value
		Info['waveLength'] = f['waveLength'].value
		Info['polarization'] = f['polarization'].value
		Info['detDistance'] = f['detDistance'].value
		Info['pixelSize'] = f['pixelSize'].value
		Info['center'] = f['center'].value
		Info['exp'] = f['exp'].value
		Info['run'] = f['run'].value
		Info['event'] = f['event'].value
		Info['rotation'] = f['rotation'].value
		Info['rot'] = f['rot'].value
		Info['scale'] = f['scale'].value
		f.close()
		return Info

	def readtxt(self, path):
		f = open(path)
		content = f.readlines()
		f.close()
		return content

def Geometry(image, Geo):
	"""
	The unit of wavelength is nm
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
def ImageMerge(model3d, weight, image, Geo, Volume):
	Vsize = Volume['volumeSize']
	Vcenter = Volume['volumeCenter']
	Vsample = Volume['volumeSampling']
	center = Geo['center']
	orientation = Geo['rotation']

	voxel = Geometry(image, Geo)

	Image = image.ravel()
	if Geo['rot']=='matrix': Rot = Geo['rotation']
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
		if (hshift<0.25) and (kshift<0.25) and (lshift<0.25): continue
		
		weight[ inth,intk,intl] += 1.
		model3d[inth,intk,intl] += Image[t] 

	return [model3d, weight]



zf = iFile()
#if args.num==-1: num = zf.counterFile(args.o+'/mergeImage', title='.slice')
#else: num = int(args.num)
num = int(args.num)

Vol = {}
Vol['volumeCenter'] = 60
Vol['volumeSampling'] = 1
Vol['volumeSize'] = 2*Vol['volumeCenter']+1
model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)

if comm_rank == 0:
	fsave = args.o+'/sp0002' #zf.makeFolder(args.o, title='sp')
	print "Folder: ", fsave
	for nrank in range(comm_size-1):
		md=mpidata()
		md.recv()
		model3d += md.model3d
		weight += md.weight
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(2)

	model3d = ModelScaling(model3d, weight)
	pathIntens = fsave+'/merge.volume'
	#ThisFile = zf.readtxt(os.path.realpath(__file__))
	#zf.h5writer(pathIntens, 'execute', ThisFile)
	zf.h5modify(pathIntens, 'intens', model3d, chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)
	zf.h5modify(pathIntens, 'weight', weight,  chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)

else:
	sep = np.linspace(0, num, comm_size).astype('int')
	for idx in range(sep[comm_rank-1], sep[comm_rank]):
		fname = args.o+'/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
		image = zf.h5reader(fname, 'image')
		Geo = zf.get_image_info(fname)
		image /= Geo['scale']
		print '### rank ' + str(comm_rank).rjust(2) + ' is processing file: '+str(idx)+'/'+str(num)
		[model3d, weight] = ImageMerge(model3d, weight, image, Geo, Vol)

	print '### rank ' + str(comm_rank).rjust(2) + ' is sending file ... '
	md=mpidata()
	md.addarray('model3d', model3d)
	md.addarray('weight', weight)
	md.small.rank = comm_rank
	md.send()
	md = None



#import numpy as np
#import h5py
#import os


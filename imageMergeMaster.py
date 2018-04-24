from mpidata import *
import numpy as np
import h5py
import os
#from fileManager import *
from imageMergeClient import *
import argparse
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
args.o = '/reg/data/ana04/users/zhensu/xpptut/volume'

zf = iFile()
if args.num==-1: num = zf.counterFile(args.o+'/mergeImage', title='.slice')
else: num = int(args.num)

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
	ThisFile = zf.readtxt(os.path.realpath(__file__))
	zf.h5writer(pathIntens, 'execute', ThisFile)
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

	def makeFolder(self, path, title='sp'):
		allFile = os.listdir(path)
		fileNumber = [0]
		for each in allFile:
			if each[:2] == title and each[-4:].isdigit():
				fileNumber.append(int(each[-4:]))
		newNumber = np.amax(fileNumber) + 1
		fnew = os.path.join(path, 'sp'+str(newNumber).zfill(4))
		if not os.path.exists(fnew): os.mkdir(fnew)
		return fnew

	def counterFile(self, path, title='.slice'):
		allFile = os.listdir(path)
		counter = 0
		for each in allFile:
			if title in each:
				counter += 1
		return counter

		# file_name = os.path.realpath(__file__)
		# if (os.path.isfile(file_name)): shutil.copy(file_name, folder_new)

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
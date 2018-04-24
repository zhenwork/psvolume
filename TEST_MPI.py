import os
import numpy as np 
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#from mpidata import mpidata, small
from fileManager import *
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
args.o = '/reg/data/ana04/users/zhensu/xpptut/volume'

class arrayinfo(object):
	def __init__(self,name,array):
		self.name = name
		self.shape = array.shape
		self.dtype = array.dtype

class small(object):
	def __init__(self):
		self.arrayinfolist = []
		self.endrun = False
	def addarray(self,name,array):
		self.arrayinfolist.append(arrayinfo(name,array))

class mpidata(object):

	def __init__(self):
		self.small=small()
		self.arraylist = []

	def endrun(self):
		self.small.endrun = True
		comm.send(self.small,dest=0,tag=comm_rank)

	def addarray(self,name,array):
		self.arraylist.append(array)
		self.small.addarray(name,array)

	def send(self):
		assert comm_rank!=0
		comm.send(self.small,dest=0,tag=comm_rank)
		for arr in self.arraylist:
			assert arr.flags['C_CONTIGUOUS']
			comm.Send(arr,dest=0,tag=comm_rank)

	def recv(self):
		assert comm_rank==0
		status=MPI.Status()	   
		self.small=comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
		recvRank = status.Get_source()
		if not self.small.endrun:
			for arrinfo in self.small.arrayinfolist:
				if not hasattr(self,arrinfo.name) or arr.shape!=arrinfo.shape or arr.dtype!=arrinfo.dtype:
					setattr(self,arrinfo.name,np.empty(arrinfo.shape,dtype=arrinfo.dtype))
				arr = getattr(self,arrinfo.name)
				comm.Recv(arr,source=recvRank,tag=MPI.ANY_TAG)


				
zf = iFile()
if args.num==-1: num = zf.counterFile(args.o+'/mergeImage', title='.slice')
else: num = int(args.num)
num = int(args.num)

Vol = {}
Vol['volumeCenter'] = 60
Vol['volumeSampling'] = 1
Vol['volumeSize'] = 2*Vol['volumeCenter']+1
model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)


if comm_rank == 0:
	fsave = zf.makeFolder(args.o, title='sp')
	print "Folder: ", fsave
	for nrank in range(comm_size-1):
		md=mpidata()
		md.recv()
		model3d += md.model3d
		weight += md.weight
		recvRank = md.small.rank
		print '### received file from ' + str(recvRank).rjust(2)+'/'+str(comm_size-1)
		print np.amax(md.model3d)

	print "### start saving files ... "

else:
	sep = np.linspace(0, num, comm_size).astype('int')
	for idx in range(sep[comm_rank-1], sep[comm_rank]):
		fname = args.o+'/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
		image = zf.h5reader(fname, 'image')
		Geo = zf.get_image_info(fname)
		image /= Geo['scale']
		print '### rank ' + str(comm_rank).rjust(2) + ' is processing file: '+str(idx)+'/'+str(num)
		#[model3d, weight] = ImageMerge(model3d, weight, image, Geo, Vol)

	print '### rank ' + str(comm_rank).rjust(2) + ' is sending file ... '
	md=mpidata()
	md.addarray('model3d', model3d)
	md.addarray('weight', weight)
	md.small.rank = comm_rank
	md.send()
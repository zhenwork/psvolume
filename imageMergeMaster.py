import os
import numpy as np 
from mpidata import *
from fileManager import *
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-U","--U", help="matrix", default="hkl", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
parser.add_argument("-volumeSampling","--volumeSampling", help="num of images to process", default=1, type=int)
parser.add_argument("-volumeCenter","--volumeCenter", help="num of images to process", default=60, type=int)
args = parser.parse_args()

zf = iFile()
if args.num==-1: num = zf.counterFile(args.o+'/mergeImage', title='.slice')
else: num = int(args.num)


Vol = {}
Vol['volumeCenter'] = int(args.volumeCenter)
Vol['volumeSampling'] = int(args.volumeSampling)
Vol['volumeSize'] = 2*Vol['volumeCenter']+1
model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)

if comm_rank == 0:
	fsave = zf.makeFolder(args.o, title='sr')
	print "### Folder: ", fsave
	print "### Images: ", num
	print "### Mode: ", args.U
	print "### Volume: ", model3d.shape
	print "### Center: ", Vol['volumeCenter']
	print "### Sampling: ", Vol['volumeSampling']

	for nrank in range(comm_size-1):
		md=mpidata()
		md.recv()
		model3d += md.model3d
		weight += md.weight
		Umatrix = md.Umatrix
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(2) + '/' + str(comm_size)

	model3d = ModelScaling(model3d, weight)
	pathIntens = fsave+'/merge.volume'
	if args.U == 'xyz': Umatrix = np.eye(3)

	print "### saving File: ", pathIntens
	ThisFile = zf.readtxt(os.path.realpath(__file__))
	zf.h5writer(pathIntens, 'execute', ThisFile)
	zf.h5modify(pathIntens, 'intens', model3d, chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)
	zf.h5modify(pathIntens, 'weight', weight,  chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)
	zf.h5modify(pathIntens, 'Umatrix', Umatrix)

else:
	sep = np.linspace(0, num, comm_size).astype('int');

	for idx in range(sep[comm_rank-1], sep[comm_rank]):
		fname = args.o+'/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
		image = zf.h5reader(fname, 'image')
		Geo = zf.get_image_info(fname)
		Geo['Umatrix'] = zf.h5reader(fname, 'Umatrix')
		
		print '### rank ' + str(comm_rank).rjust(2) + ' is processing file: '+str(sep[comm_rank-1])+'/'+str(idx)+'/'+str(sep[comm_rank])
		if args.U=='xyz': 
			[model3d, weight] = ImageMerge_XYZ(model3d, weight, image, Geo, Vol)
		else: 
			[model3d, weight] = ImageMerge_HKL(model3d, weight, image, Geo, Vol)

	print '### rank ' + str(comm_rank).rjust(2) + ' is sending file ... '
	md=mpidata()
	md.addarray('model3d', model3d)
	md.addarray('weight', weight)
	md.addarray('Umatrix', Geo['Umatrix'])
	md.small.rank = comm_rank
	md.send()
	md = None
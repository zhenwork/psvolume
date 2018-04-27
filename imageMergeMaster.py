import os
import numpy as np 
from mpidata import *
from fileManager import *
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-mode","--mode", help="matrix", default="hkl", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
parser.add_argument("-peak","--peak", help="keep the bragg peak or not", default=False, type=bool)
parser.add_argument("-vSampling","--vSampling", help="num of images to process", default=1, type=int)
parser.add_argument("-vCenter","--vCenter", help="num of images to process", default=60, type=int)
args = parser.parse_args()


zf = iFile()
if not (args.o).endswith('/'): args.o = args.o+'/'
[num, allFile] = zf.counterFile(args.o, title='.slice')
path = args.o[0:(len(args.o)-args.o[::-1].find('/',1))];
prefix = allFile[0][0:(len(allFile[0])-allFile[0][::-1].find('_',1))];
if args.num != -1: num = int(args.num)

Vol = {}
Vol['volumeCenter'] = int(args.vCenter)
Vol['volumeSampling'] = int(args.vSampling)
Vol['volumeSize'] = 2*Vol['volumeCenter']+1
model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)


if comm_rank == 0:
	fsave = zf.makeFolder(path, title='sr')

	print '### Path  : ', path
	print '### Folder: ', args.o 
	print '### Prefix: ', prefix 
	print "### Fsave : ", fsave
	print "### Images: ", num
	print "### Mode  : ", args.mode
	print "### Volume: ", model3d.shape
	print "### Center: ", Vol['volumeCenter']
	print "### Sampling: ", Vol['volumeSampling']

	for nrank in range(comm_size-1):
		md=mpidata()
		md.recv()
		model3d += md.model3d
		weight += md.weight
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(2) + '/' + str(comm_size)

	model3d = ModelScaling(model3d, weight)
	pathIntens = fsave+'/merge.volume'
	if args.mode == 'xyz': Smat = np.eye(3)
	else: Smat = zf.h5reader(prefix+str(0).zfill(5)+'.slice', 'Smat')

	print "### saving File: ", pathIntens
	ThisFile = zf.readtxt(os.path.realpath(__file__))
	zf.h5writer(pathIntens, 'execute', ThisFile)
	zf.h5modify(pathIntens, 'intens', model3d, chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)
	zf.h5modify(pathIntens, 'weight', weight,  chunks=(1, Vol['volumeSize'], Vol['volumeSize']), opts=7)
	zf.h5modify(pathIntens, 'Smat', Smat)

else:
	sep = np.linspace(0, num, comm_size).astype('int')
	for idx in range(sep[comm_rank-1], sep[comm_rank]):
		fname = prefix+str(idx).zfill(5)+'.slice'
		image = zf.h5reader(fname, 'image')
		Geo = zf.get_image_info(fname)
		image = image * Geo['scale']

		sumIntens = round(np.sum(image), 8)
		if args.mode=='xyz':
			moniter = 'xyz'
			[model3d, weight] = ImageMerge_XYZ(model3d, weight, image, Geo, Vol, Kpeak=arsg.peak)
		else:
			moniter = 'hkl'
			[model3d, weight] = ImageMerge_HKL(model3d, weight, image, Geo, Vol, Kpeak=arsg.peak)
		print '### rank ' + str(comm_rank).rjust(3) + ' is processing file: '+str(sep[comm_rank-1])+'/'+str(idx)+'/'+str(sep[comm_rank]) +'  sumIntens: '+str(sumIntens).ljust(10)

	print '### rank ' + str(comm_rank).rjust(3) + ' is sending file ... '
	md=mpidata()
	md.addarray('model3d', model3d)
	md.addarray('weight', weight)
	md.small.rank = comm_rank
	md.send()
	md = None
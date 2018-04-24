from mpidata import *
from fileManager import iFile
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
args.o = '/reg/data/ana04/users/zhensu/xpptut/volume'

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
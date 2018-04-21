from mpidata import *
from fileManager import *
from imageProcessClient import *
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()

zf = iFile()
if args.num==-1:	
	num = zf.counterFile(args.o+'/rawImage', title='.slice')
else: num = int(args.num)
sep = np.linspace(0, num, comm_size+1).astype('int')
Smatrix = np.zeros((num, num))

Tmp = zf.h5reader(args.o + '/mergeImage/mergeImage_00000.slice', 'image')
Geo = zf.get_image_info(args.o + '/mergeImage/mergeImage_00000.slice')
(nx,ny) = Tmp.shape
mask = circle_region(image=None, center=Geo['center'], rmax=450, rmin=100, size=(nx,ny))

if sep[1]-sep[0]<200: 
	imgMatrix = np.zeros((sep[comm_rank+1]-sep[comm_rank], nx,ny))
	for idx in range(sep[comm_rank], sep[comm_rank+1]): 
		filename = args.o + '/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
		image = zf.h5reader(filename, 'image')
		image[np.where(image<0.)] = 0.
		imgMatrix[idx-sep[comm_rank],:,:] = image*mask
		image = None
	for jdx in range(num):
		filename = args.o + '/mergeImage/mergeImage_'+str(jdx).zfill(5)+'.slice'
		image = zf.h5reader(filename, 'image')
		cirImage_2 = image*mask
		cirImage_2[np.where(cirImage_2<0)] = 0.
		for idx in range(len(imgMatrix)):
			cirImage_1 = imgMatrix[idx]
			Smatrix[idx+sep[comm_rank], jdx] = np.sum(cirImage_1*cirImage_2)/np.sum(cirImage_2**2)
		cirImage_2 = None

else:
	for jdx in range(num):
		filename = args.o + '/mergeImage/mergeImage_'+str(jdx).zfill(5)+'.slice'
		image = zf.h5reader(filename, 'image')
		cirImage_2 = image*mask
		cirImage_2[np.where(cirImage_2<0)] = 0.
		for idx in range(sep[comm_rank], sep[comm_rank+1]):
			filename = args.o + '/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
			image = zf.h5reader(filename, 'image')
			image[np.where(image<0.)] = 0.
			cirImage_1 = image*mask
			Smatrix[idx+sep[comm_rank], jdx] = np.sum(cirImage_1*cirImage_2)/np.sum(cirImage_2**2)
		cirImage_2 = None

if comm_rank == 0:
	for i in range(comm_size-1):
		md=mpidata()
		md.recv()
		Smatrix += md.Smatrix
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(2)
	zf.h5modify(args.o+'/image.process', 'Smatrix', Smatrix)

else:
	md=mpidata()
	md.addarray('Smatrix', Smatrix)
	md.small.rank = comm_rank
	md.send()
	md = None
	print '### rank ' + str(comm_rank).rjust(2) + ' is sending file ... '
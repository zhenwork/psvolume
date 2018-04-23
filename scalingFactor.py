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
scaleMatrix = np.zeros(num)

for idx in range(sep[comm_rank], sep[comm_rank+1]):
	filename = args.o + '/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'
	image = zf.h5reader(filename, 'image')
	image[np.where(image<0.)] = 0.
	if idx == sep[comm_rank]: 
		Geo = zf.get_image_info(filename)
		(nx,ny) = image.shape
		(cx,cy) = Geo['center']
		print 'making mask:  ('+str(nx)+','+str(ny)+')-('+str(cx)+','+str(cy)+')'
		mask = circle_region(image=None, center=(cx,cy), rmax=450, rmin=100, size=(nx,ny))
	maskImage = image*mask
	scaleMatrix[idx] = np.sum(maskImage)

if comm_rank == 0:
	for i in range(comm_size-1):
		md=mpidata()
		md.recv()
		scaleMatrix += md.scaleMatrix
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(2)
	zf.h5modify(args.o+'/image.process', 'scale', scaleMatrix)

else:
	md=mpidata()
	md.addarray('scaleMatrix', scaleMatrix)
	md.small.rank = comm_rank
	md.send()
	md = None
	print '### rank ' + str(comm_rank).rjust(2) + ' is sending file ... '
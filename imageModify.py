"""
Script for modifying the image information

The --o must be a folder name including the "mergeImage/" for example

mpirun -n 10 python imageModify --o ./mergeImage --num 100
"""

from userScript import *
from fileManager import *
from mpi4py import MPI
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()



zf = iFile()
if args.num==-1:
	num = zf.counterFile(args.o, title='.slice')
else: num = int(args.num)
sep = np.linspace(0, num, comm_size+1).astype('int')
if comm_rank == 0:
	print "### Folder: ", args.o
	print "### Images: ", num






## modify image
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	filename = args.o + '/mergeImage_'+str(idx).zfill(5)+'.slice'
	zf.h5modify(filename, 'Umatrix', Umatrix)
	print '### rank ' + str(comm_rank).rjust(2) + ' is processing: ' + str(idx)+'/'+str(num)
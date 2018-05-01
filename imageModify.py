"""
Script for modifying the image information

The --o must be a folder name including the "mergeImage/" for example

mpirun -n 10 python imageModify --o ./mergeImage --num 100
"""


from fileManager import *
from mpi4py import MPI
from shutil import copyfile
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()



zf = iFile()
if not (args.i).endswith('/'): args.i = args.i+'/'
[num, allFile] = zf.counterFile(args.i, title='.slice')
path = args.i[0:(len(args.i)-args.i[::-1].find('/',1))];
if args.num != -1: num = int(args.num)

sep = np.linspace(0, num, comm_size+1).astype('int')
if comm_rank == 0:
	print "### Path: ", path
	print "### Folder: ", args.i
	print "### Images: ", num


RingIntens_index = zf.h5reader('/reg/data/ana04/users/zhensu/xpptut/volume/ICH_wt_cds4/crystal/image.process', 'RingScale');
RingIntens_blank = zf.h5reader('/reg/data/ana04/users/zhensu/xpptut/volume/ICH_wt_cds4/blank/image.process', 'RingScale')

## modify image
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	filename = '/reg/data/ana04/users/zhensu/xpptut/volume/ICH_wt_cds4/crystal/mergeImage/'+str(idx).zfill(5)+'.slice'
	check = zf.h5reader(filename, 'scale')
	assert abs(check - RingIntens_index[0]/RingIntens_index[idx] )<0.001;
	scale_blank = RingIntens_index[0]/RingIntens_blank[idx];

	zf.h5modify('./mergeImage/'+str(idx).zfill(5)+'.slice', 'scale', scale_blank)
	
	print '### rank ' + str(comm_rank).rjust(3) + ' is processing: ' +str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1])
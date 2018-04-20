from mpidata import *
from userScript import *
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="raw file folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
num = args.num

sep = np.linspace(0, N, comm_size+1).astype('int')
for i in range(N):
	Image_i = 
	for j in range(N):


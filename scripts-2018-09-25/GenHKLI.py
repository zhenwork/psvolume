import numpy as np 
from mpidata import *
from fileManager import *
from imageProcessClient import *
from imageMergeClient import *
from numba import jit
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-rmin","--rmin", help="min radius", default=100, type=int)
parser.add_argument("-rmax","--rmax", help="max radius", default=1200, type=int)
parser.add_argument("-nmin","--nmin", help="smallest index of image", default=0, type=int)
parser.add_argument("-nmax","--nmax", help="largest index of image", default=-1, type=int)
parser.add_argument("-o","--o", help="output folder", default="HKLI_List", type=str)
parser.add_argument("-bg","--bg", help="background volume", default=None)
args = parser.parse_args()


zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[num, allFile] = zio.counterFile(folder_i, title='.slice')
if args.nmax == -1: args.nmax = int(num)
testFolder = path_i+'/testFolder'

	
if comm_rank == 0:
	print "### Resource folder: %s"%folder_i
	print "### Radial range: %d-%d"%(args.rmin, args.rmax)
	print "### Index range:  %d-%d"%(args.nmin, args.nmax)
	print "### Output file: %s"%args.o



@jit
def GenHKLI_alg1(image, Geo, backg=None, peakRing=0.25):

	Vsize = 121
	Vcenter = 60
	Vsample = 1

	HKLI_List = np.zeros((Vsize,Vsize,Vsize))
	weight = np.zeros((Vsize,Vsize,Vsize))

	voxel = Geometry(image, Geo)
	Image = image.ravel()
	Rot = Geo['rotation']
	HKL = Vsample*(Rot.dot(voxel)).T

	image_copy = Image.copy()
	
	for t in range(len(HKL)):

		if Image[t] < 0: 
			continue
		
		hkl = HKL[t] + Vcenter
		
		h = hkl[0] 
		k = hkl[1] 
		l = hkl[2] 
		
		inth = int(round(h)) 
		intk = int(round(k)) 
		intl = int(round(l)) 

		if (inth<0) or inth>(Vsize-1) or (intk<0) or intk>(Vsize-1) or (intl<0) or intl>(Vsize-1): 
			continue
		
		hshift = abs(h/Vsample-round(h/Vsample))
		kshift = abs(k/Vsample-round(k/Vsample))
		lshift = abs(l/Vsample-round(l/Vsample))

		if max(hshift, kshift, lshift)<peakRing:
			continue
		else:
			HKLI_List[inth,intk,intl] += Image[t]-backg[inth,intk,intl]
			weight[inth,intk,intl]    += 1. 
			image_copy[t] = -100

	image_copy.shape = image.shape
	
	return HKLI_List, weight, image_copy



def saveHKLI(HKLI_List, weight, fname):
	index = np.where(weight>=10)
	HKLI_List[index] /= weight[index]
	index = np.where(weight<10)
	HKLI_List[index] = -1024
	f = open(fname,'w')
	#f.writelines("H".rjust(5)+"K".rjust(5)+"L".rjust(5)+"I".rjust(10)+"\n")
	for H in range(-60,60):
		for K in range(-60,60):
			for L in range(-60,60):
				i = H+60
				j = K+60
				k = L+60
				I = HKLI_List[i,j,k]
				if I<-1000:
					continue
				I = np.around(I,3)
				f.writelines(str(H).rjust(5)+str(K).rjust(5)+str(L).rjust(5)+str(I).rjust(10)+"\n")
	f.close()




### create the mask
filename = folder_i + '/00000.slice'
Geo = zio.get_image_info(filename)
image = zf.h5reader(filename, 'image')
(nx,ny) = image.shape
(cx,cy) = Geo['center']
print 'making mask:  ('+str(nx)+','+str(ny)+')-('+str(cx)+','+str(cy)+')'
mask = circle_region(image=None, center=(cx,cy), rmax=args.rmax, rmin=args.rmin, size=(nx,ny))
#mask = mask * zf.h5reader(path_i+'image.process','mask')
if comm_rank==0:
	zf.h5modify(path_i+'/image.process', 'GenHKLmask', mask)
	folder_o = args.o
	if not os.path.isdir(folder_o):
		os.makedirs(folder_o)
	if not os.path.isdir(testFolder):
		os.makedirs(testFolder)
else:
	import time
	folder_o = args.o
	while not os.path.isdir(folder_o):
		time.sleep(5)


if args.bg is None:
	backg = np.zeros((121,121,121))
else:
	backg = zf.h5reader(args.bg,'volumeBack')
	assert backg.shape==(121,121,121)

		
sep = np.linspace(args.nmin, args.nmax, comm_size+1).astype('int')
print "### Rank %.4d will process [%.4d, %.4d]"%(comm_rank, sep[comm_rank], sep[comm_rank+1])

for idx in range(sep[comm_rank], sep[comm_rank+1]):

	fname = "%s/%.5d.slice"%(folder_i, idx)
	
	if not os.path.isfile(fname):
		print "### No such file: %s" % fname
		raise Exception("No such file: %s")
		continue

	image = zf.h5reader(fname, 'image')
	Geo = zio.get_image_info(fname)
	image = image * Geo['scale']
	
	## make pixel value to -1 for bad mask position
	index = np.where(mask==0)
	image[index] = -1
	image[np.where(image<0.001)] = -1
	image[np.where(image>10000)] = -1


	HKLI_List, weight, image_copy = GenHKLI_alg1(image, Geo, peakRing=0.25, backg = backg)
	saveHKLI(HKLI_List, weight, fname = "%s/%.5d.hkl"%(folder_o,idx) )
	zf.h5writer(testFolder+'/'+str(idx).zfill(5)+'.slice', 'image', image_copy)

	print "### Rank %.4d finished %.4d-%.4d-%.4d"%(comm_rank, sep[comm_rank], idx, sep[comm_rank+1])

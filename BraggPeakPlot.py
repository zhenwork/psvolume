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
parser.add_argument("-o","--o", help="output file name", default="BraggPeakIntensity", type=str)
parser.add_argument("-name","--name", help="name of saved dataset", default='', type=str)
args = parser.parse_args()

zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[num, allFile] = zio.counterFile(folder_i, title='.slice')
if args.nmax == -1: args.nmax = int(num)


if comm_rank == 0:
	if args.name!="": args.o = args.o+"_"+args.name+".process"
	print "### Resource folder: %s"%folder_i
	print "### Radial range: %d-%d"%(args.rmin, args.rmax)
	print "### Index range:  %d-%d"%(args.nmin, args.nmax)
	print "### Output file: %s"%args.o




@jit
def CalBraggPeakIntensity_alg1(image, Geo, peakRing=0.15, backRing=0.40):
	Vsize = 121
	Vcenter = 60
	Vsample = 1

	peakIntensity = np.zeros((Vsize,Vsize,Vsize))
	peakCounts = np.zeros((Vsize,Vsize,Vsize))
	backIntensity = np.zeros((Vsize,Vsize,Vsize))
	backCounts = np.zeros((Vsize,Vsize,Vsize))

	voxel = Geometry(image, Geo)
	Image = image.ravel()
	Rot = Geo['rotation']
	HKL = Vsample*(Rot.dot(voxel)).T

	pmark = np.zeros(Image.shape)

	for t in range(len(HKL)):

		if (Image[t] < 0): continue
		
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

		if (hshift<=peakRing) and (kshift<=peakRing) and (lshift<=peakRing):
			peakIntensity[inth,intk,intl] += Image[t] 
			peakCounts[inth,intk,intl] += 1.
			pmark[t] += 10.
		elif max(hshift, kshift, lshift)<backRing:
			backIntensity[inth,intk,intl] += Image[t] 
			backCounts[inth,intk,intl] += 1.
			pmark[t] += 5.

	## calculate average background (per pixel)
	index = np.where(backCounts>0.5)
	backIntensity[index] /= backCounts[index]
	
	## peakIntensity subtract the average background*peak numbers
	subIntensity = peakIntensity - backIntensity*peakCounts
	aveSubIntensity = subIntensity.copy()
	
	## calculate average subIntensity (only for calculating index)
	index = np.where(peakCounts>0.5)
	aveSubIntensity[index] = subIntensity[index]/peakCounts[index]

	## index is all HKLs that fit the condition
	index = np.where((peakCounts>=5)*(backCounts>=10)*(aveSubIntensity>5.)==True)
	
	## calculate sum of peak intensity at suitable HKLs
	data = subIntensity[index].copy()
	(A,B) = (np.sum(data), len(data))

	pmark.shape = image.shape

	return (A,B,pmark)




@jit
def CalBraggPeakIntensity_alg1(image, Geo, peakRing=0.15, backRing=0.40):
	Vsize = 121
	Vcenter = 60
	Vsample = 1

	peakIntensity = np.zeros((Vsize,Vsize,Vsize))
	peakCounts = np.zeros((Vsize,Vsize,Vsize))
	backIntensity = np.zeros((Vsize,Vsize,Vsize))
	backCounts = np.zeros((Vsize,Vsize,Vsize))

	voxel = Geometry(image, Geo)
	Image = image.ravel()
	Rot = Geo['rotation']
	HKL = Vsample*(Rot.dot(voxel)).T

	pmark = np.zeros(Image.shape)

	for t in range(len(HKL)):

		if (Image[t] < 0): continue
		
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

		if (hshift<=peakRing) and (kshift<=peakRing) and (lshift<=peakRing):
			peakIntensity[inth,intk,intl] += Image[t] 
			peakCounts[inth,intk,intl] += 1.
			pmark[t] += 10.
		elif max(hshift, kshift, lshift)<backRing:
			backIntensity[inth,intk,intl] += Image[t] 
			backCounts[inth,intk,intl] += 1.
			pmark[t] += 5.

	## calculate average background (per pixel)
	index = np.where(backCounts>0.5)
	backIntensity[index] /= backCounts[index]
	
	## peakIntensity subtract the average background*peak numbers
	subIntensity = peakIntensity - backIntensity*peakCounts
	aveSubIntensity = subIntensity.copy()
	
	## calculate average subIntensity (only for calculating index)
	index = np.where(peakCounts>0.5)
	aveSubIntensity[index] = subIntensity[index]/peakCounts[index]

	## index is all HKLs that fit the condition
	keepVolume = np.zeros(peakCounts.shape).astype(int)
	index = np.where((peakCounts>=5)*(backCounts>=10)*(subIntensity>5.*4)==True)
	keepVolume[index] = 1
	
	## calculate sum of peak intensity at suitable HKLs
	data = subIntensity[index].copy()
	(A,B) = (np.sum(data), len(data))


	pmark.shape = image.shape



	## This is just a test of the algorithm
	peakTest = Image.copy()
	for t in range(len(HKL)):

		if (Image[t] < 0): continue
		
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

		if keepVolume[inth, intk, intl] < 1:
			continue
		if max(hshift, kshift, lshift)<peakRing:
			continue
		elif max(hshift, kshift, lshift)<backRing:
			peakTest[t] = 1000

	peakTest.shape = image.shape

	##############################


	return (A,B,pmark, peakTest)





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
	zf.h5writer('./test_mask.h5', 'mask', mask)


BraggPeakIntensity = np.zeros(args.nmax)
BraggPeakCounts = np.zeros(args.nmax)


sep = np.linspace(args.nmin, args.nmax, comm_size+1).astype('int')
print "### Rank %.4d will process [%.4d, %.4d]"%(comm_rank, sep[comm_rank], sep[comm_rank+1])

for idx in range(sep[comm_rank], sep[comm_rank+1]):

	fname = "%s/%.5d.slice"%(folder_i, idx)
	
	if not os.path.isfile(fname):
		print "### No such file: %s" % fname
		raiseException("No such file: %s")
		continue

	image = zf.h5reader(fname, 'image')
	Geo = zio.get_image_info(fname)


	## make pixel value to -1 for bad mask position
	index = np.where(mask==0)
	image[index] = -1


	[peakIntensity, peakCounts, pmark, peakTest] = CalBraggPeakIntensity_alg1(image, Geo, peakRing=0.20, backRing=0.30 )
	BraggPeakIntensity[idx] = peakIntensity
	BraggPeakCounts[idx] = peakCounts

	zf.h5writer("./pmark/%.5d.slice"%idx, "pmark", pmark)
	zf.h5writer("./peakTest/%.5d.slice"%idx, "peakTest", peakTest)

	print "### Rank %.4d finished %.4d-%.4d-%.4d"%(comm_rank, sep[comm_rank], idx, sep[comm_rank+1])


if comm_rank == 0:
	for i in range(comm_size-1):
		md=mpidata()
		md.recv()
		BraggPeakIntensity += md.BraggPeakIntensity
		BraggPeakCounts += md.BraggPeakCounts
		recvRank = md.small.rank
		md = None
		print '### received file from ' + str(recvRank).rjust(3)
	if args.o != "":
		zf.h5writer(args.o, "BraggPeakIntensity", BraggPeakIntensity)
		zf.h5modify(args.o, "BraggPeakCounts", BraggPeakCounts)
else:
	md=mpidata()
	md.addarray('BraggPeakIntensity', BraggPeakIntensity)
	md.addarray('BraggPeakCounts', BraggPeakCounts)
	md.small.rank = comm_rank
	md.send()
	md = None
	print '### Rank ' + str(comm_rank).rjust(4) + ' is sending file ... '

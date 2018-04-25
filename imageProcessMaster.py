"""
Preprocess of the images
submit jobs: mpirun -n 10 python imageProcessMaster.py --o /reg/neh/home/zhensu --num 2000
"""

from imageProcessClient import *
from fileManager import *
from mpi4py import MPI
import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()

zf = iFile()
if args.num==-1:
	num = zf.counterFile(args.o+'/rawImage', title='.slice')
else: num = int(args.num)
sep = np.linspace(0, num, comm_size+1).astype('int')


fname = args.o+'/rawImage/rawImage_00000.slice'
Geo = zf.get_image_info(fname)
assert Geo['readout']=='image'
image = zf.h5reader(fname, 'image')

ascale = solid_angle_correction(image, Geo)
pscale = polarization_correction(image, Geo)
apscale = ascale * pscale
apscale /= np.amax(apscale)
apscale = 1./apscale
ascale  = 1./ascale
pscale  = 1./pscale

if comm_rank == 0:
	print '## Total number: '+str(num).rjust(5)
	if not os.path.exists(args.o + '/mergeImage'): 
		os.mkdir(args.o + '/mergeImage')
	zf.h5modify(args.o+'/image.process', 'ascale' , ascale)
	zf.h5modify(args.o+'/image.process', 'pscale' , pscale)
	zf.h5modify(args.o+'/image.process', 'apscale', apscale)
else:
	while not os.path.exists(args.o + '/mergeImage'): pass

mask = zf.h5reader(args.o+'/image.process', 'mask')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	fname = args.o+'/rawImage/rawImage_'+str(idx).zfill(5)+'.slice'
	Geo = zf.get_image_info(fname)
	assert Geo['readout']=='image'
	image = zf.h5reader(fname, 'image')

	image *= apscale
	image *= Geo['scale']
	#image = remove_peak_alg1(image, mask=mask, sigma=15, cwin=(11,11))

	fsave = args.o + '/mergeImage/mergeImage_'+str(idx).zfill(5)+'.slice'

	zf.h5writer(fsave, 'readout', 'image') #image, event
	zf.h5modify(fsave, 'image', image)
	zf.h5modify(fsave, 'center', Geo['center'])
	zf.h5modify(fsave, 'exp', Geo['exp'])
	zf.h5modify(fsave, 'run', Geo['exp'])
	zf.h5modify(fsave, 'event', Geo['exp'])
	zf.h5modify(fsave, 'waveLength', Geo['waveLength'])
	zf.h5modify(fsave, 'detDistance', Geo['detDistance'])
	zf.h5modify(fsave, 'pixelSize', Geo['pixelSize'])
	zf.h5modify(fsave, 'polarization', Geo['polarization'])
	zf.h5modify(fsave, 'rot', 'matrix')
	zf.h5modify(fsave, 'rotation', Geo['rotation'])
	zf.h5modify(fsave, 'scale', Geo['scale'])
	print '## Finished image:'+str(idx).rjust(5)+'/'+str(num)
	if idx>5000: break
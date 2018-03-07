from ImageProcessClient import *
from FileManager import *
from mpi4py import MPI
import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="raw file folder", default="/", type=str)
parser.add_argument("-o","--o", help="save folder", default="/", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()

zf = iFile()
if args.num==-1:	
	num = zf.counterFile(args.i, title='.slice')
else: num = int(args.num)
sep = np.linspace(0, num, comm_size+1).astype('int')


fname = args.i+'/rawImage_00000.slice'
Geo = zf.get_image_info(fname)
assert Geo['readout']=='image'
image = zf.h5reader(fname, 'image')

ascale = solid_angle_correction(image, Geo)
pscale = polarization_correction(image, Geo)
apscale = ascale * pscale
apscale /= np.amax(apscale)
if comm_rank == 0:
	print '## Total number: '+str(num).rjust(5)
	zf.h5writer(args.i+'/ascale.scale', 'scale', ascale)
	zf.h5writer(args.i+'/pscale.scale', 'scale', pscale)
	zf.h5writer(args.i+'/apscale.scale', 'scale', apscale)

mask = zf.h5reader(args.i+'/mask.mask')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	fname = args.i+'/rawImage_'+str(idx).zfill(5)+'.slice'
	Geo = zf.get_image_info(fname)
	assert Geo['readout']=='image'
	image = zf.h5reader(fname, 'image')

	image /= apscale
	image *= Geo['scale']
	image = remove_peak_alg1(image, mask=mask, sigma=15, cwin=(11,11))

	fsave = args.o + '/mergeImage_'+str(idx).zfill(5)+'.slice'

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

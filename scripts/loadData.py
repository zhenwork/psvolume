from fileManager import *
from loadDataTools import *
from imageProcessTools import *
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


zf = iFile()
zio = IOsystem()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fname","--fname", help="input files", default=None)
parser.add_argument("-o","--o", help="save folder", default=None)
parser.add_argument("-xds","--xds", help="xds file", default=None)
parser.add_argument("-nmin","--nmin", help="smallest number", default=0, type=int)
parser.add_argument("-nmax","--nmax", help="largest number", default=-1, type=int)
args = parser.parse_args()

## path_o: the path of output dir
## folder_o: the folder to save results
## path_i: the path of input dir
## folder_i: the folder to load images
[path_o, folder_o] = zio.get_path_folder(args.o)
if args.fname is not None:
	[path_i, folder_i] = zio.get_path_folder(args.fname)
	suffix_i = zio.get_suffix(args.fname)
	[counter, selectFile] = zio.counterFile(path_i, title=suffix_i)
	print "## Total number in %s is:%5d" % (path_i, counter)

if args.nmax==-1:
	args.nmax = counter+args.nmin


# load xds indexing files
if args.xds is not None:
	print "### xds file: %s"%args.xds
	Geo = {}; Bmat = None; invBmat = None;  invAmat = None
	[Geo, Bmat, invBmat, invAmat] = load_GXPARM_XDS(args.xds)
else:
	raise Exception('### No valid xds file')


## print some parameters
if comm_rank == 0:
	print "## path_i: %s"%path_i
	print "## folder_i: %s"%folder_i
	print "## path_o: %s"%path_o
	print "## folder_o: %s"%folder_o
	print "## Process %5d - %5d"%(args.nmin, args.nmax)
	if not os.path.isdir(folder_o):
		os.makedirs(folder_o)
else:
	while not os.path.isdir(folder_o): pass


Smat = Bmat*1.0/np.sqrt(np.sum(Bmat[:,1]**2))


if args.fname != '.': fname = args.fname.replace('#####', str(0).zfill(5))
else: fname=None
mask = user_get_mask(Geo, fname=fname)
Mask = expand_mask(mask, cwin=(2,2), value=0)
if comm_rank == 0:
	print "Total number of images: ", args.num
	Filename = path_o+'/image.process'
	zf.h5writer(Filename, 'mask', mask)
	zf.h5modify(Filename, 'Mask', Mask)
	zf.h5modify(Filename, 'Bmat', Bmat)
	zf.h5modify(Filename, 'Smat', Smat)
	zf.h5modify(Filename, 'invBmat', invBmat)


sep = np.linspace(0, args.num, comm_size+1).astype('int')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	if args.fname != '.':
		fname = args.fname.replace('#####', str(idx).zfill(5))
		
	image = user_get_image(fname = fname)
	image[np.where(image>100000)] = -1
	image[np.where(image<0.001)] = -1
	index = np.where(mask==0)
	image[index] = -1

	quaternion = user_get_orientation(idx, increment=Geo['Angle_increment'])
	R1 = Quat2Rotation(quaternion)

	if invAmat is None: matrix = invBmat.dot(invUmat.dot(R1))
	else: matrix = invAmat.dot(R1)


	# #####
	# FackR = Quat2Rotation( (  np.cos( np.radians(-7)/2. ),   0.,   np.sin( np.radians(-7)/2. ),   0.  ) )
	# matrix = FackR.dot(matrix)
	# #####


	fsave = folder_o + '/'+str(idx).zfill(5)+'.slice'
	zf.h5writer(fsave, 'readout', 'image')
	zf.h5modify(fsave, 'image', image)
	zf.h5modify(fsave, 'center', Geo['center'])
	zf.h5modify(fsave, 'exp', False)
	zf.h5modify(fsave, 'run', False)
	zf.h5modify(fsave, 'event', False)
	zf.h5modify(fsave, 'waveLength', Geo['wavelength'])
	zf.h5modify(fsave, 'detDistance', Geo['detDistance'])
	zf.h5modify(fsave, 'pixelSize', Geo['pixelSize'])
	zf.h5modify(fsave, 'polarization', Geo['polarization'])
	zf.h5modify(fsave, 'rot', 'matrix')
	zf.h5modify(fsave, 'rotation', matrix)
	zf.h5modify(fsave, 'scale', user_get_scalingFactor(idx))
	zf.h5modify(fsave, 'Smat', Smat)
	print '### Rank: '+str(comm_rank).rjust(3)+' finished image:  '+str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1])
	if idx>4000: break

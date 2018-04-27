from fileManager import *
from mpi4py import MPI
import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="image folder", default=".", type=str)
args = parser.parse_args()

zf = iFile()
if not (args.i).endswith('/'): args.i = args.i+'/'
[num, allFile] = zf.counterFile(args.i, title='.slice')
path = args.i[0:(len(args.i)-args.i[::-1].find('/',1))];
prefix = allFile[0][0:(len(allFile[0])-allFile[0][::-1].find('_',1))];


if comm_rank == 0:
	print '### Path  : ', path
	print '### Folder: ', args.i
	print '### Prefix: ', prefix 
	print '### Total number: '+str(num).rjust(5)
	if not os.path.exists(path + '/subImage'): 
		os.mkdir(path + '/subImage')
	print '### save folder: '+ path + '/subImage'
else:
	while not os.path.exists(path + '/subImage'): pass

imgFirst = zf.h5reader(prefix+str(0).zfill(5)+'.slice', 'image')
imgFirst = np.sum(imgFirst*(imgFirst>0))

radius = None
sep = np.linspace(0, num, comm_size+1).astype('int')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	fname = prefix+str(idx).zfill(5)+'.slice'
	Geo = zf.get_image_info(fname)
	assert Geo['readout']=='image'
	image = zf.h5reader(fname, 'image')

	sumIntens = np.sum(image*(image>0))

	image = image/sumIntens*imgFirst
	Geo['scale'] = 1.
	
	sumIntens = round(sumIntens, 8)
	image = remove_bragg_peak(image, mask=mask, sigma=15, cwin=(11,11));

	if radius is None:
		[image, radius] = remove_peak_alg3(image, center=Geo['center'], depth=2)
	else:
		[image, radius] = remove_peak_alg3(image, radius=radius)

	fsave = path + '/subImage/subImage_'+str(idx).zfill(5)+'.slice'

	zf.h5writer(fsave, 'readout', 'image') # image, event
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
	zf.h5modify(fsave, 'Smat', Geo['Smat'])
	print '### Rank: '+str(comm_rank).rjust(3)+' finished image: '+str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1]) + '  sumIntens: '+str(sumIntens).ljust(10)
	if idx>5000: break

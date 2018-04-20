from userScript import *
from fileManager import *
from imageProcessClient import *
from mpi4py import MPI
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
zf = iFile()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default="/", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
## folder = '/reg/data/ana04/users/zhensu/xpptut/volume/rawImage'

# computation
lsima = np.sqrt(np.sum(sima**2))
lsimb = np.sqrt(np.sum(simb**2))
lsimc = np.sqrt(np.sum(simc**2))
Kac = np.arccos(np.dot(sima, simc)/lsima/lsimc)
Kbc = np.arccos(np.dot(simb, simc)/lsimb/lsimc)
Kab = np.arccos(np.dot(sima, simb)/lsima/lsimb)
sima = lsima * np.array([np.sin(Kac), 0., np.cos(Kac)])
simb = lsimb * np.array([0., 1., 0.])
simc = lsimc * np.array([0., 0., 1.])
rot_three = np.linalg.inv(np.array([sima,simb,simc]).T)*Geo['pixelSize']/Geo['wavelength']/Geo['detDistance']

if comm_rank == 0:
	if not os.path.exists(args.o + '/rawImage'): os.mkdir(args.o + '/rawImage')

mask = user_get_mask()
Mask = expand_mask(mask, cwin=(2,2), value=0)


if comm_rank == 0:
	Filename = args.o+'/image.process'
	zf.h5writer(Filename, 'mask', mask)
	zf.h5modify(Filename, 'Mask', Mask)

sep = np.linspace(0, num, comm_size+1).astype('int')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	image = user_get_image(idx)
	quaternion = user_get_orientation(idx)
	rot_one = Quat2Rotation(quaternion)
	matrix = rot_three.dot(rot_three.dot(rot_one))

	fsave = args.o + '/rawImage' + '/rawImage_'+str(idx).zfill(5)+'.slice'
	zf.h5writer(fsave, 'readout', 'image')
	zf.h5modify(fsave, 'image', image*mask)
	zf.h5modify(fsave, 'center', user_get_center(idx))
	zf.h5modify(fsave, 'exp', False)
	zf.h5modify(fsave, 'run', False)
	zf.h5modify(fsave, 'event', False)
	zf.h5modify(fsave, 'waveLength', Geo['wavelength'])
	zf.h5modify(fsave, 'detDistance', Geo['detDistance'])
	zf.h5modify(fsave, 'pixelSize', Geo['pixelSize'])
	zf.h5modify(fsave, 'polarization', Geo['polarization'])
	zf.h5modify(fsave, 'rot', 'matrix')
	zf.h5modify(fsave, 'rotation', matrix)
	zf.h5modify(fsave, 'scale', scale[idx])

	print '## Finished image:'+str(idx).rjust(5)+'/'+str(num)
	if idx>4000: break

import cbf
from FileManager import *
from ImageMergeClient import *
from ImageProcessClient import *
from mpidata import *
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
zf = iFile()

num = 2000
Geo = {}
Geo['pixelSize'] = 172.0
Geo['detDistance'] = 200147.4
Geo['beamStop'] = 50
Geo['polarization'] = 'y'
Geo['wavelength'] = 0.082653
Geo['center'] = (1265.33488372, 1228.00813953)

rot_two = np.array([[-0.2438,  0.9655,  -0.0919],
                    [-0.8608, -0.2591,  -0.4381],
                    [-0.4468, -0.0277,   0.8942]])

sima = np.array([ 0.007369 ,   0.017496 ,   -0.000000])
simb = np.array([-0.000000 ,   0.000000 ,    0.017263])
simc = np.array([ 0.015730 ,   0.000000,     0.000000])
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

def makeMask(Geo):
	path = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'
	fname = os.path.join(path, 'wtich_274_10_1_'+str(1).zfill(5)+'.cbf')
	content = cbf.read(fname)
	data = np.array(content.data).astype(float)
	mask = np.ones(data.shape).astype(int)
	index = np.where(data > 100000)
	mask[index] = 0
	mask[1260:1300,1235:2463] = 0
	radius = make_radius(mask.shape, center=Geo['center'])
	index = np.where(radius<25)
	mask[index] = 0
	return mask

def getFimage(idx):
	return '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'+'/wtich_274_10_1_'+str(idx+1).zfill(5)+'.cbf'

folder = '/reg/data/ana04/users/zhensu/xpptut/volume/rawImage'
mask = makeMask(Geo)
Mask = expand_mask(mask, cwin=(2,2), value=0)
scale = zf.h5reader('/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/data-ana/scalesMike.h5')
scale = 1./scale
if comm_rank == 0:
	zf.h5writer(folder+'/mask.mask', 'mask', mask)
	zf.h5writer(folder+'/expand_mask.mask', 'mask', Mask)

sep = np.linspace(0, num, comm_size+1).astype('int')
for idx in range(sep[comm_rank], sep[comm_rank+1]):
	fname = getFimage(idx)
	content = cbf.read(fname)
	image = np.array(content.data).astype(float)   
	quaternion = (np.cos(idx*0.1*np.pi/2./180.), 0., np.sin(idx*0.1*np.pi/2./180.), 0.)
	rot_one = Quat2Rotation(quaternion)
	matrix = rot_three.dot(rot_three.dot(rot_one))

	fsave = folder + '/rawImage_'+str(idx).zfill(5)+'.slice'
	zf.h5writer(fsave, 'readout', 'image') #image, event
	zf.h5modify(fsave, 'image', image*mask)
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
	zf.h5modify(fsave, 'scale', scale[idx])

	print '## Finished image:'+str(idx).rjust(5)+'/'+str(num)
	if idx>4000: break

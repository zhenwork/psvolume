"""
Preprocess of the images
submit jobs: mpirun -n 10 python imageProcessMaster.py --o /reg/neh/home/zhensu --num 2000

--o: the former folder path. the path "before" rawImage folder
--num: The number of images to process. If no input, then it will search how many files in the rawImage folder
"""

from imageProcessClient import *
from fileManager import *
from mpi4py import MPI
import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="read folder", default=".", type=str)
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
parser.add_argument("-apscale","--apscale", help="num of images to process", default='ap', type=str)

args = parser.parse_args()

zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[path_o, folder_o] = zio.get_path_folder(args.o)
[num, allFile] = zio.counterFile(folder_i, title='.slice')
prefix = allFile[0][0:(len(allFile[0])-allFile[0][::-1].find('_',1))];
if args.num ==-1: args.num = int(num)
sep = np.linspace(0, args.num, comm_size+1).astype('int')



## get image information from sample pattern
Geo = zio.get_image_info(folder_i+'/00000.slice')
image = zf.h5reader(folder_i+'/00000.slice', 'image')

ascale = solid_angle_correction(image, Geo)
pscale = polarization_correction(image, Geo)
apscale = ascale * pscale
apscale /= np.amax(apscale)
apscale = 1./apscale
ascale  = 1./ascale
pscale  = 1./pscale

if comm_rank == 0:
	print '## Total number: '+str(args.num).rjust(5)
	print '## read from Folder: ', folder_i
	print '## save to Path: ', path_o
	print '## save to Folder: ', folder_o
	folder_p = path_o + '/pscale'
	if not os.path.exists(folder_o): os.mkdir(folder_o)
	if not os.path.exists(folder_p): os.mkdir(folder_p)
	zf.h5modify(path_o+'/image.process', 'ascale' , ascale)
	zf.h5modify(path_o+'/image.process', 'pscale' , pscale)
	zf.h5modify(path_o+'/image.process', 'apscale', apscale)
else:
	while not os.path.exists(folder_o): pass

while True:
	print '### Rank: '+str(comm_rank).rjust(3)+' is loading mask ...'
	try: 
		mask = zf.h5reader(path_o+'/image.process', 'mask');
		break
	except: continue;

if args.apscale == 'a':
	apscale = ascale.copy()
	print "### only apply the solid-angle correction "
elif args.apscale == 'p':
	apscale = pscale.copy()
	print "### only apply the polarization correction "
else:
	print "### apply both solid-angle and polarization correction "
	pass

for idx in range(sep[comm_rank], sep[comm_rank+1]):
	fname = folder_i+'/'+str(idx).zfill(5)+'.slice'
	fsave = folder_o + '/'+str(idx).zfill(5)+'.slice'
	psave = folder_p + '/'+str(idx).zfill(5)+'.slice'

	Geo = zio.get_image_info(fname)
	image = zf.h5reader(fname, 'image')
	
	image = remove_peak_alg1(image, mask=mask, sigma=15, cwin=(11,11))

	# ################
	# from imageMergeClient import expand_mask
	# Mask = np.ones(image.shape)
	# index = np.where(image<0.01)
	# Mask[index] = 0
	# Mask = expand_mask(Mask, cwin=(2,2), value=0)
	# index = np.where(Mask<0.5)
	# image[index] = -1
	# image[np.where(image<0.001)] = -1
	# ################

	pimage = image * pscale
	pimage[np.where(pimage<0)] = -1
	
	image *= apscale
	image[np.where(image<0)] = -1
	sumIntens = round(np.sum(image), 8)
	#image = remove_peak_alg1(image, mask=mask, sigma=15, cwin=(11,11))
	


	zio.copyFile(src=fname, dst=fsave)
	zio.copyFile(src=fname, dst=psave)
	zf.h5modify(fsave, 'image', image)
	zf.h5modify(psave, 'image', pimage)
	print '### Rank: '+str(comm_rank).rjust(3)+' finished image: '+str(sep[comm_rank])+'/'+str(idx)+'/'+str(sep[comm_rank+1]) + '  sumIntens: '+str(sumIntens).ljust(10)
	if idx>5000: break

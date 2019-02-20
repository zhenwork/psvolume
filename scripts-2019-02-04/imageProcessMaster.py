from imageProcessTools import *
from imageMergeTools import RemoveBragg
from fileManager import *
from mpi4py import MPI
import argparse
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="read folder", default=None)
parser.add_argument("-o","--o", help="save folder", default=None)
parser.add_argument("-nmin","--nmin", help="small number", default=0, type=int)
parser.add_argument("-nmax","--nmax", help="large number", default=-1, type=int)
parser.add_argument("-voxel","--voxel", help="voxel list", default=None)
args = parser.parse_args()

zf = iFile()
zio = IOsystem()
[path_i, folder_i] = zio.get_path_folder(args.i)
[path_o, folder_o] = zio.get_path_folder(args.o)
[num, allFile] = zio.counterFile(folder_i, title='.slice')
prefix = allFile[0][0:(len(allFile[0])-allFile[0][::-1].find('_',1))];
if args.nmax ==-1: 
    args.nmax = num+args.nmin


## FIXME: 
Geo = zio.get_image_info(folder_i+'/00000.slice')
image = zf.h5reader(folder_i+'/00000.slice', 'image')

ascale = solid_angle_correction(image, Geo)
pscale = polarization_correction(image, Geo)
apscale = ascale * pscale
apscale /= np.amax(apscale)
apscale = 1./apscale
ascale  = 1./ascale
pscale  = 1./pscale

folder_p = path_o + '/pscale'
if comm_rank == 0:
    print '## Total number: '+str(args.num).rjust(5)
    print '## read from Folder: ', folder_i
    print '## save to Path: ', path_o
    print '## save to Folder: ', folder_o
    if not os.path.isdir(folder_o): os.mkdir(folder_o)
    if not os.path.isdir(folder_p): os.mkdir(folder_p)
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



# FIXME: This is specific for the snc dataset
#########################
if args.voxel != ".":
    print "## voxel exist: %s" % args.voxel
    voxel = np.load(args.voxel)
    voxel = voxel.T
    print "## Loaded the voxel file ... "
else:
    print "## voxel doesn't exist ... "
    voxel = None
#########################

## save image to the rawImage folder
sep = np.linspace(args.nmin, args.nmax, comm_size+1).astype('int')
print "## Rank:%3d/%3d processes: %4d - %4d"%(comm_rank,comm_size,sep[comm_rank],sep[comm_rank+1])

for idx in range(sep[comm_rank], sep[comm_rank+1]):
    fname = folder_i+'/'+str(idx).zfill(5)+'.slice'
    fsave = folder_o + '/'+str(idx).zfill(5)+'.slice'
    psave = folder_p + '/'+str(idx).zfill(5)+'.slice'

    Geo = zio.get_image_info(fname)
    image = zf.h5reader(fname, 'image')
    
    image = remove_peak_alg1(image, _mask=mask, sigma=15, cwin=(11,11))
    image = RemoveBragg(image, Geo, box=0.25, voxel=voxel)
    
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

from numba import jit
import os, sys
import numpy as np  
import mpi4py.MPI as MPI  
import time
import h5py


tic = time.time()
comm = MPI.COMM_WORLD  
comm_rank = comm.Get_rank()  
comm_size = comm.Get_size()  
status = MPI.Status()
workers = ((comm_size - 1)*2)
closed_workers = 0 
comm_process = 0 

#parameter
class parameter(object):
    def __init__(self):
        self.detsize = (2527, 2463)
        self.num_pix = np.prod(self.detsize)
        self.num_data = 2000
        self.num_samp = 1000
        self.pix_size = 172.0
        self.det_dis = 200147.4
        self.lam = 1.6
        self.beamstop = 50
        self.bad_pix_mask = 0
        self.good_pix_mask = 1
        self.bad_pix_value = 100000
        self.polarization = 'y'

para = parameter()
samp_size = (para.num_samp, para.num_samp, para.num_samp)
x_chunk = (1, para.num_samp, para.num_samp)
y_chunk = (para.num_samp, 1, para.num_samp)
z_chunk = (para.num_samp, para.num_samp, 1)


folder = os.getcwd()
all_file = os.listdir(folder)
file_num = [0]
for each in all_file:
    if each[:2] == '00':
        file_num.append(int(each[-4:]))
new_name = np.amax(file_num) + 1
folder_new = os.path.join(folder, str(new_name).zfill(4))

if comm_rank == 0:
    if not os.path.exists(folder_new):
        os.mkdir(folder_new)
    file_name = os.path.realpath(__file__)
    if (os.path.isfile(file_name)):
        print 'yes'
        shutil.copy(file_name, folder_new)
    file_name = os.path.join(folder, 'config.ini')
    if (os.path.isfile(file_name)):
        print 'yes'
        shutil.copy(file_name, folder_new)

path_intens = os.path.join(folder_new,'result.h5')
path_weight = os.path.join(folder_new,'weight.h5')
#path_back = '/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/rawdata'
path_back = '/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/rawdata-mask'

@jit
def detread(para):
    path = '/reg/d/psdm/cxi/cxitut13/scratch/zhensu/wtich_274k_10/cbf'
    fname = os.path.join(path, 'wtich_274_10_1_'+str(1).zfill(5)+'.cbf')
    content = cbf.read(fname)
    data = np.array(content.data)
    
    index = np.where(data > para.bad_pix_value)      
    for i in range(-2,3):
        for j in range(-2,3):
            data[(index[0]+i)%para.detsize[0], (index[1]+j)%para.detsize[1]] = 1e7    
    data[1260:1300,1235:2463] = 1e7
    return data

@jit
def detread_bak_bak(para):
    path = '/reg/data/ana04/users/zhensu/xpptut/experiment/0024/ICHg150off/rawcbf'
    fname = os.path.join(path, 'I1ichg150a3off_10_'+str(1).zfill(5)+'.cbf')
    content = cbf.read(fname)
    data = np.array(content.data)
    if (data.shape[0]!=para.detsize[0]) or (data.shape[1]!=para.detsize[1]):
        print 'wrong ... '
    else:
        print 'right ... '
    index = np.where(data > para.bad_pix_value)      
    for i in range(-2,3):
        for j in range(-2,3):
            data[(index[0]+i)%para.detsize[0], (index[1]+j)%para.detsize[1]] = 1e7    
    #data[1260:1300,1235:2463] = 1e7
    data[1310:1340, 1173:] = 1e8
    return data

@jit
def make_rot_quat(quaternion):
    # right-hand screw rule
    rot = np.zeros([3,3]);
    
    q0 = quaternion[0] ;
    q1 = quaternion[1] ;
    q2 = quaternion[2] ;
    q3 = quaternion[3] ;
    
    q01 = q0*q1 ;
    q02 = q0*q2 ;
    q03 = q0*q3 ;
    q11 = q1*q1 ;
    q12 = q1*q2 ;
    q13 = q1*q3 ;
    q22 = q2*q2 ;
    q23 = q2*q3 ;
    q33 = q3*q3 ;
    
    rot[0, 0] = (1. - 2.*(q22 + q33)) ;
    rot[0, 1] = 2.*(q12 - q03) ;
    rot[0, 2] = 2.*(q13 + q02) ;
    rot[1, 0] = 2.*(q12 + q03) ;
    rot[1, 1] = (1. - 2.*(q11 + q33)) ;
    rot[1, 2] = 2.*(q23 - q01) ;
    rot[2, 0] = 2.*(q13 - q02) ;
    rot[2, 1] = 2.*(q23 + q01) ;
    rot[2, 2] = (1. - 2.*(q11 + q22)) ;

    return rot

def perpixel(lam=3.0, wavelength=1.3, pixsize=89.0, detd=53300.0):
    return lam * float(pixsize) / float(wavelength) / float(detd)


rot2 = np.array([[-0.2438,  0.9655,  -0.0919],
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

lscale = perpixel(lam=1.6, wavelength=0.82653, pixsize=172.0, detd=200147.4)
print 'sima, simb, simc, angle = ', lsima, lsimb, lsimc, Kab, Kac, Kbc
print 'sima, simb, simc = ', lsima/lscale, lsimb/lscale, lsimc/lscale

sima = lsima/lscale * np.array([np.sin(Kac), 0., np.cos(Kac)])
simb = lsimb/lscale * np.array([0., 1., 0.])
simc = lsimc/lscale * np.array([0., 0., 1.])
print 'sima = ', sima
print 'simb = ', simb
print 'simc = ', simc

imat_inv = np.linalg.inv(np.transpose(np.array([sima,simb,simc])))


@jit
def slice_merge(quaternion, _slice, model3d, weight, detector, mask, rot2, imat_inv, para):
    
    size = para.num_samp
    num_pix = para.num_pix
    bad_pix_mask = para.bad_pix_mask

    center = (size - 1.)/2.   
    rot = make_rot_quat(quaternion)
    
    for t in range(0, num_pix):
        
        if (mask[t] == bad_pix_mask) or (_slice[t] < -0.5):
            continue
            
        rot_ori = np.dot(rot, detector[t, 0:3])
        rot_xyz = np.dot(rot2, rot_ori)
        rot_pix = rot_xyz + center
        
        tx = rot_pix[0] ;
        ty = rot_pix[1] ;
        tz = rot_pix[2] ;
        
        x = int(tx) ;
        y = int(ty) ;
        z = int(tz) ;
        
        # throw one line more
        if (tx < 0) or x > (size-2) or (ty < 0) or y > (size-2) or (tz < 0) or z > (size-2):
            continue ;

        [xh, yk, zl] = np.dot(imat_inv, rot_xyz).ravel()
        axh = np.around(xh)
        ayk = np.around(yk)
        azl = np.around(zl)
        dxh = np.abs(axh - xh)
        dyk = np.abs(ayk - yk)
        dzl = np.abs(azl - zl)
        if (dxh < 0.25) and (dyk < 0.25) and (dzl < 0.25): 
            #_slice[t] = -1000
            continue


        fx = tx - x ;
        fy = ty - y ;
        fz = tz - z ;
        cx = 1. - fx ;
        cy = 1. - fy ;
        cz = 1. - fz ;
        
        # Correct for solid angle and polarization
        #_slice[t] /= detector[t, 3]
        w = _slice[t] ;
        
        # save to the 3D volume
        f = cx*cy*cz ;
        weight[x, y, z] += f ;
        model3d[x, y, z] += f * w ;
        
        f = cx*cy*fz ;
        weight[x, y, ((z+1)%size)] += f ;
        model3d[x, y, ((z+1)%size)] += f * w ;
        
        f = cx*fy*cz ;
        weight[x, ((y+1)%size), z] += f ;
        model3d[x, ((y+1)%size), z] += f * w ;
        
        f = cx*fy*fz ;
        weight[x, ((y+1)%size), ((z+1)%size)] += f ;
        model3d[x, ((y+1)%size), ((z+1)%size)] += f * w ;
        
        f = fx*cy*cz ;
        weight[((x+1)%size), y, z] += f ;
        model3d[((x+1)%size), y, z] += f * w ;
        
        f = fx*cy*fz ;
        weight[((x+1)%size), y, ((z+1)%size)] += f ;
        model3d[((x+1)%size), y, ((z+1)%size)] += f * w ;
        
        f = fx*fy*cz ;
        weight[((x+1)%size), ((y+1)%size), z] += f ;
        model3d[((x+1)%size), ((y+1)%size), z] += f * w ;
        
        f = fx*fy*fz ;
        weight[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f ;
        model3d[((x+1)%size), ((y+1)%size), ((z+1)%size)] += f * w ;
        
    return [model3d, weight]


@jit
def scaling(model3d, weight):
    index = np.where(weight > 0) ;
    model3d[index] /= weight[index] ;
    return model3d 


@jit
def make_detector(para):
    
    detd = float(para.det_dis)/float(para.pix_size)
    #center = ((para.detsize[0] - 1.)/2., (para.detsize[1] - 1.)/2.)
    #center = (1321, 1173)
    center = (1265.33488372, 1228.00813953)
    detector = np.zeros([para.num_pix, 5])
    detector[:,4] = para.good_pix_mask

    data = detread(para);
    data.shape = para.num_pix,

    xaxis = np.arange(para.detsize[0]) - center[0]
    yaxis = np.arange(para.detsize[1]) - center[1]   
    [x,y] = np.meshgrid(xaxis,yaxis)
    
    x = np.transpose(x).reshape(para.num_pix,)
    y = np.transpose(y).reshape(para.num_pix,)
    detd = np.array([detd]*para.num_pix)
     
    radius = np.sqrt(x**2+y**2)    

    index = np.where(data > para.bad_pix_value)
    detector[index,4] = para.bad_pix_mask
    
    index = np.where(radius < para.beamstop)
    detector[index,4] = para.bad_pix_mask
    
    norm = np.sqrt(x**2 + y**2 + detd**2)
    detector[:,0:3] = np.transpose((np.array([x,y,detd])/norm - np.array([[0.],[0.],[1.]]))/para.lam*detd)
    
    detector[:,3] = detd/norm**3

    maxscale = np.amax(detector[:,3])
    detector[:,3] /= float(maxscale)

    return detector


@jit
def make_quaternion(para):
    quaternion = np.zeros([para.num_data, 4])
    for idx in range(para.num_data):
        quaternion[idx,:] = [np.cos(idx*0.1*np.pi/2./180.), 0., np.sin(idx*0.1*np.pi/2./180.), 0.]
    return quaternion

# making detector and quaternion
print('making detector and quaternion and mask ... ') 
quaternion = make_quaternion(para)
detector = make_detector(para)
print 'maxscale = ', np.amax(detector[:,3])
mask = detector[:,4].copy()
mask = mask.astype(int)

print ('Rank ' + str(comm_rank) + ' is ready')
f = h5py.File('/reg/data/ana04/users/zhensu/xpptut/experiment/0024/wtich/data-ana/scalesMike.h5','r')
scale = np.array(f[f.keys()[0]]).astype(float)
f.close()


print 'scale min/max: ', np.amin(scale), np.amax(scale)

if __name__ == "__main__":  
    
    local_offset = np.linspace(0, para.num_data, comm_size + 1).astype('int')    #para.num_data

    model3d = np.zeros(samp_size)
    weight = np.zeros(samp_size)
    num_pix = para.num_pix

    if (comm_rank == 0) and (comm_size>1):
        toc = 0
        while(toc<200):
            toc = time.time()-tic

    for idx in range(local_offset[comm_rank], local_offset[comm_rank + 1]):       
        fname = os.path.join(path_back,'rawdata_mask_'+str(idx+1).zfill(5)+'.h5')
        f = h5py.File(fname,'r')
        data = np.array(f[f.keys()[0]]).astype(float)
        f.close()

        data.shape = num_pix,
        data *= scale[idx]
        
        [model3d, weight] = slice_merge(quaternion[idx], data, model3d, weight, detector, mask, rot2, imat_inv, para) 
        
        print(str(comm_rank) + '--model3d--' + str(local_offset[comm_rank]) + '/' + str(idx) + '/' + str(local_offset[comm_rank + 1] - 1) + '----' + str(np.amax(model3d)) + '/' + str(np.amin(model3d)))
        print(str(comm_rank) + '--weight--' + str(local_offset[comm_rank]) + '/' + str(idx) + '/' + str(local_offset[comm_rank + 1] - 1) + '----' + str(np.amax(weight)) + '/' + str(np.amin(weight)) )
    

    data_receive = None   

    if (comm_rank == 0):

        print "rank 0 finished. time (s): ", time.time()-tic
        closed_workers += 1

        while(closed_workers < comm_size):
            data_receive = np.empty_like(model3d)
            comm.recv(comm_process, source=MPI.ANY_SOURCE, status = status)
            source = status.Get_source()        
            comm.send(None, dest = source)  
            print ('receiving data from ' + str(source))
            comm.Recv(data_receive, source = source, tag = 1, status = status)
            model3d += data_receive
            comm.Recv(data_receive, source = source, tag = 0, status = status)
            weight += data_receive

            comm.recv(comm_process, source=source)
            closed_workers += 1
            print('rank ' + str(source) + ' just finished...')

    else:
        comm.send(None, dest = 0)  
        comm.recv(comm_process, source=0)

        print (str(comm_rank) + ' is sending one model3d to rank 0...')
        comm.Send(model3d, dest = 0, tag = 1)
        print (str(comm_rank) + ' is sending one weight to rank 0...')
        comm.Send(weight, dest = 0, tag = 0)        
        comm.send(None, dest = 0)
        
    if comm_rank == 0:
        
        print "time (s): ", time.time()-tic 
        model3d = scaling(model3d, weight)
        
    index = np.where(weight<0.01)
    model3d[index] = -10000

        print ('start saving files... ')  
        f1 = h5py.File(path_intens,'w')
        f2 = h5py.File(path_weight,'w')
        
        data1 = f1.create_dataset('diff_x', samp_size, chunks = x_chunk, compression = 'gzip', compression_opts = 7)
        data2 = f1.create_dataset('diff_y', samp_size, chunks = y_chunk, compression = 'gzip', compression_opts = 7)
        data3 = f1.create_dataset('diff_z', samp_size, chunks = z_chunk, compression = 'gzip', compression_opts = 7)        
        w1 = f2.create_dataset('weight', samp_size, chunks = x_chunk, compression = 'gzip', compression_opts = 7)
        
        data1[...] = model3d
        data2[...] = model3d
        data3[...] = model3d
        f1.close() 

        w1[...] = weight               
        f2.close()
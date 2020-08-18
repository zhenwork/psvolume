import numpy as np 
from numba import jit
import mathTools



def image_2d_to_pixel_px(size=None, center=None):
    ## get [x,y] in pixel
    x = np.arange(size[0]) - center[0]
    y = np.arange(size[1]) - center[1]
    xaxis, yaxis = np.meshgrid(x, y, indexing="ij")
    return np.stack([xaxis,yaxis]).T


def pixel_px_to_realxyz_mm(pixel_px=None, pixel_size_um=None, detector_distance_mm=None):
    ## convert pixel position to [x,y,z] in mm
    xy_mm = pixel_px * pixel_size_um / 1000.
    z_mm = np.ones((nx,ny)) * detector_distance_mm
    return np.hstack([xy_mm, z_mm[:,None]])


def realxyz_mm_to_recixyz_invA(realxyz_mm=None, wavelength_A=None, rot_matrix=None):
    ## get reciprocal position, direct conversion + rotation of crystal
    if len(realxyz_mm.shape)==2:
        norm_mm = np.sqrt(np.sum(realxyz_mm**2, axis=1, keepdims=True)) 
    elif len(realxyz_mm.shape)==3:
        norm_mm = np.sqrt(np.sum(realxyz_mm**2, axis=2, keepdims=True)) 

    scaled = realxyz_mm/norm_mm
    scaled[:,2] -= 1.0
    reciprocal_invA = scaled / wavelength_A
    if rot_matrix is not None:
        return reciprocal_invA.dot(np.linalg.inv(rot_matrix).T)
    return reciprocal_invA 


def recixyz_invA_to_voxel_hkl(Amat_invA=None,recixyz_invA=None):
    """
    Return: voxel (N*N*3 or N*3)
    voxelSize: 0.015 for 'cartesian' coordinate; 1.0 for "hkl" coordinate
    """
    return recixyz_invA.dot(np.linalg.inv(Amat_invA).T) 


def image_2d_to_voxel_hkl(size=None, center=None, Amat_invA=None, phi_deg=0., rot_axis="x", \
                wavelength_A=None, pixel_size_um=None, center=None, detector_distance_mm=None):

    """
    # This function combines mapPixel2RealXYZ, mapRealXYZ2Reciprocal and mapReciprocal2Voxel. 
    # Input: real 2D image in N*N
    # Output: voxel in N*N*3 shape
    """
    pixel_px = image_2d_to_pixel_px(size=size, center=center)
    rot_matrix = dutils.quaternion_to_rotation(dutils.phi_to_quaternion(phi_deg=phi_deg, rotAxis=rot_axis))
    realxyz_mm = pixel_px_to_realxyz_mm(pixel_px=pixel_px,pixel_size_um=pixel_size_um,detector_distance_mm=detector_distance_mm)
    recixyz_invA = realxyz_mm_to_recixyz_invA(realxyz_mm=realxyz_mm, wavelength_A=wavelength_A, rot_matrix=rot_matrix)
    voxel_hkl = recixyz_invA_to_voxel_hkl(Amat_invA=Amat_invA,recixyz_invA=recixyz_invA)
    return voxel_hkl


@jit
def bragg_peak_mask(Amat_invA=None, size=None, center=None, detector_distance_mm=None,window_reject_hkl=(0, 0.25), \
                h_range_keep=(-1000,1000), k_range_keep=(-1000,1000), l_range_keep=(-1000,1000), \
                wavelength_A=None, pixel_size_um=None, phi_deg=0., rot_axis="x"):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    ## peakMask = 1 when pixels are in window
    """
    voxel_hkl = image_2d_to_voxel_hkl(size=size, center=center, Amat_invA=Amat_invA, phi_deg=phi_deg, rot_axis=rot_axis, \
                wavelength_A=wavelength_A, pixel_size_um=pixel_size_um, detector_distance_mm=detector_distance_mm)

    ## For Loop to map one image
    npixels = np.prod(size)
    peak_mask = np.zeros(npixels).astype(int) 
    shift_hkl = np.abs(np.around(voxel_hkl) - voxel_hkl)

    for t in range(npixels):
        
        hshift = shift_hkl[t, 0]
        kshift = shift_hkl[t, 1]
        lshift = shift_hkl[t, 2]

        hh = voxel_hkl[t, 0]
        kk = voxel_hkl[t, 1]
        ll = voxel_hkl[t, 2]

        if (hshift>=window_reject_hkl[1]) or (kshift>=window_reject_hkl[1]) or (lshift>=window_reject_hkl[1]):
            continue
        if (hshift<window_reject_hkl[0]) and (kshift<window_reject_hkl[0]) and (lshift<window_reject_hkl[0]):
            continue
        if hh < h_range_keep[0] or hh >= h_range_keep[1]:
            continue
        if kk < k_range_keep[0] or kk >= k_range_keep[1]:
            continue
        if ll < l_range_keep[0] or ll >= l_range_keep[1]:
            continue
        
        peak_mask[t] = 1

    return peak_mask.reshape(size)


@jit
def image_2d_to_volume_3d(volume=None, weight=None, Amat_invA=None, image=None, center=None, mask=None, \
                keep_peak=False, window_reject_hkl=(0, 0.25), wavelength_A=None, pixel_size_um=None, detector_distance_mm=None, \
                volume_center=60, volume_size=121, oversample=1, phi_deg=0., rot_axis="x"):
    """
    Method: pixels collected to nearest voxels
    returnFormat: "HKL" or "cartesian"
    voxelSize: unit is nm^-1 for "cartesian", NULL for "HKL" format 
    If you select "cartesian", you may like voxelSize=0.015 nm^-1
    """
    voxel = image_2d_to_voxel_hkl(size=size, center=center, Amat_invA=Amat_invA, phi_deg=phi_deg, rot_axis=rot_axis, \
                wavelength_A=wavelength_A, pixel_size_um=pixel_size_um, detector_distance_mm=detector_distance_mm)

    ## For Loop to map one image
    npixels = np.prod(image.shape)
    data  = image.ravel()
    mask  = mask.ravel()
    
    if volume is None:
        volume = np.zeros((volume_size, volume_size, volume_size))
        weight = np.zeros((volume_size, volume_size, volume_size))
        
    for t in range(npixels):

        if mask[t] == 0:
            continue
        
        hkl = voxel_hkl[t] + volume_center
        
        h_float = hkl[0] 
        k_float = hkl[1] 
        l_float = hkl[2] 

        h_assign = int(round(h_float * oversample)) + volume_center
        k_assign = int(round(k_float * oversample)) + volume_center
        l_assign = int(round(l_float * oversample)) + volume_center

        if (h_assign<0) or h_assign>(volume_size-1) or (k_assign<0) or k_assign>(volume_size-1) or (l_assign<0) or l_assign>(volume_size-1):
            continue
        
        h_bragg = int(round(h_float)) 
        k_bragg = int(round(k_float)) 
        l_bragg = int(round(l_float))

        h_shift = abs( h_float - h_bragg )
        k_shift = abs( k_float - k_bragg )
        l_shift = abs( l_float - l_bragg )

        ## window[0] < reject < window[1]
        if (h_shift>=window_reject_hkl[1]) or (k_shift>=window_reject_hkl[1]) or (l_shift>=window_reject_hkl[1]):
            pass
        elif (h_shift<window_reject_hkl[0]) and (k_shift<window_reject_hkl[0]) and (l_shift<window_reject_hkl[0]):
            pass
        else:
            continue
        
        weight[h_assign,k_assign,l_assign] += 1
        volume[h_assign,k_assign,l_assign] += data[t]

    return volume, weight


def image_2d_resolution_invA(size=None, wavelength_A=None, detector_distance_mm=None, detector_center_px=None, pixel_size_um=None):
    pixel_px = image_2d_to_pixel_px(size=size, center=center)
    rot_matrix = dutils.quaternion_to_rotation(dutils.phi_to_quaternion(phi_deg=phi_deg, rotAxis=rot_axis))
    realxyz_mm = pixel_px_to_realxyz_mm(pixel_px=pixel_px,pixel_size_um=pixel_size_um,detector_distance_mm=detector_distance_mm)
    recixyz_invA = realxyz_mm_to_recixyz_invA(realxyz_mm=realxyz_mm, wavelength_A=wavelength_A)
    rep_norm_invA = np.sqrt(np.sum(recixyz_invA**2, axis=2))
    return rep_norm_invA
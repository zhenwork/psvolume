import numpy as np
from scipy.ndimage.filters import median_filter


def solid_angle_scaler(size=None, beam_x_pix=None, beam_y_pix=None, \
            detector_distance_cm=None, x_pixel_size_mm=None, y_pixel_size_mm=None):
    """
    Params: detectorDistance, pixelSize must have the same unit
    Returns: scaleMask -> image *= scaleMask
    Note: scaleMask -> min=1 (at center), value increases to the detector edge
    """
    (nx, ny) = size
    (cx, cy) = (beam_x_pix, beam_y_pix)
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij") 
    xaxis = xaxis * x_pixel_size_mm
    yaxis = yaxis * y_pixel_size_mm
    zaxis = np.ones((nx,ny))*detector_distance_cm*10.0   ## cm -> mm
    norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    solidAngle = zaxis * 1.0 / norm**3
    solidAngle /= np.amax(solidAngle)
    scaleMask = 1./solidAngle
    return scaleMask

def polarization_scaler(size=None, polarization=-1, detector_distance_cm=None, \
            x_pixel_size_mm=None, y_pixel_size_mm=None, beam_x_pix=None, beam_y_pix=None):
    """
    p =1 means y polarization
    p=-1 means x polarization
    # Default is p=-1 (x polarization)
    # Note: scaleMask -> min=1
    """
    (nx, ny) = size
    (cx, cy) = (beam_x_pix, beam_y_pix)
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    [xaxis, yaxis] = np.meshgrid(x, y, indexing="ij") 
    xaxis = xaxis * x_pixel_size_mm
    yaxis = yaxis * y_pixel_size_mm
    zaxis = np.ones((nx,ny))*detector_distance_cm*10.0
    norm = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    
    if polarization is not None:
        detectScale = (2.*zaxis**2 + (1+polarization)*xaxis**2 + (1-polarization)*yaxis**2 )/(2.*norm**2)
    else: 
        detectScale = np.ones(size)

    detectScale /= np.amax(detectScale)
    scaleMask = 1. / detectScale
    return scaleMask
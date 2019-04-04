import math
import numpy as np 

def dotproduct(x, y):
    return sum((a*b) for a, b in zip(x, y))

def length(x):
    return math.sqrt(dotproduct(x, x))

def angle(x, y):
    """
    Returns value in degree.
    """
    return math.acos(dotproduct(x, y) / (length(x) * length(y))) * 180. / np.pi


def Euler2Rotation(alpha, beta, gamma):
    return None


def RandomRotation():
    return None


def phi2quaternion(phi, rotAxis="x"):
    """
    phi: in degree unit
    """
    angle = phi*np.pi/180.
    if rotAxis.lower() == "x":
        return (np.cos(angle/2.), np.sin(angle/2.), 0., 0.)
    elif rotAxis.lower() == "y":
        return (np.cos(angle/2.), 0., np.sin(angle/2.), 0.)
    elif rotAxis.lower() == "z":
        return (np.cos(angle/2.), 0., 0., np.sin(angle/2.))
    else:
        return None


def quaternion2rotation(quaternion):
    rot = np.zeros([3,3])
    (q0, q1, q2, q3) = quaternion
    q01 = q0*q1 
    q02 = q0*q2 
    q03 = q0*q3 
    q11 = q1*q1 
    q12 = q1*q2 
    q13 = q1*q3 
    q22 = q2*q2 
    q23 = q2*q3 
    q33 = q3*q3 

    rot[0, 0] = (1. - 2.*(q22 + q33)) 
    rot[0, 1] = 2.*(q12 - q03) 
    rot[0, 2] = 2.*(q13 + q02) 
    rot[1, 0] = 2.*(q12 + q03) 
    rot[1, 1] = (1. - 2.*(q11 + q33)) 
    rot[1, 2] = 2.*(q23 - q01) 
    rot[2, 0] = 2.*(q13 - q02) 
    rot[2, 1] = 2.*(q23 + q01) 
    rot[2, 2] = (1. - 2.*(q11 + q22)) 
    return rot

def meshgrid2D(size, center=None):
    (nx, ny) = size
    if center is None:
        cx = (nx-1.)/2.
        cy = (ny-1.)/2.
    else:
        (cx,cy) = center

    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    xaxis, yaxis = np.meshgrid(x, y, indexing='ij')
    return xaxis, yaxis

def meshgrid3D(size, center=None):
    (nx, ny, nz) = size
    if center is None:
        cx = (nx-1.)/2.
        cy = (ny-1.)/2.
        cz = (nz-1.)/2.
    else:
        (cx,cy,cz) = center

    x = np.arange(nx)-cx
    y = np.arange(ny)-cy
    z = np.arange(nz)-cz
    xaxis, yaxis, zaxis = np.meshgrid(x,y,z, indexing='ij')
    return xaxis, yaxis, zaxis

def make2DRadius(size, center=None):
    xaxis, yaxis = meshgrid2D(size, center=center)
    radius = np.sqrt(xaxis**2 + yaxis**2)
    return radius, xaxis, yaxis

def make3DRadius(size, center=None):
    xaxis, yaxis, zaxis = meshgrid2D(size, center=center)
    radius = np.sqrt(xaxis**2 + yaxis**2 + zaxis**2)
    return radius, xaxis, yaxis, zaxis


def eulerAngles2rotation(_theta, mode="rad"):
    """
    _theta = (angle 1, angle 2, angle 3)
    """
    if mode == "rad":
        theta = _theta
    else:
        theta = np.array(_theta)/180.0*np.pi

    Rx = np.array([[1,         0,                  0               ],
                   [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                   [0,         np.sin(theta[0]),  np.cos(theta[0]) ]])
         
    Ry = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                   [0,                   1,                    0   ],
                   [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]])
                 
    Rz = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),       0],
                   [np.sin(theta[2]),     np.cos(theta[2]),       0],
                   [0,                   0,                       1]])
                     
    R = np.dot(Rz, np.dot( Ry, Rx ))
    return R
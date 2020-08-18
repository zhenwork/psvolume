import math
import numpy as np 


def phi_to_quaternion(phi_deg, rotAxis="x"):
    """
    phi: in degree unit
    """
    angle = phi_deg*np.pi/180.
    if rotAxis.lower() == "x":
        return (np.cos(angle/2.), np.sin(angle/2.), 0., 0.)
    elif rotAxis.lower() == "y":
        return (np.cos(angle/2.), 0., np.sin(angle/2.), 0.)
    elif rotAxis.lower() == "z":
        return (np.cos(angle/2.), 0., 0., np.sin(angle/2.))
    else:
        return None


def quaternion_to_rotation(quaternion):
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


def euler_angle_to_rotation(theta_deg):
    """
    _theta = (angle 1, angle 2, angle 3)
    """
    theta = np.array(theta_deg)/180.0*np.pi

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
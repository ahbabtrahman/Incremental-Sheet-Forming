#oriented so that robot is facing the bottom of the frame
import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
#Units in mm   
#Jog robot to each corner of frame and obtain each coordinate

#Rig coordinates in base frame
def calibrate(c1,c2,c3,c4):
    '''
    
    Parameters
    ----------
    c1 : 3x1 vector
        x,y,z coordinates of the bottom left corner of the rig
    c2 : 3x1 vector
        x,y,z coordinates of the bottom right corner of the rig
    c3 : 3x1 vector
        x,y,z coordinates of the top left corner of the rig
    c4 : 3x1 vector
        x,y,z coordinates of the top right corner of the rig

    Returns
    -------
    Pbr : 3x1 numpy vector
        Position vector from the rig to the robot base
    qbr : 4x1 numpy vector
        Quaternion vector from the rig to the robot base

    '''
    
    #Define axis vectors
    v1 = (c4-c2)
    v2 = (c1-c2)
    
    #Unit Axis Vectors
    vx = v1/np.linalg.norm(v1)
    #Get rid of the component parallel to vx so that vx and vy are orthonormal
    vya = v2 - np.dot(v2,vx)*vx 
    vy = vya/np.linalg.norm(vya)
    vz = np.cross(vx,vy)
    
    #Rotation Matrix from rig to base
    Rbr = np.column_stack((vx,vy,vz))
    #quaternion
    qbr = R2q(Rbr)
    
    #Position Vector
    Pbr = c2
    #Homogenous matrix
    H = Htransform(Rbr, Pbr)
    np.savetxt("rig_pose_raw.csv", H, delimiter = ',')
    
    print("New raw rig pose is")
    print(H)
    
    return H

def Htransform(R,P):
    """

    Parameters
    ----------
    R : 3x3 numpy array
        Rotation Matrix
    P : 3x1 numpy array
        Position Vector

    Returns
    -------
    H : 4x4 numpy array
        Homogenous transform matrix

    """
    H = np.column_stack((R,P))
    lastrow = np.array([0,0,0,1])
    H = np.vstack((H,lastrow))
    return H
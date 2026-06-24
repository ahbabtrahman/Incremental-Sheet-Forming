import numpy as np
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
from matplotlib import pyplot as plt
import pandas as pd
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM

from traj_gen import get_trajectory

# helper functions 
def calc_lam_js(curve_js,robot):
    curve_p = [] # flange position
    for i in range(len(curve_js)):
        curve_p.append(fwdkin(robot, curve_js[i]).p)
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) # executed path length
    lam = np.insert(lam, 0, 0)
    return lam

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

Pft = np.array([-55.755, 0, 130.05])

tool_T = Transform(np.eye(3), Pft)

robot.R_tool=tool_T.R
robot.p_tool=tool_T.p

#Obtain waypoints from excel sheet
filename = "NewWaypoints.xlsx"# replace with file path, keep the r"" though or else funky errors
sheet_name = "FinalData"  # Replace with your sheet name
data = pd.read_excel(filename, sheet_name=sheet_name, header=None)

# Convert to NumPy array
points = data.to_numpy()
print(points)
tempPoints = points[:, :3]

#lowest
minimum = -0.32
increm = 0.04

minval = 0
height = 0
positions = []
for i in range(len(tempPoints)):
    if tempPoints[i][2] < minval:
        minval = tempPoints[i][2]

#initializes the set of points as a list of lines
tempPos = []
startPoint = tempPoints[0]
line = -1
lineList = []
for i in range(len(tempPoints)):
    #print(startPoint[1], tempPoints[i][1])
    if tempPoints[i][1] == startPoint[1] and tempPoints[i][2] == 10:
        line += 1
        lineList.append([[tempPoints[i][0], tempPoints[i][1], tempPoints[i][2], line]])
    else:
        #print(line)
        lineList[line].append([tempPoints[i][0], tempPoints[i][1], tempPoints[i][2], line])
#reverses every other line
for i in range(len(lineList)):
    if i % 2 == 0:
        lineList[i].reverse()

#converts the list of lines to a list of points
for i in range(len(lineList)):
    for j in range(len(lineList[i])):
        tempPos.append(lineList[i][j])
for i in range(len(lineList)):
    lineList[i].reverse()
    for j in range(len(lineList[i])):
        tempPos.append(lineList[i][j])

#converts the final points to multiple passes switching which set of points each time
j=0
while height >= minimum:
    for i in range(len(tempPos)//2):
        if tempPos[i + j][2] < 0 and height > tempPos[i + j][2]:
            positions.append([tempPos[i + j][0], tempPos[i + j][1], height])
        else:
            positions.append([tempPos[i + j][0], tempPos[i + j][1], tempPos[i + j][2]])
    j += len(tempPos)//2
    height += -increm
    if j > len(tempPos)//2:
        j = 0
if height + increm > minimum and height != minimum:
    for i in range(len(tempPos)):
        positions.append([tempPos[i][0], tempPos[i][1], tempPos[i][2]])

dlam_des = 0.05
corner_p_wp = points
#print(corner_p_wp.shape)
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

curve_p = []
curve_R = []
for i in range(corner_p_wp.shape[0]-1):
    this_seg_p = np.linspace(corner_p_wp[i], corner_p_wp[i+1], int(np.linalg.norm(corner_p_wp[i]-corner_p_wp[i+1])/dlam_des)+1)
    this_seg_R = np.tile(corner_R, (len(this_seg_p), 1, 1))
    curve_p.extend(this_seg_p[:-1])
    curve_R.extend(this_seg_R[:-1])
curve_p.append(corner_p_wp[-1])
curve_R.append(corner_R)
curve_p = np.array(curve_p)
curve_R = np.array(curve_R)

#############################################Rig Kinematics###################################################
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

# Run it on the robot
qbr = R2q(Rbr)
################################################################################################
#Transformation to rig
for i in range(0,curve_p.shape[0]):
    curve_pbr = Rbr@curve_p[i,:] + Pbr
    curve_p[i] = deepcopy(curve_pbr)

new_curve_R = []
for i in range(0,curve_R.shape[0]):
     Rcalc = Rbr@curve_R[i]
     new_curve_R.append(Rcalc)

## Planning a curve js using 5 dof constraints (position + 2 dof normal vector)
# get the initial joint angles

q_init = robot6_sphericalwrist_invkin(robot,Transform(new_curve_R[0], curve_p[0]),np.zeros(6))[0]

q_init = robot6_sphericalwrist_invkin(robot,Transform(new_curve_R[0], curve_p[0]),np.zeros(6))[0]

terminate_threshold = 0.001
angle_weight=1
alpha=1 # we can use a line search to find the best alpha. Here we use a fixed alpha.

# get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
curve_js = [q_init]
for i in range(1,len(curve_p)):
    print("-----------------")
    print(f"{i} out of {len(curve_p)}")
    print("Rotation")
    print(new_curve_R[i])
    print("Pose")
    print(curve_p[i])
    this_q = robot6_sphericalwrist_invkin(robot,Transform(new_curve_R[i], curve_p[i]),curve_js[-1])[0]
    flange_T = fwdkin(robot, this_q)
    vd = curve_p[i]-flange_T.p

    assert np.linalg.norm(vd) < terminate_threshold, "The inverse kinematics gave a large error"
    curve_js.append(this_q)
print(type(curve_js))
np.savetxt("curve_js.csv", curve_js, delimiter=',')
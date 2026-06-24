import time

import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from matplotlib import pyplot as plt
from calibration import calibrate

import pandas as pd

final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]

Rbr = final_rig_pose[0:3, 0:3]@Rz

Pft = np.array([-55.755, 0, 130.05])

tool_T = Transform(np.eye(3), Pft)

# Run it on the robot
qbr = R2q(Rbr)

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

#Measure the z and y displacement from the flange to the tool tip
my_tool = abb.tooldata(True,abb.pose(tool_T.p,R2q(tool_T.R)),abb.loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0))

#for run on sheet metal
my_wobj = abb.wobjdata(False,True,"",abb.pose(Pbr + [7,0,0],qbr),abb.pose([0,0,0],[1,0,0,0]))#callibration work object
#for run in space
#my_wobj = abb.wobjdata(False,True,"",abb.pose([592.98,61.32,297.24],[1,0,0,0]),abb.pose([0,0,0],[1,0,0,0]))

def quadrant(q,robot):
	cf146=np.floor(np.array([q[0],q[3],q[5]])/(np.pi/2))
	eef=fwdkin(robot,q).p
	
	REAR=(1-np.sign((rot([0,0,1],q[0])@np.array([1,0,0]))@np.array([eef[0],eef[1],eef[2]])))/2

	LOWERARM= q[2]<-np.pi/2
	FLIP= q[4]<0

	return np.hstack((cf146,[4*REAR+2*LOWERARM+FLIP])).astype(int)

mp = abb.MotionProgram(tool=my_tool,wobj=my_wobj)

### You may want to use moveabsj to move to the initial joint angles
### Provide the initial joint angles
# jt_init = abb.jointtarget(np.degrees(curve_js[0]),[0]*6) # 
# mp.MoveAbsJ(jt_init,abb.v1000,abb.fine)
###

# Rotation are all np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
#Points 0 is -1

#Check origin
'''
p1 = [0,0,50] 

corner_p = np.array([p1])
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
for i in range(len(corner_p)):
    robt = abb.robtarget(corner_p[i],R2q(corner_R),abb.confdata(0,-1,-1,0),[0]*6) # create the robtarget, position (mm), orientation (quaternion)
    if i==0 or i==4:
        mp.MoveL(robt,abb.v5,abb.fine) # last zone is fine
    else:
        mp.MoveL(robt,abb.v5,abb.z50) # using zone = z50

print("Robot start moving")

#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results
exit()
'''
filename = "PathpointsforSheetMetal.xlsx"# replace with file path, keep the r"" though or else funky errors
sheet_name = "Reversed"  # Replace with your sheet name
#sheet_name = "HorizontalLineTest"
data = pd.read_excel(filename, sheet_name=sheet_name, header=None)

# Convert to NumPy array
points = data.to_numpy()
#print(points)
tempPos = points[:, :3]
#print(tempPos)

#lowest
minimum = -0.15
increm = 0.05

minval = 0
height = 0
positions = []
for i in range(len(tempPos)):
    #reverse and offset the points
    tempPos[i] = [tempPos[i][0], 80.68 - tempPos[i][1], -tempPos[i][2]]
    if tempPos[i][2] < minval:
        minval = tempPos[i][2]


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


corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
for i in range(len(positions)):
    robt = abb.robtarget(positions[i], R2q(corner_R), abb.confdata(0, -1, -1, 0), [0]*6)
    mp.MoveL(robt, abb.v5, abb.fine)

print(positions)
print("Robot start moving")

#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results

exit()

cmd_num=log_results.data[:,1] # command number
start_id = np.where(cmd_num==2)[0][0]
curve_js_exe = np.radians(log_results.data[start_id:,2:8]) # executed joint angles
curve_p_exe = [] # executed flange position
for i in range(len(curve_js_exe)):
    curve_p_exe.append(fwdkin(robot, curve_js_exe[i]).p)
lam_exe = np.cumsum(np.linalg.norm(np.diff(curve_p_exe, axis=0), axis=1)) # executed path length
lam_exe = np.insert(lam_exe, 0, 0)

# visualize the joint trajectory
fig = plt.figure()
for i in range(6):
    plt.plot(lam_exe,curve_js_exe[:,i], label=f'Joint {i+1} executed', linestyle='--')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle (rad)')
plt.title('Joint trajectory')
plt.show()

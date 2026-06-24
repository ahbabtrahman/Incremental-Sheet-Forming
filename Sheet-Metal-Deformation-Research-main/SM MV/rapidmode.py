import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from matplotlib import pyplot as plt
from calibration import calibrate

#bottom left
c1 = np.array([549.03,139.03,329.79])
#bottom right --> Po
c2 = np.array([549.91,56.94,329.69])
#top left
c3 = np.array([645.7,140.98,327.6])
#top right
c4 = np.array([645.68,58.10,328.19])

Pbr, qbr = calibrate(c1,c2,c3,c4)

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

#Flange to tool
tool_T = Transform(np.eye(3), np.array([48.5, 0, 115]))
#Input offset from EE center

# Run it on the robot
my_tool = abb.tooldata(True,abb.pose(tool_T.p,R2q(tool_T.R)),abb.loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0))
my_wobj = abb.wobjdata(False,True,"",abb.pose(Pbr,qbr),abb.pose([0,0,0],[1,0,0,0]))
#Position Vector, quaternion from rotation matrix
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

#Straight Line Testing Paths
# Rotation are all np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
p1 = [-25, -25, 0]
p2 = [-25, 55, 0]
p3 = [25, -55, 0]
p4 = [25, -55, 50]
p5 = [25, -25, 50]
corner_p = np.array([p1, p2, p3, p4, p5])
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
for i in range(5):
    robt = abb.robtarget(corner_p[i],R2q(corner_R),abb.confdata(0,-1,-1,0),[0]*6) # create the robtarget, position (mm), orientation (quaternion)
    if i==0 or i==4:
        mp.MoveL(robt,abb.v10,abb.fine) # last zone is fine
    else:
        mp.MoveL(robt,abb.v10,abb.z50) # using zone = z50

print("Robot start moving")

client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
#client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results

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
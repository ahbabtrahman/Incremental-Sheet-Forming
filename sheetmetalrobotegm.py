import numpy as np
import pandas as pd
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
from matplotlib import pyplot as plt
import sys
import time

import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM
sys.path.append('toolbox')
# from robot_def import *
# from utils import *
# from rpi_ati_net_ft import *
# sys.path.append('robot_motion')
# from RobotMotionController import *

from traj_gen import get_trajectory
TIMESTEP = .004

# helper functions 
def calc_lam_js(curve_js,robot):
    #curve_js is the list of joint angles, Nx1
    curve_p = [] # flange position
    for i in range(len(curve_js)):
        curve_p.append(fwdkin(robot, curve_js[i]).p) #Calculate pose for each set of joint angles
    #???
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) # executed path length. Takes the diff btwn each joint angle set, find the frobenius norm of that vector, and then compute cumulative sum 
    lam = np.insert(lam, 0, 0)
    return lam

# generate a SMOOTH trajectory given the curve joint path        
def trajectory_generate(curve_js,robot,lin_vel,lin_acc):
    
    # Calculate the path length
    lam = calc_lam_js(curve_js, robot)
    
    # find the time stamp for each segment, with acceleration and deceleration
    if len(lam)>2 and lin_acc>0:
        time_bp = np.zeros_like(lam)
        acc = lin_acc
        vel = 0
        for i in range(0,len(lam)):
            #print(f"{i} out of {len(lam)}")
            if vel>=lin_vel:
                time_bp[i] = time_bp[i-1]+(lam[i]-lam[i-1])/lin_vel
            else:
                time_bp[i] = np.sqrt(2*lam[i]/acc)
                vel = acc*time_bp[i]
        time_bp_half = []
        vel = 0
        for i in range(len(lam)-1,-1,-1):
            #print(f"{i} out of the {len(lam)}")
            if vel>=lin_vel or i<=len(lam)/2:
                break
            else:
                time_bp_half.append(np.sqrt(2*(lam[-1]-lam[i])/acc))
                vel = acc*time_bp_half[-1]
        time_bp_half = np.array(time_bp_half)[::-1]
        time_bp_half = time_bp_half*-1+time_bp_half[0]
        time_bp[-len(time_bp_half):] = time_bp[-len(time_bp_half)-1]+time_bp_half\
            +(lam[-len(time_bp_half)]-lam[-len(time_bp_half)-1])/lin_vel
    else:
        time_bp = lam/lin_vel
    
    # total time stamps
    time_stamps = np.arange(0,time_bp[-1],TIMESTEP)
    time_stamps = np.append(time_stamps,time_bp[-1])

    # Calculate the number of steps for the trajectory
    num_steps = int(time_bp[-1] / TIMESTEP)

    # Initialize the trajectory list
    traj_q = []

    # Generate the trajectory
    for step in range(num_steps):
        # Calculate the current time
        current_time = step * TIMESTEP
        # Find the current segment
        for i in range(len(time_bp)-1):
            if current_time >= time_bp[i] and current_time < time_bp[i+1]:
                seg = i
                break
        # Calculate the fraction of time within the current segment
        frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
        # Calculate the desired joint position for the current step
        q_des = frac * curve_js[seg+1] + (1 - frac) * curve_js[seg]
        # Append the desired position to the trajectory
        traj_q.append(q_des)
        
    return np.array(traj_q), np.array(time_bp)

# read current joint position using egm
def read_position(egm):
    for i in range(0,20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if not res:
            #raise Exception("Robot communication lost")
            client.stop_egm() # stop egm
            mm = abb.egm_minmax(-1e-3,1e-3)
            mp = abb.MotionProgram(egm_config = egm_config)
            mp.EGMRunJoint(10, 0.05, 0.05)
            client.stop_motion_program()
            lognum = client.execute_motion_program(mp, wait=False)
            print("Communication Lost, Reconnecting")
        else:
            break

    return np.radians(state.joint_angles)

# send the next joint position to the robot using egm
def position_cmd(q,egm):
    egm.send_to_robot(np.degrees(q))

############################################# Rig Kinematics ###################################################
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


# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

#Define Tool Offset
Pft = np.array([-55.755, 0, 130.05])
tool_T = Transform(np.eye(3), Pft)

robot.R_tool=tool_T.R
robot.p_tool=tool_T.p

fullruntraj_q = np.loadtxt("fullruntraj_q.csv", delimiter = ',')
fullruntime_bp = np.loadtxt("fullruntime_bp.csv", delimiter = ',')
curve_js_exe = np.loadtxt("curve_js_exe.csv", delimiter = ',').tolist()
curve_js = np.loadtxt("curve_js.csv", delimiter = ',').tolist()
lam_planned = np.loadtxt("lam_planned.csv", delimiter = ',')
lam_planned = lam_planned[0]

test = np.zeros((fullruntraj_q.shape[0], 3))
for i in range(0,fullruntraj_q.shape[0]):
    test = fwdkin(robot,fullruntraj_q[i])
    pos_base = test.p
    rot_base = test.R

    pos_rig =  (Rbr.T)@(pos_base-Pbr)#Rig Frame



# set up egm config
mm = abb.egm_minmax(-1e-3,1e-3)
egm_config = abb.EGMJointTargetConfig(
    mm, mm, mm, mm, mm ,mm, 1000, 1000
)
mp = abb.MotionProgram(egm_config = egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)
#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
lognum = client.execute_motion_program(mp, wait=False)
egm = EGM()

# Run it on the robot using EGM

print("Robot start moving. EGM is running")

tool_vel = 5
tool_acc = 5
#first jog the robot to the initial position
q_start=read_position(egm)
q_all = np.linspace(q_start,curve_js[0],num=100)
traj_q, time_bp=trajectory_generate(q_all,robot,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
for i in range(len(traj_q)):
    read_q = read_position(egm) # reading joint position take approximately 4 ms
    position_cmd(traj_q[i],egm)
print("Robot in Start Position")

total_depth = -.32
inc = .04
steps = -int(total_depth // inc) #make sure is positive and integer
depth = 0 #Current depth
print_count = 0
offset = 1.5 #Find manually through checking 0 offset
#startheight = 1.5 #Defined in waypoints sheet

rig_data = []
for j in range(0,steps):
    depth = -j*inc
    print(f"Current depth is {depth}")
    print(f"Currently on pass {j+1}")
    for i in range(len(fullruntraj_q)):
        #print_count +=1
        read_q = read_position(egm) # reading joint position take approximately 4 ms
        #Position Adjustment
        #Grab pos, rot from joints to be sent to robot
        t_base = fwdkin(robot, fullruntraj_q[i])
        pos_base = t_base.p
        rot_base = t_base.R

        pos_rig =  (Rbr.T)@(pos_base-Pbr)#Rig Frame
        #task_rot = fwdkin(robot, fullruntraj_q[i]).R
        pos_rig[2] += depth + offset

        new_pos_base= Rbr@pos_rig + Pbr #Base Frame
        #curve_pbr = Rbr@curve_p[i,:] + Pbr --> Original transformation to base frame
        rig_data.append(deepcopy(pos_rig))

        #Find joints to send to robot
        run_q = robot6_sphericalwrist_invkin(robot,Transform(rot_base, new_pos_base),np.zeros(6))[0]
        curve_js_exe.append(read_q)
        #Move robot
        position_cmd(run_q,egm)

client.stop_egm() # stop egm
np.savetxt("transformeddata.csv", rig_data, delimiter = ',')

curve_js_rig = []
for i in range(len(curve_js_exe)):
        #Grab pos, rot from joints to be sent to robot
        t_base = fwdkin(robot, curve_js_exe[i])
        pos_base = t_base.p
        rot_base = t_base.R

        pos_rig =(Rbr.T)@(pos_base-Pbr) #Rig Frame
        curve_js_rig.append(deepcopy(pos_rig))

np.savetxt("liverigrobotpos.csv", curve_js_rig, delimiter = ',')

curve_js_exe = np.array(curve_js_exe) # executed joint angles
lam_exe = calc_lam_js(curve_js_exe, robot) # executed path length

# visualize the joint trajectory
curve_js = np.array(curve_js)
fig = plt.figure()
for i in range(6):
    plt.plot(lam_planned,curve_js[:,i], label=f'Joint {i+1} plannes')
    plt.plot(lam_exe,curve_js_exe[:,i], label=f'Joint {i+1} executed', linestyle='--')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle (rad)')
plt.title('Joint trajectory')
plt.show()

    
   
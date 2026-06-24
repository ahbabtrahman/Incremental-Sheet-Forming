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
        print(f"{step} out of the {(num_steps)} steps")
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

def generate_robot_joints_along_path(robot, tool_T, dlam_des, waypoints, Rbr, Pbr):

    robot.R_tool=tool_T.R
    robot.p_tool=tool_T.p

    corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

    curve_p = [] #Position
    curve_R = [] #Rotation
    for i in range(waypoints.shape[0] - 1): #number of waypoints -1
        this_seg_p = np.linspace(waypoints[i], waypoints[i+1], int(np.linalg.norm(waypoints[i]-waypoints[i+1])/dlam_des)+1) #Increment the path between current and next waypoint as integer. Why add +1?
        this_seg_R = np.tile(corner_R, (len(this_seg_p), 1, 1)) #Rotation doesn't change, repeat for each segment between the waypoints
        curve_p.extend(this_seg_p[:-1]) #Add all segments except for the next waypoints to end of position list
        curve_R.extend(this_seg_R[:-1]) #Add all rotations except for the next waypoint's rotation to end of rotation list
    curve_p.append(waypoints[-1]) #Add last position from originl list
    curve_R.append(corner_R) #Add rotation matrix to end of list for last waypoint
    curve_p = np.array(curve_p) #convert to Numpy array
    curve_R = np.array(curve_R) #convert to Numpy array

    print(curve_p)

    #Transformation rig to base
    for i in range(0,curve_p.shape[0]):
        curve_pbr = Rbr@curve_p[i,:] + Pbr
        curve_p[i] = deepcopy(curve_pbr)

    new_curve_R = []
    for i in range(0,curve_R.shape[0]):
        Rcalc = Rbr@curve_R[i]
        new_curve_R.append(Rcalc)

    ## Planning a curve js using 5 dof constraints (position + 2 dof normal vector)
    # get the initial joint angles
    q_init = robot6_sphericalwrist_invkin(robot,Transform(curve_R[0], curve_p[0]),np.zeros(6))[0]

    terminate_threshold = 0.001

    # get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
    curve_js = [q_init] #Stick the initial joint positions into the joint list
    st = time.perf_counter()
    #For every waypoint generated from before, find the list of joint angles to send to the robot
    for i in range(1,len(curve_p)):
        #print(f"{i} out of {len(curve_p)}")
        this_q = robot6_sphericalwrist_invkin(robot,Transform(curve_R[i], curve_p[i]),curve_js[-1])[0] #Solve for the joint angle (desired pose) per increment using inverse kinmeatics. [0] grabs the joint angles closest to the current joint angles.
        flange_T = fwdkin(robot, this_q) #Position of the robot flange at each set of joints using forward kinematics
        vd = curve_p[i]-flange_T.p #If the given position is exactly the same as the last position, the kinematics failed?
        assert np.linalg.norm(vd) < terminate_threshold, "The inverse kinematics gave a large error"
        curve_js.append(this_q) #Add joint angles to joint list
    lam_planned = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) #Find the path length
    lam_planned = np.insert(lam_planned, 0, 0) #??
    print((time.perf_counter() - st)/len(lam_planned))
    return lam_planned, curve_js

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

rigdim = 85 #Inner width of rig
rigoffset = rigdim - 45

#Define Tool Offset
Pft = np.array([-55.755, 0, 130.05])
tool_T = Transform(np.eye(3), Pft)

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


#################################################### Grab Waypoints from Excel #################
lines = ["line2", "line3", "line4", "line5", "line6", "line7"]
filename = "finalwaypoints2.xlsx"# replace with file path, keep the r"" though or else funky errors

sheet_name = "line1"  # Replace with your sheet name
data = (pd.read_excel(filename, sheet_name=sheet_name, header=None)).to_numpy()

waypoints_original = data[:,0:3]

for path in lines:
    print(f"Reading sheet {path}")
    sheet_name = path  # Replace with your sheet name
    data = (pd.read_excel(filename, sheet_name=sheet_name, header=None)).to_numpy()
    # Convert to NumPy array
    waypoints_original = np.append(waypoints_original, data[:,0:3], axis = 0)

waypoints_original = np.append(waypoints_original, np.reshape(waypoints_original[0,0:3], (1,3)), axis = 0)

#Shift X,Y points such that origin is geometry is centered within rig
for i in range(0, waypoints_original.shape[0]):
    waypoints_original[i,0] += rigoffset
    waypoints_original[i,1] += rigoffset
np.savetxt("testfile.csv",waypoints_original,delimiter=',')
################################################################################################

#Test Waypoints - Lines
#Spaced out lines
# p0 = [10, 10, 10]
# p1 = [10, 10, -.1]
# p2 = [10, 50, -.1]
# p3 = [10, 50, 10]
# p4 = [10.2, 50, 10]
# p5 = [10.2, 50, -.1]
# p6 = [10.2, 10, -.1]
# p7 = [10.2, 10, 10]
# waypoints_original = np.array([p0, p1, p2, p3, p4, p5, p6])
# print(waypoints_original.shape)

#Increment along path
dlam_des = .1
lam_planned, curve_js = generate_robot_joints_along_path(robot, tool_T, dlam_des, waypoints_original, Rbr, Pbr)
print(lam_planned)
############# Execute the trajectory on the robot #############
TIMESTEP = 0.004 # 4 ms for egm control

# robot tool velocity and acceleration 
tool_vel = 5 # mm/s
tool_acc = 5 # mm/s^2

# execute the trajectory
fullruntraj_q, fullruntime_bp=trajectory_generate(curve_js,robot,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
curve_js_exe = []
print("Trajectory Generated")

np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=',')
np.savetxt("fullruntime_bp.csv", fullruntime_bp, delimiter = ',')
np.savetxt("curve_js.csv", fullruntraj_q, delimiter=',')
np.savetxt("curve_js_exe.csv", fullruntraj_q, delimiter=',')
np.savetxt("lam_planned.csv", lam_planned, delimiter = ',')


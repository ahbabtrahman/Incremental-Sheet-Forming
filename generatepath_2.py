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

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from sklearn.decomposition import PCA
from calibration import *

sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
sys.path.append('robot_motion')
from RobotMotionController import *
from robot_def import *

from traj_gen import get_trajectory

# helper functions 
def calc_lam_js(curve_js,mctrl):
    #curve_js is the list of joint angles, Nx1
    curve_p = [] # flange position
    for i in range(len(curve_js)):
        curve_p.append(mctrl.robot.fwd(curve_js[i]).p) #Calculate pose for each set of joint angles
    #???
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) # executed path length. Takes the diff btwn each joint angle set, find the frobenius norm of that vector, and then compute cumulative sum 
    lam = np.insert(lam, 0, 0)
    return lam

# generate a SMOOTH trajectory given the curve joint path        
def trajectory_generate(curve_js,mctrl,lin_vel,lin_acc):

    pos_rig_list = []
    
    # Calculate the path lengthmctrl
    lam = calc_lam_js(curve_js, mctrl)
    
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
    counter = 0
    # Generate the trajectory
    for step in range(num_steps):
        counter += 1
        if counter % 10000 == 0:
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
        pos_base = mctrl.robot.fwd(q_des).p
        pos_rig =  (Rbr.T)@(pos_base-Pbr)
        traj_q.append(q_des)
        pos_rig_list.append(pos_rig)
        
    return np.array(traj_q), np.array(time_bp), np.array(pos_rig_list)

def generate_robot_joints_along_path(mctrl, tool_T, dlam_des, waypoints, Rbr, Pbr):

    mctrl.robot.R_tool=tool_T.R
    mctrl.robot.p_tool=tool_T.p

    corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T #Rotation of pi about y axis

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
    print("curve_p")
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
    q_init = mctrl.robot.inv(Transform(curve_R[i], curve_p[i]).p, Transform(curve_R[i], curve_p[i]).R,np.zeros(6))[0]

    terminate_threshold = 0.001

    # get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
    curve_js = [q_init] #Stick the initial joint positions into the joint list
    st = time.perf_counter()
    #For every waypoint generated from before, find the list of joint angles to send to the robot
    for i in range(1,len(curve_p)):
        #print(f"{i} out of {len(curve_p)}")
        this_q = mctrl.robot.inv(Transform(curve_R[i], curve_p[i]).p, Transform(curve_R[i], curve_p[i]).R,curve_js[-1])[0] #Solve for the joint angle (desired pose) per increment using inverse kinmeatics. [0] grabs the joint angles closest to the current joint angles.
        flange_T = mctrl.robot.fwd(this_q) #Position of the robot flange at each set of joints using forward kinematics
        vd = curve_p[i]-flange_T.p #If the given position is exactly the same as the last position, the kinematics failed?
        assert np.linalg.norm(vd) < terminate_threshold, "The inverse kinematics gave a large error"
        curve_js.append(this_q) #Add joint angles to joint list
    lam_planned = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) #Find the path length
    lam_planned = np.insert(lam_planned, 0, 0) #??
    print((time.perf_counter() - st)/len(lam_planned))
    return lam_planned, curve_js

############################################# Rig Kinematics ###################################################
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]

# Run it on the robot
qbr = R2q(Rbr)
################################################################################################

H_pentip2ati=np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip=np.linalg.inv(H_pentip2ati)
ad_ati2pentip=adjoint_map(Transform(H_ati2pentip[:3,:3],H_ati2pentip[:3,-1]))
ad_ati2pentip_T=ad_ati2pentip.T

RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')


#Define Tool Offset
Pft = np.array([-55.755, 0, 130.05])
tool_T = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T, delimiter = ',')

robot=robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",tool_file_path="rig_pen.csv")
q_seed=np.zeros(6)

tool_T = Transform(np.eye(3), Pft)

robot.R_tool=tool_T.R
robot.p_tool=tool_T.p


abb_robot_ip = '192.168.60.101'
TIMESTEP=0.004
controller_params = {
    "force_ctrl_damping": 60.0, # 200, 180, 90, 60
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 6.0, # 10 Unit: mm/sec
    "moveL_acc_lin": 7.2, # Unit: mm/sec^2 0.6, 1.2, 3.6
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 20.0, # Unit mm/sec 10
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 0.2, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec, 0.02
    "jogging_speed": 50, # Unit: mm/sec
    "jogging_acc": 10, # Unit: mm/sec^2
    'force_filter_alpha': 0.9 # force low pass filter alpha
    }


rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')
mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=9,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

rigoffset = 10

#################################################### Grab Waypoints from Excel #################
lines = ["gcode_coordinates_new"]
filename = "gcode_coordinates_new.xlsx"

sheet_name = "gcode_coordinates_new"  # Replace with your sheet name
data = (pd.read_excel(filename, sheet_name=sheet_name, header=None)).to_numpy()

waypoints_original = data[:,0:3]

# for path in lines:
#     print(f"Reading sheet {path}")
#     sheet_name = path  # Replace with your sheet name
#     data = (pd.read_excel(filename, sheet_name=sheet_name, header=None)).to_numpy()
#     # Convert to NumPy array
#     waypoints_original = np.append(waypoints_original, data[:,0:3], axis = 0)

# waypoints_original = np.append(waypoints_original, np.reshape(waypoints_original[0,0:3], (1,3)), axis = 0)

#Shift X,Y points such that origin is geometry is centered within rig
for i in range(0, waypoints_original.shape[0]):
    waypoints_original[i,0] += rigoffset
    waypoints_original[i,1] += rigoffset
np.savetxt("gcode_coordinates.csv",waypoints_original,delimiter=',')
################################################################################################


#Increment along path
dlam_des = .07 #Try not to set higher than .1, curves tend to disappear in generated trajectory
lam_planned, curve_js = generate_robot_joints_along_path(mctrl,tool_T, dlam_des, waypoints_original, Rbr, Pbr)
print(lam_planned)
############# Execute the trajectory on the robot #############
TIMESTEP = 0.004 # 4 ms for egm control

# robot tool velocity and acceleration (Setting speed faster can reduce number of points)
tool_vel = 4 # mm/s
tool_acc = 2 # mm/s^2

# execute the trajectory
fullruntraj_q, fullruntime_bp, pos_check =trajectory_generate(curve_js,mctrl,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
curve_js_exe = []
print("Trajectory Generated")

np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=',')
np.savetxt("fullruntime_bp.csv", fullruntime_bp, delimiter = ',')
np.savetxt("curve_js.csv", fullruntraj_q, delimiter=',')
np.savetxt("curve_js_exe.csv", fullruntraj_q, delimiter=',')
np.savetxt("lam_planned.csv", lam_planned, delimiter = ',')
np.savetxt("pos_check.csv", pos_check, delimiter = ',')
print("Data saved")
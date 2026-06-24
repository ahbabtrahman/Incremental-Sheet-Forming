#///////////////////////////////////////////////////////////
# Editor(s): Ahbab R. 
# Last Edited: 04/03/2026
# Notes:
# This script is based on zcalib.py. It draws a star using
# mctrl. While running, it outputs and saves the force 
# data. After running, it prints out the path taken and 
# force vs time plot.
#
# Change IP at Line 200
#
# Assumptions: 
# Z-Calibration has already been run
#///////////////////////////////////////////////////////////


# Import Statements
import sys
sys.path.append("/home/fusing-ubuntu/Sheet-Metal-Deformation-Research/SM MV/")
from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from sklearn.decomposition import PCA
from general_robotics_toolbox import *
from calibration import * # NOT pip
from utils import * # NOT pip
from robot_def import * # NOT pip
#from rpi_ati_net_ft import * # NOT pip
from RobotMotionController import * # NOT pip
import time
from copy import deepcopy
# End of Import Statements

#////////////////////////////#

# Helper Functions
def calc_lam_js(curve_js, mctrl):
	curve_p = []
	for i in range(len(curve_js)):
		curve_p.append(mctrl.robot.fwd(curve_js[i]).p)
	curve_p = np.array(curve_p)
	lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
	lam = np.insert(lam, 0, 0)
	return lam

def trajectory_generate(curve_js,mctrl,lin_vel,lin_acc):

    pos_rig_list = []
    
    # Calculate the path length
    lam = calc_lam_js(curve_js, mctrl)
    
    # find the time stamp for each segment, with acceleration and deceleration
    if len(lam)>2 and lin_acc>0:
        time_bp = np.zeros_like(lam)
        acc = lin_acc
        vel = 0
        for i in range(0,len(lam)):
            if vel>=lin_vel:
                time_bp[i] = time_bp[i-1]+(lam[i]-lam[i-1])/lin_vel
            else:
                time_bp[i] = np.sqrt(2*lam[i]/acc)
                vel = acc*time_bp[i]
        time_bp_half = []
        vel = 0
        for i in range(len(lam)-1,-1,-1):
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


def read_position(mctrl):
	for _ in range(20):
		res, state = mctrl.egm.receive_from_robot(timeout = 0.1)
		if res:
			return np.radians(state.joint_angles)

	mctrl.stop_egm()
	mctrl.start_egm()
	raise Exception("EGM communication lost")

# End of Helper Functions 

#//////////////////////////////#

# Force-Torque Sensor Connection
H_pentip2ati=np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip=np.linalg.inv(H_pentip2ati)
ad_ati2pentip=adjoint_map(Transform(H_ati2pentip[:3,:3],H_ati2pentip[:3,-1]))
ad_ati2pentip_T=ad_ati2pentip.T
# Establish connection with the Force Sensor
RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')
# End of Force-Torque Sensor Connection

#/////////////////////////////////#

# Robot Configuration Parameterers #
# MAKE SURE THIS IS RIGHT
# Measure the z and y displacement from the flange to the tool tip
Pft = np.array([-55.45, 0, 131.2])
# x to inside of pen holder: -49.55
# z to end of pen holder 92.4

# Testing
print(Htransform.__module__)

tool_T = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T, delimiter = ',')

robot=robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",tool_file_path="rig_pen.csv")
q_seed=np.zeros(6)

# First Matrix is rotational of tool to flange
# Second Matrix is position offset of tool to flange
print(robot.R_tool,robot.p_tool)

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

# End of Robot Configuration Paramameters #

#////////////////////////////////#

# Star Coordinates
z_dir = -1.2
waypoints_rig = np.array([
	[35, 55, 5],
	[35, 55, z_dir],
	[39.5, 41.2, z_dir],
	[54, 41.2, z_dir],
	[42.3, 32.6, z_dir],
	[46.8, 18.8, z_dir],
	[35, 27.4, z_dir],
	[23.2, 18.8, z_dir],
	[27.7, 32.6, z_dir],
	[16.0, 41.2, z_dir],
	[30.5, 41.2, z_dir],
	[35, 55, z_dir],
	[35, 55, 5]
])


# Robot IP
abb_robot_ip = '192.168.60.101'
# Windows IP
# abb_robot_ip = '127.0.0.1:80'

dlam_des = 0.5 # mm between interpolated points

curve_p_rig = []

for i in range(len(waypoints_rig)-1):
	p_start = waypoints_rig[i]
	p_end = waypoints_rig[i+1]

	seg_len = np.linalg.norm(p_end - p_start)
	n_pts = max(int(seg_len / dlam_des) +1, 2)

	seg = np.linspace(p_start, p_end, n_pts)

	if i < len(waypoints_rig) - 2:
		curve_p_rig.extend(seg[:-1])
	else:
		curve_p_rig.extend(seg)

curve_p_rig = np.array(curve_p_rig)

# The pen orientation stays contant
tool_R_rig = np.array([
	[-1, 0, 0],
	[ 0, 1, 0],
	[ 0, 0, -1]
	]).T

curve_R_rig = np.tile(tool_R_rig, (len(curve_p_rig), 1, 1))

### Transform from rig frame to robot base frame
rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
Pbr = rig_pose[:3, -1]
Rbr = rig_pose[:3, :3]

curve_p_base = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R_base = np.array([Rbr @ R for R in curve_R_rig])
### End of Tranform, the path is in the robot base form now

#///////////////////////////////////#

###Convert Cartesian points to Joint angles
q_seed = np.zeros(6)
curve_js = []

for i in range(len(curve_p_base)):
	q_sol = robot.inv(curve_p_base[i], curve_R_base[i], q_seed)

	if q_sol is None or len(q_sol) == 0:
		raise Exception(f"IK Failed at point {i}")

	q = q_sol[0]
	curve_js.append(q)
	q_seed = q

curve_js = np. array(curve_js)
### Now curve_js is in joint space path directly from known coords

### Print Trajectory info for debugging purposes
#print("Trajectory generated successfully.")
#print("Number of Cartesian points:", len(curve_p_rig))
#print("Number of joint points:", len(curve_js))
#print("First rig-frame point:", curve_p_rig[0])
#print("Last rig-frame point:", curve_p_rig[-1])
#print("First base-frame point:", curve_p_base[0])
#print("Last base-frame point:", curve_p_base[-1])
#print("First joint target:", curve_js[0])
#print("Last joint target:", curve_js[-1])

### Plot the trajectory
plt.figure()
plt.plot(curve_p_rig[:, 0], curve_p_rig[:, 1], label="Planned Path")
plt.scatter(waypoints_rig[:, 0], waypoints_rig[:, 1], label="Waypoints")
plt.xlabel("Rig X (mm)")
plt.ylabel("Rig Y (mm)")
plt.title("Generated Star Trajectory in Rig Frame")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

### Generate smooth streamed joint trajectory
tool_vel = 2 #mm / s
tool_acc = 2 # mm/ s^2
TIMESTEP = 0.004

mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=9,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

fullruntraj_q, fullruntime_bp, pos_rig_list = trajectory_generate(
    curve_js, mctrl, lin_vel=tool_vel, lin_acc=tool_acc
)

print()
print(pos_rig_list)
input("Press Enter")
print("Smoothed trajectory generated.")
print("Number of streamed joint points:", len(fullruntraj_q))

f_d = -1

pos_base = mctrl.robot.fwd(fullruntraj_q[0]).p

pos_rig =  (Rbr.T)@(pos_base-Pbr)
print("")
print(pos_rig)

### Execute with mctrl + EGM
curve_js_exe = []
EE_pos_rig = []

time.sleep(0.1)
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
time.sleep(0.1)

print("Starting EGM ... ")
mctrl.start_egm()

force_log = []
time_log = []
t0 = time.time()

try:
	print("Reading current robot position...")
	q_start = read_position(mctrl)

	print("Jogging to first point...")
	q_all = np.linspace(q_start, curve_js[0], num = 100)
	traj_to_start, _, _ = trajectory_generate(
		q_all, mctrl, lin_vel=tool_vel, lin_acc=tool_acc
	)
	
	for i in range(len(traj_to_start)):
		read_q = read_position(mctrl)
		mctrl.position_cmd(traj_to_start[i])

	print("Robot in start position.")
	print("Executing main trajectory...")

	for i in range(len(fullruntraj_q)):
		read_q = read_position(mctrl)
		curve_js_exe.append(read_q)

		### Read force data
		ft = mctrl.ft_reading
		force_log.append(ft)
		time_log.append(time.time() - t0)
		print("Current force reading: ", ft)

        # Append the desired position to the trajectory
		pos_base = mctrl.robot.fwd(fullruntraj_q[i]).p
		pos_rig =  (Rbr.T)@(pos_base-Pbr)
		print(pos_rig)

		#Current EE position in base frame
		current_EEpos = mctrl.robot.fwd(read_q).p

		#Convert to rig frame for plotting like before
		coordinates_in_rig_frame = Rbr.T @ (current_EEpos - Pbr)
		EE_pos_rig.append([coordinates_in_rig_frame[0], coordinates_in_rig_frame[1]])

		#Stream next point
		mctrl.position_cmd(fullruntraj_q[i])

	print("Trajectory execution complete.")
	
finally:
	print("Stopping EGM...")
	mctrl.stop_egm()

force_log = np.array(force_log)
time_log = np.array(time_log)

curve_js_exe = np.array(curve_js_exe)
EE_pos_rig = np.array(EE_pos_rig)

np.savetxt("curve_js_exe.csv", curve_js_exe, delimiter=',')
np.savetxt("EE_pos.csv", EE_pos_rig, delimiter=',')

### Plot Executed Path like Before 
plt.figure()
plt.plot(EE_pos_rig[:, 0], EE_pos_rig[:, 1], label="Executed Path")
plt.plot(curve_p_rig[:, 0], curve_p_rig[:, 1], '--', label="Planned Path")
plt.xlabel("Rig X (mm)")
plt.ylabel("Rig Y (mm)")
plt.title("Executed Robot X-Y Path")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

### Plot Force Vs Time Plot
plt.figure()
plt.plot(time_log, force_log[:,2], label="Fz")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Force Vs Time")
plt.grid(True)
plt.legend()
plt.show()

# Print Statement for Debugging Purposes
print("End of Program")
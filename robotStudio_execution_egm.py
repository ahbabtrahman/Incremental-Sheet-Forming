import numpy as np
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
from matplotlib import pyplot as plt

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

print(fwdkin(robot, np.radians([0,30,30,0,20,0])))

# Define a curve in 3D space with postion and orientation in the rbots base frame
# sample a parabolic curve in 3D space
final_l = 300 # 300 mm curve
x_sample_range = np.linspace(0, final_l, int(final_l/0.1)+1) # sample every 0.1 mm
curve_p, curve_n = get_trajectory(x_sample_range)

# fix orientation R
curve_R = []
for i in range(len(curve_p)):
    if len(curve_R)==0:
        curve_R.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T)
    else:
        y_axis = curve_p[i]-curve_p[i-1]
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = curve_n[i]
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        curve_R.append(np.vstack((x_axis, y_axis, z_axis)).T)

# curve position translation
curve_p = curve_p + np.array([400, 0, 325])

## Planning a curve js using 5 dof constraints (position + 2 dof normal vector)
# get the initial joint angles
q_init = robot6_sphericalwrist_invkin(robot,Transform(curve_R[0], curve_p[0]),np.zeros(6))[0]

terminate_threshold = 0.001
angle_weight=1
alpha=1 # we can use a line search to find the best alpha. Here we use a fixed alpha.

# get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
curve_js = [q_init]
for i in range(1,len(curve_p)):

    q_all = [curve_js[-1]]
    while True:
        flange_T = fwdkin(robot, q_all[-1])
        vd = curve_p[i]-flange_T.p
        ezd = (curve_n[i]-flange_T.R[:,-1])
        
        nu_d = np.concatenate((ezd*angle_weight,vd))

        if np.linalg.norm(nu_d) < terminate_threshold:
            break

        J = robotjacobian(robot, q_all[-1])
        hatR = -np.eye(6)
        hatR[:3,:3] = hat(flange_T.R[:,-1])
        hatR[3:,:3] = hat(flange_T.R[:,-1])
        J_mod = -hatR@J

        H = J_mod.T @ J_mod + np.ones(6)*0.0001 # add a small value to make the matrix positive definite
        H = (H + H.T) / 2
        f = -J_mod.T @ nu_d
        qdot = solve_qp(H, f, lb=robot.joint_lower_limit+np.ones(6)*0.0001-q_all[-1], ub=robot.joint_upper_limit-np.ones(6)*0.0001+q_all[-1], solver="quadprog")

        q_all.append(q_all[-1] + alpha*qdot)
    curve_js.append(q_all[-1])
lam_planned = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
lam_planned = np.insert(lam_planned, 0, 0)

############# Execute the trajectory on the robot #############
TIMESTEP = 0.004 # 4 ms for egm control

# set up egm config
mm = abb.egm_minmax(-1e-3,1e-3)
egm_config = abb.EGMJointTargetConfig(
    mm, mm, mm, mm, mm ,mm, 1000, 1000
)
mp = abb.MotionProgram(egm_config = egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)
client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80")
lognum = client.execute_motion_program(mp, wait=False)
egm = EGM()

# read current joint position using egm
def read_position(egm):
    res, state = egm.receive_from_robot(timeout=0.1)
    if not res:
        raise Exception("Robot communication lost")
    return np.radians(state.joint_angles)

# send the next joint position to the robot using egm
def position_cmd(q,egm):
    egm.send_to_robot(np.degrees(q))

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
        ###
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

# Run it on the robot using EGM

# robot tool velocity and acceleration 
tool_vel = 300 # mm/s
tool_acc = 50 # mm/s^2

# first jog the robot to the initial position
q_start=read_position(egm)
q_all = np.linspace(q_start,curve_js[0],num=100)
traj_q, time_bp=trajectory_generate(q_all,robot,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
for i in range(len(traj_q)):
    read_q = read_position(egm) # reading joint position take approximately 4 ms
    position_cmd(traj_q[i],egm)

# execute the trajectory
traj_q, time_bp=trajectory_generate(curve_js,robot,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
curve_js_exe = []
for i in range(len(traj_q)):
    read_q = read_position(egm) # reading joint position take approximately 4 ms
    curve_js_exe.append(read_q)
    ####
    # In practice you can do control here
    # Eg. force control, joint position control, etc. 
    ####
    position_cmd(traj_q[i],egm)

client.stop_egm() # stop egm

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

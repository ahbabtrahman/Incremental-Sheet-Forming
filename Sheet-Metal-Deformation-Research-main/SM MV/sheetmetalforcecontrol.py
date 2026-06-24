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
from general_robotics_toolbox import *
from calibration import *

sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
sys.path.append('robot_motion')
from RobotMotionController import *
from robot_def import *

from traj_gen import get_trajectory
#This file incorporated force control with published joint positions
TIMESTEP = .004

# helper functions 
def calc_lam_js(curve_js,robot):
    #curve_js is the list of joint angles, Nx1
    curve_p = [] # flange position
    for i in range(len(curve_js)):
        curve_p.append(robot.fwd(curve_js[i]).p) #Calculate pose for each set of joint angles
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
def read_position():
    for i in range(0,20):
        res, state = mctrl.egm.receive_from_robot(timeout=0.1)
        if not res:
            #raise Exception("Robot communication lost")
            mctrl.stop_egm() # stop egm
            mctrl.start_egm()
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


####################################################FT Connection####################################################
# ati_tf=NET_FT('192.168.60.100')
# ati_tf.start_streaming()
H_pentip2ati=np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip=np.linalg.inv(H_pentip2ati)
ad_ati2pentip=adjoint_map(Transform(H_ati2pentip[:3,:3],H_ati2pentip[:3,-1]))
ad_ati2pentip_T=ad_ati2pentip.T
#################### FT Connection ####################
RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')

#########################################################Robot config parameters#########################################################
#MAKE SURE THIS IS RIGHT
#Measure the z and y displacement from the flange to the tool tip
Pft = np.array([-55.45, 0, 131.2])
#x to inside of pen holder: -49.55
#z to end of pen holder 92.4

tool_T = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T, delimiter = ',')

robot=robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",tool_file_path="rig_pen.csv")
q_seed=np.zeros(6)


#Define Tool Offset
Pft = np.array([-55.755, 0, 130.05])
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
mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

fullruntraj_q = np.loadtxt("fullruntraj_q.csv", delimiter = ',')
fullruntime_bp = np.loadtxt("fullruntime_bp.csv", delimiter = ',')
curve_js_exe = np.loadtxt("curve_js_exe.csv", delimiter = ',').tolist()
curve_js = np.loadtxt("curve_js.csv", delimiter = ',').tolist()
lam_planned = np.loadtxt("lam_planned.csv", delimiter = ',')
lam_planned = lam_planned[0]

# test = np.zeros((fullruntraj_q.shape[0], 3))
# for i in range(0,fullruntraj_q.shape[0]):
#     test[i] = fwdkin(robot,fullruntraj_q[i]).p
# np.savetxt("testfile.csv",test,delimiter=',')

# set up egm config
mctrl.start_egm()

# Run it on the robot using EGM
print("Robot start moving. EGM is running")

tool_vel = 5
tool_acc = 5
#first jog the robot to the initial position
q_start=read_position()
q_all = np.linspace(q_start,curve_js[0],num=100)
traj_q, time_bp=trajectory_generate(q_all,robot,lin_vel=tool_vel,lin_acc=tool_acc) # generate a smooth trajectory
for i in range(len(traj_q)):
    read_q = read_position() # reading joint position take approximately 4 ms
    mctrl.position_cmd(traj_q[i])
print("Robot in Start Position")


#iterations = 20
#for i in range(0,iterations):
time.sleep(0.1)
# ati_tf.set_tare_from_ft()	#clear bias
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
# res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
time.sleep(0.1)
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias

total_depth = -.32
inc = .04
steps = -int(total_depth // inc) #make sure is positive and integer
depth = 0 #Current depth
print_count = 0
offset = 1.5 #Find manually through checking 0 offset
startheight = 10 #Defined in waypoints sheet
forces = []

st = time.perf_counter()
ft_record = []
runtime = []
qdata = []
rig_data = []
for j in range(0,len(forces)):
    # depth = -j*inc
    # print(f"Current depth is {depth}")
    # print(f"Currently on pass {j}")
    fz_des = forces[i]
    for i in range(len(fullruntraj_q)):
        load_speed = mctrl.params['load_speed']
        #Record Last Force
        ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading
        fz_now = float(ft_tip[-1]) #Grab latest force
        read_q = read_position() # reading joint position take approximately 4 ms
        #Record Data
        ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(read_q))) #time, tip force, joints
        #Make sure not crashing
        # force protection
        if np.linalg.norm(ft_tip[3:])>mctrl.FORCE_PROTECTION:
            print("force: ",ft_tip[3:])
            print("force too large")
            break
        if time.time()-mctrl.last_ft_time>0.1:
            print("force reading lost")
            break
        #Position Adjustment
        #Grab pos, rot from joints to be sent to robot
        t_base = robot.fwd(fullruntraj_q[i])
        pos_base = t_base.p
        rot_base = t_base.R
        pos_rig =  (Rbr.T)@(pos_base-Pbr) #Rig Frame

        #Force Control for Depth
        tip_now = mctrl.robot.fwd(read_q) #Get current tip EE pos
        tip_now_rig = mctrl.ipad_pose_T.inv()*tip_now #Transform into ipad/rig frame?
        tip_now_rig = deepcopy(tip_now_rig)
        if np.abs(fz_now) < mctrl.params['force_epsilon'] and touch_t is None: # if not touch ipad
            tip_now_rig.p[2] = tip_now_rig.p[2] + -1*load_speed * mctrl.TIMESTEP # move in -z direction , loadspeed*time = distance down
        else: # when touch ipad
            if touch_t is None:
                print("contact made")
                touch_t=time.time()
                if np.abs(fz_now - fz_des) < .05 and desired_force == False:
                    desired_force = True
			# track a trapziodal force profile
            this_fz_des = min(fz_des,(time.time()-touch_t)*mctrl.params["trapzoid_slope"])
			#print(f"Printing fz_des: {this_fz_des}")
			#print(f"Printing from FT sensor: {fz_now}")
            f_err = this_fz_des-fz_now# feedback error
            Kf = .1
            v_des = mctrl.force_impedence_ctrl(f_err) # force control, returns "force_ctrl_damping"*f_err
            tip_now_rig.p[2] = tip_now_rig.p[2] + v_des * mctrl.TIMESTEP*Kf # force impedence control
        
        # get joint angles using ik
        tip_next = mctrl.ipad_pose_T*tip_now_rig #???
		# print(tip_next.p)
        q_des = mctrl.robot.inv(tip_next.p,tip_next.R,q_now)[0] #Find desired joint angles
		#Send joint angles to robot
        position_cmd(q_des)
            
        new_pos_base= Rbr@pos_rig + Pbr #Base Frame
        #curve_pbr = Rbr@curve_p[i,:] + Pbr --> Original transformation to base frame
        rig_data.append(deepcopy(pos_rig))
        #Find joints to send to robot
        run_q = robot.inv(new_pos_base,rot_base,read_q)[0]
        curve_js_exe.append(read_q)
        #Move robot
        mctrl.position_cmd(run_q)


mctrl.stop_egm() # stop egm
 
print("All done! Saving data...")

curve_js_exe = np.array(curve_js_exe) # executed joint angles
lam_exe = calc_lam_js(curve_js_exe, robot) # executed path length

#save force data
forcedata = np.column_stack((runtime, ft_record))
np.savetxt("FTData.csv", forcedata, delimiter=',')

#save actual joint data
jointdata = np.column_stack((runtime, qdata))
np.savetxt("JointData.csv", jointdata, delimiter=',')

#save curve_js_exe
np.savetxt("curvejsexedata.csv", curve_js_exe, delimiter=',')

print("Data saved!")
    
   
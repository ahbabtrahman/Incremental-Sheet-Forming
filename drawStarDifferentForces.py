#///////////////////////////////////////////
# Editor(s): Ahbab R.
# Last Edited: 04/13/2026
# Notes:
# This script takes inspiration from linetestegmFT.py
# It draws a star at a fixed negative Z force.
#
# Change IP at Line 238
#
# Make sure to physically update the zcalib file on the computer 
#
# Assumptions:
# Z-Calibration has already been run
#//////////////////////////////////////////

print("Start of Program")

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

#////////////////////////////

# --- Parameters & Waypoints ---
vd = 2                 # Desired dragging velocity (mm/s)
forces = [-1.0, -1.2, -1.5, -1.8, -2.0, -2.2, -2.5, -2.8, -3.0, -3.2]
z_dir = -1.2           # Contact depth in rig frame
TIMESTEP = 0.004       # 4ms EGM interval

# Your specific star/polygon coordinates
waypoints_rig = np.array([
    [35, 55, 5],       # 0: Safe Approach
    [35, 55, z_dir],    # 1: Start Contact
    [39.5, 41.2, z_dir],
    [54, 41.2, z_dir],
    [42.3, 32.6, z_dir],
    [46.8, 18.8, z_dir],
    [35, 27.4, z_dir],
    [23.2, 18.8, z_dir],
    [27.7, 32.6, z_dir],
    [16.0, 41.2, z_dir],
    [30.5, 41.2, z_dir],
    [35, 55, z_dir],    # 11: End Contact
    [35, 55, 5]        # 12: Retract
])

# Helper Functions 

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
def position_cmd(q):
    mctrl.egm.send_to_robot(np.degrees(q))
def maintain_z_force(mctrl,fz_des, pos_des, start_x, vd, load_speed=None):
	#Function that draws a straight line to desired position at a constant force, records force data and joints
	#pos_des - Desired Point
	#fz_des - Desired Constant Force
	#Returns recorded force and joint data

	if load_speed is None:
		load_speed = mctrl.params['load_speed']

	touch_t = None
	this_st=None
	set_time=None
	desired_force = False
	ft_record=[]

	counter = 0

	while True:
		# tool pose reading
		counter += 1
		q_now = read_position() #Get current joints
		tip_now = mctrl.robot.fwd(q_now) #Get current tip EE pos
		tip_now_rig = mctrl.ipad_pose_T.inv()*tip_now #Transform into ipad/rig frame?
		#Break if at end of line
		if np.linalg.norm(tip_now_rig.p[:2] - pos_des[:2]) < 1:
			print("End of line reached")
			break
		# force reading
		ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading #Array of force at pen tip
		fz_now = float(ft_tip[-1]) #Grab latest force
		if counter % 200 == 0:
			print(fz_now)
		# force protection
		if np.linalg.norm(ft_tip[3:])>(mctrl.FORCE_PROTECTION):
			print("force: ",ft_tip[3:])
			print("force too large")
			break
		if time.time()-mctrl.last_ft_time>0.1:
			print("force reading lost")
			break
		# time force joint angle record
		#ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now))) #time, tip force, joints
		ft_record.append(np.hstack((
    		[time.time(), fz_now],
    		tip_now_rig.p,              # x, y, z
    		np.degrees(q_now)
		)))



		# force control
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
			this_fz_des = max(fz_des, -(time.time()-touch_t)*mctrl.params["trapzoid_slope"])
			#print(f"Printing fz_des: {this_fz_des}")
			#print(f"Printing from FT sensor: {fz_now}")
			f_err = this_fz_des-fz_now# feedback error
			Kf = .05
			v_des = mctrl.force_impedence_ctrl(f_err) # force control, returns "force_ctrl_damping"*f_err
			tip_now_rig.p[2] = tip_now_rig.p[2] + v_des * mctrl.TIMESTEP*Kf # force impedence control
			if desired_force == True:
				#ALSO MOVE IN X DIR
				pos_delta_x = vd*mctrl.TIMESTEP
				pos_delta_x = np.clip(pos_delta_x, -3, 3) #Safety
				#Add change in x dir. to current x coordinate
				start_x += pos_delta_x
		# get joint angles using ik


		#tip_now_rig.p[0] = start_x
		direction = pos_des - tip_now_rig.p
		direction[2] = 0
		norm = np.linalg.norm(direction)

		if norm < 1e-6:
			print("Reached Waypoint")
    		break

		direction = direction / norm

		tip_now_rig.p[:2] += direction[:2] * vd * mctrl.TIMESTEP


		tip_next = mctrl.ipad_pose_T*tip_now_rig #???
		q_des = mctrl.robot.inv(tip_next.p,tip_next.R,q_now)[0] #Find desired joint angles
		#Send joint angles to robot
		position_cmd(q_des)
	
	return ft_record

# End of Helper Functions

#//////////////////////////

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

# Robot IP
#abb_robot_ip = '192.168.60.101'
# Windows IP
abb_robot_ip = '127.0.0.1:80'

rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')
mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=10,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

print("Robot obj created")
#Rig Kinematics
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radiansforce pro
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

# Run it on the robot
qbr = R2q(Rbr)
#////////////////////////////////////

corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
corner_R_base = Rbr @ corner_R

# Main loop
print("Starting EGM...")
mctrl.start_egm()

try:
    # --- 1. Move to safe approach point ---
    q_read = read_position()
    approach_base = Rbr @ waypoints_rig[0] + Pbr
    q_approach = robot.inv(approach_base, corner_R_base, q_read)[0]
    mctrl.jog_joint_position_cmd(q_approach, v=controller_params["jogging_speed"])
    input("At safe approach. Press Enter...")

    # --- 2. Move to contact start (no force yet) ---
    q_read = read_position()
    contact_base = Rbr @ waypoints_rig[1] + Pbr
    q_contact = robot.inv(contact_base, corner_R_base, q_read)[0]
    mctrl.jog_joint_position_cmd(q_contact, v=controller_params["jogging_speed"])
    input("At contact point. Press Enter...")

    # --- 3. Zero FT sensor ---
    time.sleep(0.1)
    mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
    time.sleep(0.1)
    mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))

    print(f"Drawing star with varying forces")

    ft_all = []

    # --- 4. Follow star path with force control ---
    for i, force_des in zip(range(2, len(waypoints_rig) - 1), forces):
        print(f"Segment {i-1} -> {i}")

        # Current segment start x
        start_x = waypoints_rig[i-1][0]

        # Target waypoint
        pos_des = waypoints_rig[i]

        # Run force-controlled motion
        ft_record = maintain_z_force(
            mctrl,
            force_des,
            pos_des,
            start_x,
            vd
        )

        ft_all.extend(ft_record)

    # --- 5. Retract safely ---
    q_read = read_position()
    retract_base = Rbr @ waypoints_rig[-1] + Pbr
    q_retract = robot.inv(retract_base, corner_R_base, q_read)[0]
    mctrl.jog_joint_position_cmd(q_retract, v=controller_params["jogging_speed"])

    # --- 6. Save data ---
    ft_all = np.array(ft_all)
    np.savetxt("star_ft_record.csv", ft_all, delimiter=',')

    print("Star drawing complete.")

except (Exception, KeyboardInterrupt) as e:
    print("Error:", e)
    mctrl.stop_egm()
    import traceback
    traceback.print_exc()

mctrl.stop_egm()

if len(ft_all) == 0:
	print("No data Recorded")
else:
	
	# Convert to numpy
	ft_all = np.array(ft_all)

	# Extract data
	t = ft_all[:, 0]
	fz = ft_all[:, 1]
	x = ft_all[:, 2]
	y = ft_all[:, 3]
	z = ft_all[:, 4]

	# --- Plot 1: XY path colored by Z ---
	plt.figure()
	sc = plt.scatter(x, y, c=z, cmap='viridis', s=5)
	plt.colorbar(sc, label='Z (mm)')
	plt.xlabel('X (mm)')
	plt.ylabel('Y (mm)')
	plt.title('Robot Path (colored by Z)')
	plt.axis('equal')
	plt.grid()
	
	# --- Plot 2: Force vs Time ---
	plt.figure()
	plt.plot(t - t[0], fz)
	plt.xlabel('Time (s)')
	plt.ylabel('Force Z (N)')
	plt.title('Force vs Time')
	plt.grid()

	plt.show()





print("End of Program")
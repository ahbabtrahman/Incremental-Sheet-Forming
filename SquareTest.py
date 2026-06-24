#///////////////////////////////////////////////////////////
# Editor(s): Ahbab R. 
# Last Edited: 02/09/2026
# This script is based on the pointTestEGMFT.py program and 
# behaves very similarly to it. The program draws a square
# at a fixed force. It also reads and outputs the force 
# values and joint positions.
# Change Potential Gain, Kf @ line 112
# Change Coordinates for square @ line 212
# Change Force @ line 230
#
# Assumptions: 
# Z-Calibration has already been run
#///////////////////////////////////////////////////////////

#Import Statements
from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from general_robotics_toolbox import *
from calibration import *
sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
sys.path.append('robot_motion')
from RobotMotionController import *
from copy import deepcopy
import numpy as np

# Function: Read Position. This function reads the postion of the robot joints.
# Input: None
# Output: Joint angles in radians
def read_position():
    for i in range(0,20):
        res, state = mctrl.egm.receive_from_robot(timeout=0.1)
        if not res:
            # stop egm
            mctrl.stop_egm()
            mctrl.start_egm()
            print("Communication Lost, Reconnecting")
        else:
            break
    return np.radians(state.joint_angles)

# Function: Position cmd
# Input: joint angle q
# Output: None
def position_cmd(q):
  mctrl.egm.send_to_robot(np.degrees(q))

# Function: Maintain Z-Force. This function draws a straight line to desired position at
#           at a constant force, records force data and joints.
#           This function also has a potential gain,Kf, for PID control. 
# Input: mctrl, fz_des(desired constant force), pos_des(desired position), load_speed(Set to none)
# Output: Recorded Force and Joint data
def maintain_z_force(mctrl,fz_des, pos_des, load_speed=None):

  #Set Initial Parameters
  if load_speed is None:
	load_speed = mctrl.params['load_speed']
	touch_t = None
	this_st = None
	set_time = None
	desired_force = False
	ft_record = []

  while True:
    
		# Tool pose reading
		q_now = read_position()
		tip_now = mctrl.robot.fwd(q_now)
		tip_now_rig = mctrl.ipad_pose_T.inv()*tip_now
    
		# Force reading
		ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading
		fz_now = float(ft_tip[-1])
		print(fz_now)
    
		# Force protection
		if np.linalg.norm(ft_tip[3:])>(mctrl.FORCE_PROTECTION):
			print("force: ",ft_tip[3:])
			print("force too large")
			break
		if time.time()-mctrl.last_ft_time>0.1:
			print("force reading lost")
			break
      
		# Time force joint angle record
    # Time, tip force, joints
		ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now))) 

		# Force control
		tip_now_rig = deepcopy(tip_now_rig)

    # If not touch ipad, move in -z direction , loadspeed*time = distance down
		if np.abs(fz_now) < mctrl.params['force_epsilon'] and touch_t is None: 
			tip_now_rig.p[2] = tip_now_rig.p[2] + -1*load_speed * mctrl.TIMESTEP 
		else:
			if touch_t is None:
				print("contact made")
				touch_t=time.time()
			if np.abs(fz_now - fz_des) < .05 and desired_force == False:
				desired_force = True
        
			# Track a trapziodal force profile
			this_fz_des = min(fz_des,(time.time()-touch_t)*mctrl.params["trapzoid_slope"])
      # Feedback Error
			f_err = this_fz_des-fz_now
      # Potantial Gain = 0.05
			Kf = .05
			v_des = mctrl.force_impedence_ctrl(f_err) 
			tip_now_rig.p[2] = tip_now_rig.p[2] + v_des * mctrl.TIMESTEP*Kf 

		# check if force achieved
		if np.fabs(fz_des-fz_now)<mctrl.params['force_epsilon']:
			if set_time is None:
				set_time = time.time()
			if (time.time()-set_time)>mctrl.params['settling_time']:
				break
		else:
			set_time = time.time()

		# get joint angles using ik
		tip_next = mctrl.ipad_pose_T*tip_now_rig
		q_des = mctrl.robot.inv(tip_next.p,tip_next.R,q_now)[0]
    
		#Send joint angles to robot
		position_cmd(q_des)
	
	return ft_record

# End of Functions
#///////////////////////////////////////////////////////////////////////////

# FT Connection
H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_ati2pentip = adjoint_map(Transform(H_ati2pentip[:3,:3],H_ati2pentip[:3,-1]))
ad_ati2pentip_T = ad_ati2pentip.T
RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')
# End of FT Connection


# Robot Config Parameters

# MAKE SURE THIS IS RIGHT
# Measure the z and y displacement from the flange to the tool tip
Pft = np.array([-55.45, 0, 131.2])
# x to inside of pen holder: -49.55
# z to end of pen holder: 92.4

tool_T = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T, delimiter = ',')
robot = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",tool_file_path="rig_pen.csv")
q_seed = np.zeros(6)
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# Robot IP
#abb_robot_ip = '192.168.60.101'

# Windows IP
abb_robot_ip = '127.0.0.1:80'

# Move Speed of 0.004s or 4ms
TIMESTEP = 0.004
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

rig_pose = np.loadtxt("rig_pose.csv",delimiter=',')
mctrl = MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=15,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

print("Robot obj created")
# End of Robot Config Parmeters

# //////////////////////////////////////////////////////////////////
# Rig Kinematics
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

# Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radiansforce pro
# Current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

# Run it on the robot
qbr = R2q(Rbr)
# End of Rig Kinematics


# Points array
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
corner_R_base = Rbr@corner_R

#///////////////////////////////////////
# Control Points Here:
# Replace a, b, c and d to make a square
x1, x2, y1, y2 = a, b, c, d
points = []
points.append([[x1, y1, 10], [x1, y1, 3], [x2, y1, 10]])
points.append([[x2, y1, 10], [x2, y1, 3], [x2, y2, 10]])
points.append([[x2, y2, 10], [x2, y2, 3], [x1, y2, 10]])
points.append([[x1, y2, 10], [x1, y2, 3], [x1, y1, 10]])
point_array = np.array(points)
#//////////////////////////////////////

# End of Points Array

# Draw Sqare
st = time.perf_counter()
mctrl.start_egm()

for point in point_array:
  #//////////////////////////////////
  # Control force here
  force_des = -4.0
  #/////////////////////////////////
  try:
    print(f"Crrent Force: {force_des} Current Point: {point[0]}")
    #Jog to position above start point of line, 10mm above 
		q_read = read_position()
		line_start_base = Rbr@point[0] + Pbr
		start_point = robot.inv(line_start_base, corner_R_base, q_read)[0]	
		mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])
		input("Above Start Position...")

		#Jog to point at 3mm above
		q_read = read_position()
		line_contact_base = Rbr@point[1] + Pbr
		start_point=robot.inv(line_contact_base, corner_R_base, q_read)[0]	
		mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])
		input("Contacting Sheet Metal...")

		time.sleep(0.1)
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
		time.sleep(0.1)
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))

		##DRAW LINE###
		ft_record = maintain_z_force(mctrl, force_des, point[2])
		##############

		#Jog to 10mm above
		q_read = read_position()
		line_end_base = Rbr@point[2] + Pbr
		start_point=robot.inv(line_end_base, corner_R_base, q_read)[0]
		mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])

		ft_record = np.array(ft_record)
		np.savetxt(f"pointftrecord{force_des}.csv", ft_record, delimiter = ',')
	
	except (Exception,KeyboardInterrupt) as e:
		print("Error:", e)
		mctrl.stop_egm()
		traceback().print_exc()
		break
# End of Draw Array

mctrl.stop_egm()
# End of Program





















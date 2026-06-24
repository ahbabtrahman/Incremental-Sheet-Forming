from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
#import glob, cv2, sys, time
import sys
#from sklearn.decomposition import PCA
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

#This file draws 7 lines in the sheet metal at different forces. Records force values and joint positions.

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
def maintain_z_force(mctrl,fz_des, pos_des, load_speed=None):
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

	while True:
		# tool pose reading
		q_now = read_position() #Get current joints
		tip_now = mctrl.robot.fwd(q_now) #Get current tip EE pos
		tip_now_rig = mctrl.ipad_pose_T.inv()*tip_now #Transform into ipad/rig frame?
		#Break if at end of line
		if np.abs(tip_now_rig.p[1] -  pos_des[1]) < 1:
			break
		# force reading
		ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading #Array of force at pen tip
		fz_now = float(ft_tip[-1]) #Grab latest force
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
		ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now))) #time, tip force, joints

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
			this_fz_des = min(fz_des,(time.time()-touch_t)*mctrl.params["trapzoid_slope"])
			#print(f"Printing fz_des: {this_fz_des}")
			#print(f"Printing from FT sensor: {fz_now}")
			f_err = this_fz_des-fz_now# feedback error
			Kf = .05
			v_des = mctrl.force_impedence_ctrl(f_err) # force control, returns "force_ctrl_damping"*f_err
			tip_now_rig.p[2] = tip_now_rig.p[2] + v_des * mctrl.TIMESTEP*Kf # force impedence control
			if desired_force == True:
				#ALSO MOVE IN Y DIR
				Kp = 1.5
				v_des_y = (pos_des[1] - tip_now_rig.p[1])*Kp
				v_des_y = 2*np.clip(v_des_y, -10, 10)
				tip_now_rig.p[1] =  tip_now_rig.p[1] + v_des_y*mctrl.TIMESTEP
		
		# check if force achieved
		# if np.fabs(fz_des-fz_now)<mctrl.params['force_epsilon']:
		# 	if set_time is None:
		# 		set_time = time.time()
		# 	if (time.time()-set_time)>mctrl.params['settling_time']:
		# 		break
		# else:
		# 	set_time = time.time()

		# get joint angles using ik
		tip_next = mctrl.ipad_pose_T*tip_now_rig #???
		# print(tip_next.p)
		q_des = mctrl.robot.inv(tip_next.p,tip_next.R,q_now)[0] #Find desired joint angles
		#Send joint angles to robot
		position_cmd(q_des)
	
	return ft_record
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

print(tool_T)

print(robot.R_tool,robot.p_tool)
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

print("Robot obj created")
#############################################Rig Kinematics###################################################
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
################################################################################################

corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
corner_R_base = Rbr@corner_R
#Create 7 lines
x_points = [10,20,30,40,50,60]

lines = []
for startpoints in x_points:
	segments = []
	buffer1 = np.array([startpoints, 10, 10])
	line_start = np.array([startpoints, 10, 3])
	line_end = np.array([startpoints, 70, 3])
	buffer2 = np.array([startpoints, 70, 10])
	segments.append(buffer1)
	segments.append(line_start)
	segments.append(line_end)
	segments.append(buffer2)
	seg_array = np.array(segments)
	lines.append(seg_array)
lines = np.array(lines)

# #Transformation to rig
# for i in range(0,lines.shape[0]):
# 	for j in range(0,4):
# 		lines[i][j] = Rbr@lines[i][j] + Pbr

increments = [4, 5, 6, 7, 8, 9, 10]
force = -.25 #N

st = time.perf_counter() #N
#force_des = -.25*i
st = time.perf_counter()
mctrl.start_egm()
for inc, line in zip(increments, lines):
    for i in range(1, inc+1):
        try:        
			#Jog to position above start point of line, 10mm above 
            q_read = read_position()
            print(line[0])
            print(Rbr)
            line_start_base = Rbr@line[0] + Pbr
            start_point=robot.inv(line_start_base, corner_R_base, q_read)[0]	###initial joint position
            mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])
            input("Above Start Position...")

            #Jog to start of line at zero depth
            q_read = read_position()
            line_contact_base = Rbr@line[1] + Pbr
            start_point=robot.inv(line_contact_base, corner_R_base, q_read)[0]	###initial joint position
            mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])
            input("Contacting Sheet Metal...")

            time.sleep(0.1)
            # ati_tf.set_tare_from_ft()	#clear bias
            mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
            # res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
            time.sleep(0.1)
            mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias

            ##DRAW LINE###
            force_des = -.25*i  
            ft_record = maintain_z_force(mctrl, force_des, line[2])
            ##############

            #Jog to start of end of line 10mm above
            q_read = read_position()
            line_end_base = Rbr@line[3] + Pbr
            start_point=robot.inv(line_end_base, corner_R_base, q_read)[0]	###initial joint position
            mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])

            ft_record = np.array(ft_record)
            np.savetxt(f"lineftrecord{force_des}line{line[0][0]}.csv", ft_record, delimiter = ',')

        except (Exception,KeyboardInterrupt) as e:
            print("Error:", e)
            mctrl.stop_egm()
            traceback().print_exc()
            break

mctrl.stop_egm()
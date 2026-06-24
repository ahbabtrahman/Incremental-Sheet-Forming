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


rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')


mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)


f_d=-1	#10N push down
mctrl.start_egm()
for corner in corners:
	try:
		corner_top=corner+20*rig_pose[:3,-2]
		corner_top_safe=corner+50*rig_pose[:3,-2]
		print(corner_top)
		print(corner_top_safe)
		input("Move to corner")
		q_corner_top=robot.inv(corner_top,R_pencil,q_seed)[0]	###initial joint position
		q_corner_top_safe=robot.inv(corner_top_safe,R_pencil,q_seed)[0]
		mctrl.jog_joint_position_cmd(q_corner_top_safe,v=controller_params["jogging_speed"])
		input("Push")
		mctrl.jog_joint_position_cmd(q_corner_top,v=controller_params["jogging_speed"])

		time.sleep(0.1)
		# ati_tf.set_tare_from_ft()	#clear bias
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
		# res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
		time.sleep(0.1)
		mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
		time.sleep(0.1)
		print("Current force reading:",mctrl.ft_reading)
		input("Start pushing")
                
		for i in range(100): # making sure to get the latest joint position
			q_cur = mctrl.read_position()

		ft_record = mctrl.force_load_z(f_d)
		
		for i in range(100): # making sure to get the latest joint position
			q_cur = mctrl.read_position()
		corners_adjusted.append(robot.fwd(q_cur).p)
		print("Adjusted corner:",corners_adjusted[-1])

		mctrl.jog_joint_position_cmd(q_corner_top_safe,v=controller_params["jogging_speed"])

		ft_record = np.array(ft_record)
		plt.plot(ft_record[:,0],ft_record[:,1],'-o')
		plt.xlabel('Time')
		plt.ylabel('Force')
		plt.show()
	except (Exception,KeyboardInterrupt) as e:
		print("Error:", e)
		mctrl.stop_egm()
		exit()

mctrl.stop_egm()


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
import time
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

rig_pose=np.loadtxt("rig_pose_raw.csv",delimiter=',')

print(rig_pose)

w = 85.2 # width of the rig (parallel to y-axis)
h = 100.07
l = 19.7
thickness = 2.84

rig_pose[:3,-1]=rig_pose[:3,-1]+rig_pose[:3,:3]@np.array([h/2,w/2,0])

R_pencil=rig_pose[:3,:3]@Ry(np.pi)

mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)
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
import time
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

rig_pose=np.loadtxt("rig_pose_raw.csv",delimiter=',')

print(rig_pose)

w = 85.2 # width of the rig (parallel to y-axis)
h = 100.07
l = 19.7
thickness = 2.84

rig_pose[:3,-1]=rig_pose[:3,-1]+rig_pose[:3,:3]@np.array([h/2,w/2,0])

R_pencil=rig_pose[:3,:3]@Ry(np.pi)

mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

filename = input("Enter the name of the file to save the data: ")

plt.xlabel('Time')
plt.ylabel('Force')


ft_record = []
runtime = []
#iterations = 20
#for i in range(0,iterations):
time.sleep(0.1)
# ati_tf.set_tare_from_ft()	#clear bias
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
# res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
time.sleep(0.1)
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
while True:
	try:
		t = time.time()
		time.sleep(0.1)
		print("Current force reading:",mctrl.ft_reading)
		print("Time:",t)
		#Record Last Force
		ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading
		print("tip force")
		print(ft_tip[-1][0])
		ft_record.append(ft_tip[-1][0])
		runtime.append(t)

	except (Exception,KeyboardInterrupt) as e:
		print("Error:", e)
		#mctrl.stop_egm()
		print(ft_record)
		plt.title("Force Torque Sensor Reading")
		plt.plot(runtime,ft_record,'-o')
		plt.show()
		data = np.column_stack((runtime, ft_record))
		np.savetxt(filename, data, delimiter=',')
		exit()


print(ft_record)
plt.title("Force Torque Sensor Reading")
plt.plot(runtime,ft_record,'-o')
plt.show()
np.savetxt(filename, ft_record, delimiter=',')

		

filename = input("Enter the name of the file to save the data: ")

plt.xlabel('Time')
plt.ylabel('Force')


ft_record = []
runtime = []
#iterations = 20
#for i in range(0,iterations):
time.sleep(0.1)
# ati_tf.set_tare_from_ft()	#clear bias
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
# res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
time.sleep(0.1)
mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool")) # clear bias
while True:
	try:
		t = time.time()
		time.sleep(0.1)
		print("Current force reading:",mctrl.ft_reading)
		print("Time:",t)
		#Record Last Force
		ft_tip = mctrl.ad_ati2pentip_T@mctrl.ft_reading
		print("tip force")
		print(ft_tip[-1][0])
		ft_record.append(ft_tip[-1][0])
		runtime.append(t)

	except (Exception,KeyboardInterrupt) as e:
		print("Error:", e)
		#mctrl.stop_egm()
		print(ft_record)
		plt.title("Force Torque Sensor Reading")
		plt.plot(runtime,ft_record,'-o')
		plt.show()
		data = np.column_stack((runtime, ft_record))
		np.savetxt(filename, data, delimiter=',')
		exit()


print(ft_record)
plt.title("Force Torque Sensor Reading")
plt.plot(runtime,ft_record,'-o')
plt.show()
np.savetxt(filename, ft_record, delimiter=',')

		

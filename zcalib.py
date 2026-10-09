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
Pft = np.array([-55.755, 0, 130.05])
#Pft = np.array([0, 0, 124.00])
#186
#Pft = np.array([0, 0, 148.64])
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
    "force_ctrl_damping": 40.0, # 200, 180, 90, 60 	
    "force_epsilon": 1.5, # Unit: N
    "moveL_speed_lin": 6.0, # 10 Unit: mm/sec
    "moveL_acc_lin": 7.2, # Unit: mm/sec^2 0.6, 1.2, 3.6
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 10.0, # Unit mm/sec
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 0.2, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec, 0.02
    "jogging_speed": 25, # Unit: mm/sec
    "jogging_acc": 10, # Unit: mm/sec^2
    'force_filter_alpha': 0.9 # force low pass filter alpha
    }
#Make sure work object is in global
#Smaller rig
#Top is for regular holder
#c1 = np.array([446.54, 42.66, 351.75])
#Bottom is for longer holder
#c1 = np.array([604.54, 67.66, 334.75])  
# # # # # # #bottom right --> Po, rig origin point
#c2 = np.array([513.05, -64.5, 351.29])
#c2 = np.array([606.05, -17.5, 334.29])  
# # # # # # #top left
#c3 = np.array([592.41, 43.35, 348.02])
#c3 = np.array([705.41, 68.35, 331.02])
# # # # # # #top right
#c4 = np.array([593.78, -41.53, 347.79])
#c4 = np.array([706.78, -16.53, 330.79])

#larger rig coord:
#Top is for base holder
c1 = np.array([246.55, 107.4, 331.16])t
#Bottom is for longer holder 
#c1 = np.array([292.55, 97.4, 366.16])
c2 = np.array([249.10, -42.84, 330.65])
#c2 = np.array([295.10, -52.84, 365.65]) 
c3 = np.array([397.67, 109.86, 330.52])
#c3 = np.array([353.67, 99.86, 365.52])
c4 = np.array([398.72, -39.88, 329.71])
#c4 = np.array([454.72, -59.88, 364.71])


#Quaternion from teachpendant, base to flange
RoF = q2R(np.array([.01916, .00040, -.99982, -.00112]))

c1 = c1 + RoF@Pft
c2 = c2 + RoF@Pft
c3 = c3 + RoF@Pft
c4 = c4 + RoF@Pft
print("c1")
print(c1)
print("c2")
print(c2)
print("c3")
print(c3)
print("c4")
print(c4)
#saves raw Pbr, Rbr to .csv file, ready for second calibration for z axis
calibrate(c1,c2,c3,c4)

rig_pose=np.loadtxt("rig_pose_raw.csv",delimiter=',')

print(rig_pose)

#Smaller rig
#w = 85.2 # width of the rig (parallel to y-axis)
#h = 100.07
#l = 19.7


#Larger rig
#Make sure h>w for PCA
w = 150
h = 151
l = 25

thickness = 3.175

rig_pose[:3,-1]=rig_pose[:3,-1]+rig_pose[:3,:3]@np.array([h/2,w/2,0])

R_pencil=rig_pose[:3,:3]@Ry(np.pi)

mctrl=MotionController(robot,rig_pose,H_pentip2ati,controller_params,TIMESTEP,FORCE_PROTECTION=6,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

corners_offset=np.array([[h/2+l/2,0,0],[0,w/2+l/2,0],[-h/2-l/2,0,0],[0,-w/2-l/2,0]])

corners=np.dot(rig_pose[:3,:3],corners_offset.T).T+np.tile(rig_pose[:3,-1],(4,1))

###loop four corners to get precise position base on force feedback
corners_adjusted=[]
f_d=-1	#10N push down
mctrl.start_egm()
for corner in corners:
	try:
		#This is for the larger rig with the longer pen
		#z_offset = np.array([0, 0, 35])
		#This is for the smaller rig with the longer pen
		#z_offset = np.array([0,0,40])
		#This is for both rigs with the base holder
		z_offset = np.array([0,0,0])
		corner_top= corner + 20*rig_pose[:3,-2] + z_offset 
		corner_top_safe= corner + 80*rig_pose[:3,-2] + z_offset
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
		time.sleep(1.0)  # allow force filter to settle after tare
		print("Current force reading:",mctrl.ft_reading)
		input("Start pushing")
                
		for i in range(100): # making sure to get the latest joint position
			q_cur = mctrl.read_position()

		ft_record = mctrl.force_load_z(f_d)
		#print("Current force reading:",mctrl.ft_record)
		
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
try:
	mctrl.stop_egm()
except Exception as e:
	print("stop_egm warning:", e)

#mctrl.stop_egm()


###UPDATE IPAD POSE based on new corners
p_all=np.array(corners_adjusted)
np.savetxt("corners_adjusted.csv", p_all, delimiter=',')
#identify the center point and the plane
center=np.mean(p_all,axis=0)
pca = PCA()
pca.fit(p_all)
R_temp = pca.components_.T		###decreasing variance order
if R_temp[:,0]@center<0:		###correct orientation
	R_temp[:,0]=-R_temp[:,0]
if R_temp[:,-1]@R_pencil[:,-1]>0:
	R_temp[:,-1]=-R_temp[:,-1]

R_temp[:,1]=np.cross(R_temp[:,2],R_temp[:,0])
# center = center - R_temp[:,2]*thickness
# center = center - R_temp[:,0]*h/2
# center = center - R_temp[:,1]*w/2
center = center + R_temp@np.array([-h/2,-w/2,-thickness])

# Origin correction: shift origin to physical top-left corner, and correct
# Z so that Z=0 corresponds to the sheet surface (robot touches sheet at -1.5mm).
center = center + R_temp @ np.array([-30.0, 8.0, 1.5])

print("New rig R:", R_temp)
print("New rig center:", center)
np.savetxt("rig_pose.csv", H_from_RT(R_temp,center), delimiter=',')

		

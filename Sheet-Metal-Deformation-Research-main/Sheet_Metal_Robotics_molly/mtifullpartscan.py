
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from RobotRaconteur.Client import *
#import open3d as o3d
import time

############## Single scan test ################

# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.60.17:60830/?service=MTI2D")
mti_client.setExposureTime("25")

line_scan = np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])

## remove all data with z < 40
#line_scan = line_scan[:,line_scan[1]>40]
print(line_scan.shape)

np.savetxt("300_6.csv",line_scan)

plt.plot(line_scan[0],line_scan[1])
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.title('Line Test 3.00N 2_10_26')
plt.savefig("3.00N.png")
plt.show()

exit()
############## Continuous scan test using robot motion ################
# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.60.17:60830/?service=MTI2D")
mti_client.setExposureTime("25")
## rr drivers and all other drivers

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

scan_speed = 10 # mm/s

###streaming
mctrl.start_egm()
start_time=time.time()
state_flag=0
joint_recording=[]
robot_stamps=[]
mti_recording=None
mti_recording=[]

#Scan Path
starting_q = np.radians([24.9,33.8,-22,0,-34.8,3])
ending_q = np.radians([25.0,37.45,-16,0,-37.2,3])
mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])
mctrl.jog_joint_position_cmd(start_point,v=controller_params["jogging_speed"])

while True:
    if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
        break 
    res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
    if res:
        joint_angle=np.hstack((fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
        state_flag=fb_data.controller_flags
        joint_recording.append(joint_angle)
        timestamp=fb_data.time
        robot_stamps.append(timestamp)
        ###MTI scans YZ point from tool frame
        try:
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
        except Exception as e:
            if not mti_break_flag:
                print(e)
            mti_break_flag=True
mctrl.stop_egm()

# mti_recording=np.array(mti_recording)
q_out_exe=np.array(joint_recording)[:,:6]

#Create point cloud
pcd_all = o3d.geometry.PointCloud()
for i in range(len(mti_recording)):
    scanner_T = mctrl.robot.fwd(q_out_exe[i])

    ## remove all data with z < 40
    # scan_points = mti_recording[i][:,mti_recording[i][1]>40]
    
    scan_points = np.insert(mti_recording[i],1,np.zeros(len(mti_recording[i][0])),axis=0)
    scan_points[0]=scan_points[0]*-1 # reversed x-axis
    
    scan_points = scan_points.T
    scan_points = np.transpose(np.matmul(scanner_T.R,np.transpose(scan_points)))+scanner_T.p

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(scan_points)
    pcd_all+=pcd

# visualize pcd with frames
o3d.visualization.draw_geometries([pcd_all])
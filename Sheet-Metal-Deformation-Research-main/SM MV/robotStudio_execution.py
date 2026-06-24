import numpy as np
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from matplotlib import pyplot as plt

from traj_gen import get_trajectory

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



# Run it on the robot
#Define by taking Solidworks model, import to RobotStudio?
my_tool = abb.tooldata(True,abb.pose([0,0,0.1],[1,0,0,0]),abb.loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0))

def quadrant(q,robot):
	cf146=np.floor(np.array([q[0],q[3],q[5]])/(np.pi/2))
	eef=fwdkin(robot,q).p
	
	REAR=(1-np.sign((rot([0,0,1],q[0])@np.array([1,0,0]))@np.array([eef[0],eef[1],eef[2]])))/2

	LOWERARM= q[2]<-np.pi/2
	FLIP= q[4]<0

	return np.hstack((cf146,[4*REAR+2*LOWERARM+FLIP])).astype(int)

mp = abb.MotionProgram(tool=my_tool)
jt_init = abb.jointtarget(np.degrees(curve_js[0]),[0]*6) # using moveabsj to move to the initial joint angles
mp.MoveAbsJ(jt_init,abb.v1000,abb.fine)

breakpoints = np.linspace(0,len(curve_js)-1,11).astype(int) # chop it into 10 move L segments
for i in breakpoints[1:]:
    flange_T = fwdkin(robot, curve_js[i]) # get the flange pose
    cf = quadrant(curve_js[i],robot) # get the configuration
    #Need to define robot configuration
    robt = abb.robtarget(flange_T.p,R2q(flange_T.R),abb.confdata(cf[0],cf[1],cf[2],cf[3]),[0]*6) # create the robtarget, position (mm), orientation (quaternion)
    if i==len(curve_js)-1:
        mp.MoveL(robt,abb.v200,abb.fine) # last zone is fine
    else:
        mp.MoveL(robt,abb.v200,abb.z50) # using zone = z50
        
print(mp.get_program_rapid)

client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80")
log_results = client.execute_motion_program(mp) # run on the robot/robotstudio and log the results

cmd_num=log_results.data[:,1] # command number
start_id = np.where(cmd_num==2)[0][0]
curve_js_exe = np.radians(log_results.data[start_id:,2:8]) # executed joint angles
curve_p_exe = [] # executed flange position
for i in range(len(curve_js_exe)):
    curve_p_exe.append(fwdkin(robot, curve_js_exe[i]).p)
lam_exe = np.cumsum(np.linalg.norm(np.diff(curve_p_exe, axis=0), axis=1)) # executed path length
lam_exe = np.insert(lam_exe, 0, 0)

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
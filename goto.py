import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM

target = np.array([477.0 - 244.12, -98.0 - -48.95, 359.0 - 343.10])   # rig frame, mm
move_time = 30                            # seconds
TIMESTEP = 0.004

# robot + tool
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f:
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([-55.45, 0, 131.2])
tool_T = Transform(np.eye(3), Pft)

robot.R_tool=tool_T.R
robot.p_tool=tool_T.p

# rig
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

#Transformation to rig
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
target_pbr = Rbr@target + Pbr
target_Rbr = Rbr@corner_R


# start EGM
mm = abb.egm_minmax(-1e-3, 1e-3)
mp = abb.MotionProgram(egm_config=abb.EGMJointTargetConfig(mm, mm, mm, mm, mm, mm, 1000, 1000))
mp.EGMRunJoint(10, 0.05, 0.05)
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80")  # real robot
#client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80")         # RobotStudio sim
client.execute_motion_program(mp, wait=False)
egm = EGM()

def read_q():
    for i in range(20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if res:
            return np.radians(state.joint_angles)
    raise Exception("Robot communication lost")

# move
try:
    q_start = read_q()
    sols = robot6_sphericalwrist_invkin(robot, Transform(target_Rbr, target_pbr), q_start)
    q_target = sols[0]
    for q in np.linspace(q_start, q_target, int(move_time / TIMESTEP)):
        read_q()
        egm.send_to_robot(np.degrees(q))
    print("Reached target")
finally:
    client.stop_egm()
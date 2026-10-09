# This is Emma's script to get the robot to move to the corret coordinates. lowk wanna die 

#Imports
# I need to bring in all of the math and robot libraries that our program
# would need to actually process the information we give it. The notes numbered are for the lines below in order.
#  (1) Brings in a toolbox/tools we'll need
#   as np gives numpy a "nickname", we'll need np for our target, matrix multiplication, and creating the small steps of the move.
#  (2) A toolbox specifically made for robot - math tools like Transform, rotation (rot), and inverse kinematics (robot6_sphericalwrist_invkin)
#   The * means pull everything at BASE level from here, so no subtoolboxes.

import numpy as np 
from general_robotics_toolbox import * 

from general_robotics_toolbox import robotraconteur as rr_rox
# We have the general toolbox for general_robotics_toolbox, but we also specifically want this subtoolbox from it. 
# We nicknamed it rr_rox 

import abb_motion_program_exec as abb
#this library essentially allows us to talk to the abb robot by telling it 
#to be ready for what we're going to give it

from abb_robot_client.egm import EGM
# this is our live link, it's how we'll read the joint positions and send new ones

#-------------------------Settings------------------------
# Here, this is where I am going to feed it the target point that I want to get to! :D 
# REMEMBER THAT WE HAVE TO SUBTRACT IT FROM THE ORIGIN!!
origin = np.array ([244.12, -48.95, 343.10])
jogged_point = np.array([ 477.0, -98.0 , 359.0]) 
target_rig = jogged_point - origin# The point we want to get to with robot. np.array creates a vector of these values.
move_time = 30
TIMESTEP = 0.004

#---------------------------Loading to ABB ---------------
# Stole this part from ChallengeTask.py to load the robot 
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f: 
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([ -55.45, 0, 131.2])
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

#--------------------------- Configuring the rig -----------------
#Stole also from ChallengeTask.py
#We're basically giving it the information it needs about the rig?

#############################################Rig Kinematics###################################################
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

print("Pbr:", Pbr)
print("Rbr:", Rbr)

#---------Converting the Target into points for the Rig?---------------
#Here, we have to convert the coordinates in the ROBOTS FRAME to the rig's frame
# Learned how to do this in Robotics btw, look at notes. 

target_pbr = Rbr @ target_rig + Pbr #Literally P_ac = R_ab * Pbc + Pab -> where a is the robts base, b is the rig, c is our target point. 
tool_R_rig = np.array ([[ -1,0,0], [0,1,0], [0,0,-1]]).T #the orientation of the pen
target_Rbr = Rbr @ tool_R_rig #

print("Target in rig frame    :", target_rig)
print("Target in robot frame  :", target_pbr)

#-------------- Connecting to the REAL ROBOT or ABB --------
#Stole from Challenge Task, set up egm config
mm = abb.egm_minmax(-1e-3,1e-3)
egm_config = abb.EGMJointTargetConfig(mm, mm, mm, mm, mm ,mm, 1000, 1000)
mp = abb.MotionProgram(egm_config = egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)

#MAKE SURE TO UNCOMMENT ROBOT PORTION WHEN WORKING WITH THE REAL ROBOT!!
# Windows
client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80") # for simulation in RobotStudio
# Robot
#client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80") # for real robot
lognum = client.execute_motion_program(mp, wait=False)
egm = EGM()

#-----------------------MOVING---------------

def read_q():
    for i in range(20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if res:
            return np.radians(state.joint_angles)
    raise Exception("Robot communication lost")

try:
    q_start = read_q()

    sols = robot6_sphericalwrist_invkin(robot, Transform(target_Rbr, target_pbr), q_start)
    if len(sols) == 0:
        raise Exception("No joint solution found - target may be out of reach")
    q_target = sols[0]

    print("Biggest joint change (deg):", np.degrees(np.max(np.abs(q_target - q_start))))

    n_steps = int(move_time / TIMESTEP)
    for q in np.linspace(q_start, q_target, n_steps):
        read_q()
        egm.send_to_robot(np.degrees(q))

    print("Reached target")

finally:
    client.stop_egm()



 
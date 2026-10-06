import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import matplotlib.pyplot as plt
import numpy as np


num_sides = 7  
radius = 18.5  
center_x = 50.0
center_y = 50.0 
z_draw = 30.0
z_clearance = 35.0 

angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, num_sides, endpoint=False) # gets angles from formula
vertices = [ # converts from polar to cartisan
[center_x + radius * np.cos(a), center_y + radius * np.sin(a), z_draw]
    for a in angles
]

first_vert = vertices[0] # initial for refernece
approach_pt = [first_vert[0], first_vert[1], z_clearance] # hovers over the first vert
corner_p_wp = np.array([approach_pt, *vertices, first_vert, approach_pt]) 
# draws them all out, hover -> verts -> first vert (close loop) -> hover


# copied from drawStar
def calc_lam_js(curve_js,robot):
    curve_p = [] # flange position
    for i in range(len(curve_js)):
        curve_p.append(fwdkin(robot, curve_js[i]).p)
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)) # executed path length
    lam = np.insert(lam, 0, 0)
    return lam


with open("ABB_1200_5_90_robot_default_config.yml", "r") as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

Pft = np.array([-55.45, 0, 131.2])

tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

square_size = 50
dlam_des = 0.02
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

curve_p = []
curve_R = []
for i in range(corner_p_wp.shape[0]-1):
    this_seg_p = np.linspace(corner_p_wp[i], corner_p_wp[i+1], int(np.linalg.norm(corner_p_wp[i]-corner_p_wp[i+1])/dlam_des)+1)
    this_seg_R = np.tile(corner_R, (len(this_seg_p), 1, 1))
    curve_p.extend(this_seg_p[:-1])
    curve_R.extend(this_seg_R[:-1])
curve_p.append(corner_p_wp[-1])
curve_R.append(corner_R)
curve_p = np.array(curve_p)
curve_R = np.array(curve_R)

final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=",")

z_theta = 2.0549 * np.pi / 180  # Radians

vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz


for i in range(curve_p.shape[0]):
    curve_p[i] = Rbr @ curve_p[i, :] + Pbr
    

new_curve_R = [Rbr @ R for R in curve_R]


q_init = robot6_sphericalwrist_invkin(
    robot, Transform(new_curve_R[0], curve_p[0]), np.zeros(6)
)[0]

terminate_threshold = 0.001
curve_js = [q_init]

for i in range(1,len(curve_p)):
    print("-----------------")
    print(f"{i} out of {len(curve_p)}")
    this_q = robot6_sphericalwrist_invkin(robot,Transform(new_curve_R[i], curve_p[i]),curve_js[-1])[0]
    flange_T = fwdkin(robot, this_q)
    vd = curve_p[i]-flange_T.p

    assert np.linalg.norm(vd) < terminate_threshold, "The inverse kinematics gave a large error"
    curve_js.append(this_q)

print(curve_js)
print(type(curve_js))

# Trajectory Generation
TIMESTEP = 0.004  # 4 ms for EGM control


def read_position(egm):
    for _ in range(20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if res:
            return np.radians(state.joint_angles)
        else:
            #client.stop_egm()
            mp = abb.MotionProgram(egm_config=egm_config)
            mp.EGMRunJoint(10, 0.05, 0.05)
            client.stop_motion_program()
            client.execute_motion_program(mp, wait=False)
            print("Communication Lost, Reconnecting...")
    return np.radians(state.joint_angles)


def position_cmd(q, egm):
    egm.send_to_robot(np.degrees(q))


def trajectory_generate(curve_js, robot, lin_vel, lin_acc):
    lam = calc_lam_js(curve_js, robot)
    if len(lam) > 2 and lin_acc > 0:
        time_bp = np.zeros_like(lam)
        acc, vel = lin_acc, 0
        for i in range(len(lam)):
            if vel >= lin_vel:
                time_bp[i] = time_bp[i - 1] + (lam[i] - lam[i - 1]) / lin_vel
            else:
                time_bp[i] = np.sqrt(2 * lam[i] / acc)
                vel = acc * time_bp[i]
        time_bp_half, vel = [], 0
        for i in range(len(lam) - 1, -1, -1):
            if vel >= lin_vel or i <= len(lam) / 2:
                break
            else:
                time_bp_half.append(np.sqrt(2 * (lam[-1] - lam[i]) / acc))
                vel = acc * time_bp_half[-1]
        time_bp_half = np.array(time_bp_half)[::-1]
        time_bp_half = time_bp_half * -1 + time_bp_half[0]
        time_bp[-len(time_bp_half) :] = (
            time_bp[-len(time_bp_half) - 1]
            + time_bp_half
            + (lam[-len(time_bp_half)] - lam[-len(time_bp_half) - 1]) / lin_vel
        )
    else:
        time_bp = lam / lin_vel

    num_steps = int(time_bp[-1] / TIMESTEP)
    traj_q = []
    for step in range(num_steps):
        current_time = step * TIMESTEP
        seg = 0
        for i in range(len(time_bp) - 1):
            if time_bp[i] <= current_time < time_bp[i + 1]:
                seg = i
                break
        frac = (current_time - time_bp[seg]) / (time_bp[seg + 1] - time_bp[seg])
        q_des = frac * curve_js[seg + 1] + (1 - frac) * curve_js[seg]
        traj_q.append(q_des)
    return np.array(traj_q), np.array(time_bp)


tool_vel = 5  # mm/s
tool_acc = 5  # mm/s^2

fullruntraj_q, _ = trajectory_generate(
    curve_js, robot, lin_vel=tool_vel, lin_acc=tool_acc
)

mm = abb.egm_minmax(-1e-3, 1e-3)
egm_config = abb.EGMJointTargetConfig(mm, mm, mm, mm, mm, mm, 1000, 1000)
mp = abb.MotionProgram(egm_config=egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)

# Client execution for RobotStudio
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80")
lognum = client.execute_motion_program(mp, wait=False)

egm = EGM()
print("Robot start moving. EGM is running")

q_start = read_position(egm)
q_all = np.linspace(q_start, curve_js[0], num=100)
traj_q, _ = trajectory_generate(
    q_all, robot, lin_vel=tool_vel, lin_acc=tool_acc
)

for i in range(len(traj_q)):
    read_position(egm)
    position_cmd(traj_q[i], egm)

print("Robot in Start Position")

EE_pos = []
for i in range(len(fullruntraj_q)):
    read_q = read_position(egm)

    current_EEpos = fwdkin(robot, read_q).p
    coordinates_in_rig_frame = Rbr.T @ (current_EEpos - Pbr)
    EE_pos.append([coordinates_in_rig_frame[0], coordinates_in_rig_frame[1]])

    position_cmd(fullruntraj_q[i], egm)

EE_pos = np.array(EE_pos)
#client.stop_egm()

np.savetxt("EE_pos.csv", EE_pos, delimiter=",")

plt.figure()
plt.plot(EE_pos[:, 0], EE_pos[:, 1])
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Executed Robot X-Y Polygon Path at z = -1.25")
plt.axis("equal")
plt.grid(True)
plt.show()
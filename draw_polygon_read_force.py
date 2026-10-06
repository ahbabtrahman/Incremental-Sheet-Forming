import sys
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import abb_motion_program_exec as abb
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
from RobotMotionController import *
from RobotRaconteur.Client import *
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
from robot_def import *

sys.path.append("toolbox")
sys.path.append("robot_motion")

from sklearn.decomposition import PCA
from rpi_ati_net_ft import *


# =========================================================================================
num_sides = 7  
radius = 18.5  
center_x = 50.0
center_y = 50.0 
z_draw = -65.0 # EDIT
z_clearance = -60.0  

angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, num_sides, endpoint=False) # gets angles from formula
vertices = [ # converts from polar to cartisan
[center_x + radius * np.cos(a), center_y + radius * np.sin(a), z_draw]
    for a in angles
]

first_vert = vertices[0] # initial for refernece
approach_pt = [first_vert[0], first_vert[1], z_clearance] # hovers over the first vert
corner_p_wp = np.array([approach_pt, *vertices, first_vert, approach_pt]) 
# draws them all out, hover -> verts -> first vert (close loop) -> hover

# =========================================================================================
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

# ===========================================================================================
H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=",")
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_T = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1])).T

RR_ati_cli = RRN.ConnectService("rr+tcp://localhost:59823?service=ati_sensor")
ati = RR_ati_cli

print(Htransform.__module__)
tool_T_csv = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T_csv, delimiter=",")

robot_tb = robot_obj("ABB_1200_5_90", "ABB_1200_5_90_robot_default_config.yml", tool_file_path="rig_pen.csv")
robot_tb.R_tool = tool_T.R
robot_tb.p_tool = tool_T.p

abb_robot_ip = "192.168.60.101"
TIMESTEP = 0.004
controller_params = {
    "force_ctrl_damping": 60.0,
    "force_epsilon": 0.1,
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1,
    "load_speed": 20.0,
    "unload_speed": 1.0,
    "settling_time": 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 50,
    "jogging_acc": 10,
    "force_filter_alpha": 0.9,
}

rig_pose = np.loadtxt("rig_pose.csv", delimiter=",")
mctrl = MotionController(
    robot_tb,
    rig_pose,
    H_pentip2ati,
    controller_params,
    TIMESTEP,
    FORCE_PROTECTION=10,
    RR_ati_cli=RR_ati_cli,
    abb_robot_ip=abb_robot_ip,
)

# =========================================================================================
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

# Start EGM session via mctrl
mctrl.start_egm()

def read_position():
    for _ in range(20):
        res, state = mctrl.egm.receive_from_robot(timeout=0.1)
        if res:
            return np.radians(state.joint_angles)
        else:
            mctrl.stop_egm()
            mctrl.start_egm()
            print("Communication Lost, Reconnecting...")
    return np.radians(state.joint_angles)

def position_cmd(q):
    mctrl.egm.send_to_robot(np.degrees(q))

# Move robot to starting pose smoothly
q_start = read_position()
q_all = np.linspace(q_start, curve_js[0], num=100)
traj_q_start, _ = trajectory_generate(q_all, robot, lin_vel=tool_vel, lin_acc=tool_acc)

for q_cmd in traj_q_start:
    read_position()
    position_cmd(q_cmd)

print("Robot in Start Position")

# Zero/Tare force sensor
time.sleep(0.1)
ati.setf_param("set_tare", RR.VarValue(True, "bool"))
time.sleep(0.5)

# =========================================================================================
EE_pos, fz, t_log = [], [], []
start_time = time.time()

for q_cmd in fullruntraj_q:
    read_q = read_position()
    p = Rbr.T @ (fwdkin(robot, read_q).p - Pbr)  # Tip position in rig frame
    EE_pos.append(p)
    
    # Directly query RobotRaconteur ATI client
    w = ati.wrench
    raw_ft = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])
    
    # Transform to tool tip frame and get Fz 
    tip_wrench = ad_T @ raw_ft
    fz.append(float(tip_wrench[2]))
    
    t_log.append(time.time() - start_time)
    position_cmd(q_cmd)

EE_pos = np.array(EE_pos)
fz = np.array(fz)
t_log = np.array(t_log)

mctrl.stop_egm()

# =========================================================================================
np.savetxt("polygon_force.csv", np.column_stack([t_log, fz, EE_pos[:, 0], EE_pos[:, 1]]), delimiter=",")

plt.figure()
plt.plot(t_log, fz)
plt.xlabel("Time (s)")
plt.ylabel("Force Z (N)")
plt.title("Force vs Time")
plt.grid(True)

plt.figure()
sc = plt.scatter(EE_pos[:, 0], EE_pos[:, 1], c=fz, cmap="viridis", s=5)
plt.colorbar(sc, label="Force Z (N)")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Executed Polygon Path (colored by force)")
plt.axis("equal")
plt.grid(True)
plt.show()

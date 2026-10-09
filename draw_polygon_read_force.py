print("Start of Program")

# Imports
import sys
sys.path.append("/documents/github/Sheet-Metal-Deformation-Research/SM MV/")
sys.path.append("toolbox")
sys.path.append("robot_motion")

import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from RobotRaconteur.Client import *
from general_robotics_toolbox import *
from utils import *                      # NOT pip
from robot_def import *                  # NOT pip
from RobotMotionController import *      # NOT pip
from general_robotics_toolbox import robotraconteur as rr_rox

def Htransform(R, p):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p
    return H

def H_from_RT(R, T):
    return Htransform(R, T)

# Geometry
num_sides   = 7
radius      = 18.5
center_x    = 50.0
center_y    = 50.0
z_draw      = -0.5   # edit
z_clearance = 5.0    # Hover height

angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, num_sides, endpoint=False)
vertices = [[center_x + radius * np.cos(a), center_y + radius * np.sin(a), z_draw] for a in angles]

first_vert = vertices[0]
approach_pt = [first_vert[0], first_vert[1], z_clearance]
corner_p_wp = np.array([approach_pt, *vertices, first_vert, approach_pt])
# hover -> verts -> first vert (close loop) -> hover

# Settings
TIMESTEP = 0.004          # 4 ms EGM interval
tool_vel = 5              # mm/s
tool_acc = 5              # mm/s^2
dlam_des = 0.02           # mm, path discretization
abb_robot_ip = '127.0.0.1:80'   

controller_params = {
    "force_ctrl_damping": 60.0,
    "force_epsilon": 0.1,
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1,
    "load_speed": 20.0,
    "unload_speed": 1.0,
    'settling_time': 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 50,
    "jogging_acc": 10,
    'force_filter_alpha': 0.9
}

# Helper functions
def calc_lam_js(curve_js, robot):
    curve_p = [fwdkin(robot, q).p for q in curve_js]   # flange position
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
    return np.insert(lam, 0, 0)


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
        time_bp[-len(time_bp_half):] = (
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
        traj_q.append(frac * curve_js[seg + 1] + (1 - frac) * curve_js[seg])
    return np.array(traj_q), np.array(time_bp)


def read_position():
    state = None
    for _ in range(20):
        res, state = mctrl.egm.receive_from_robot(timeout=0.1)
        if not res:
            mctrl.stop_egm()
            mctrl.start_egm()
            print("Communication Lost, Reconnecting")
        else:
            break
    return np.radians(state.joint_angles)


def position_cmd(q):
    mctrl.egm.send_to_robot(np.degrees(q))

# Robot models 
Pft = np.array([-55.45, 0, 131.2])   # flange -> pen tip

# Robot Raconteur model
with open("ABB_1200_5_90_robot_default_config.yml", "r") as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# Toolbox model
np.savetxt("rig_pen.csv", Htransform(np.eye(3), Pft), delimiter=',')
robot_tb = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",
                     tool_file_path="rig_pen.csv")

# Rig kinematics 
rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = 2.0549 * np.pi / 180
Rz = rot(rig_pose[0:3, 2], z_theta)
Pbr = rig_pose[0:3, -1]
Rbr = rig_pose[0:3, 0:3] @ Rz

# Force sensor + controller 
H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_ati2pentip_T = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1])).T
RR_ati_cli = RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')

mctrl = MotionController(robot_tb, rig_pose, H_pentip2ati, controller_params, TIMESTEP,
                         FORCE_PROTECTION=10, RR_ati_cli=RR_ati_cli, abb_robot_ip=abb_robot_ip)
print("Robot obj created")

# Build the trajectory (open loop)
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

curve_p, curve_R = [], []
for i in range(corner_p_wp.shape[0] - 1):
    n = int(np.linalg.norm(corner_p_wp[i] - corner_p_wp[i + 1]) / dlam_des) + 1
    seg_p = np.linspace(corner_p_wp[i], corner_p_wp[i + 1], n)
    curve_p.extend(seg_p[:-1])
    curve_R.extend(np.tile(corner_R, (len(seg_p), 1, 1))[:-1])
curve_p.append(corner_p_wp[-1])
curve_R.append(corner_R)
curve_p = np.array(curve_p)
curve_R = np.array(curve_R)

for i in range(curve_p.shape[0]):               # rig frame -> base frame
    curve_p[i] = Rbr @ curve_p[i, :] + Pbr
new_curve_R = [Rbr @ R for R in curve_R]

q_init = robot6_sphericalwrist_invkin(robot, Transform(new_curve_R[0], curve_p[0]), np.zeros(6))[0]
curve_js = [q_init]
for i in range(1, len(curve_p)):
    if i % 500 == 0:
        print(f"IK {i} out of {len(curve_p)}")
    this_q = robot6_sphericalwrist_invkin(robot, Transform(new_curve_R[i], curve_p[i]), curve_js[-1])[0]
    err = curve_p[i] - fwdkin(robot, this_q).p
    assert np.linalg.norm(err) < 0.001, "The inverse kinematics gave a large error"
    curve_js.append(this_q)

fullruntraj_q, _ = trajectory_generate(curve_js, robot, lin_vel=tool_vel, lin_acc=tool_acc)
print(f"Trajectory ready: {len(fullruntraj_q)} steps ({len(fullruntraj_q) * TIMESTEP:.1f} s)")

# Main
t_log, fz_log, EE_pos = [], [], []
aborted = False

print("Starting EGM...")
mctrl.start_egm()

try:
    # Move to the hover point above the first vertex
    mctrl.jog_joint_position_cmd(curve_js[0], v=controller_params["jogging_speed"])
    input("At safe approach (hovering). Press Enter to tare and draw...")

    # Zero the FT sensor while the pen is NOT touching
    for _ in range(2):
        time.sleep(0.1)
        mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
    time.sleep(1.0)   # let the force filter settle
    print("Tared. Force reading:", ad_ati2pentip_T @ mctrl.ft_reading)

    # Draw (open loop) and read force every step
    t0 = time.time()
    for q_cmd in fullruntraj_q:
        q_now = read_position()
        tip = robot_tb.fwd(q_now)
        p_rig = Rbr.T @ (tip.p - Pbr)                    # tip position in rig frame

        ft_tip = ad_ati2pentip_T @ mctrl.ft_reading      # sensor -> pen tip
        fz_now = float(ft_tip[-1])                       # same index as star script

        if np.linalg.norm(ft_tip[3:]) > mctrl.FORCE_PROTECTION:
            print("force too large:", ft_tip[3:])
            aborted = True
            break
        if time.time() - mctrl.last_ft_time > 0.1:
            print("force reading lost")
            aborted = True
            break

        t_log.append(time.time() - t0)
        fz_log.append(fz_now)
        EE_pos.append(p_rig)

        position_cmd(q_cmd)

    # If aborted, lift straight up so the pen leaves the sheet
    if aborted:
        q_now = read_position()
        tip = robot_tb.fwd(q_now)
        q_lift = robot_tb.inv(tip.p + 10 * Rbr[:, 2], tip.R, q_now)[0]
        mctrl.jog_joint_position_cmd(q_lift, v=controller_params["jogging_speed"])

    print("Polygon drawing complete." if not aborted else "Aborted - pen lifted.")

except (Exception, KeyboardInterrupt) as e:
    print("Error:", e)
    traceback.print_exc()

finally:
    try:
        mctrl.stop_egm()
    except Exception as e:
        print("stop_egm warning:", e)

# Save + plot (after EGM is stopped)
if len(fz_log) == 0:
    print("No data Recorded")
else:
    t_log = np.array(t_log)
    fz_log = np.array(fz_log)
    EE_pos = np.array(EE_pos)
    np.savetxt("polygon_force.csv", np.column_stack([t_log, fz_log, EE_pos]), delimiter=',')

    plt.figure()
    plt.plot(t_log, fz_log)
    plt.xlabel("Time (s)")
    plt.ylabel("Force Z (N)")
    plt.title("Force vs Time")
    plt.grid(True)

    plt.figure()
    sc = plt.scatter(EE_pos[:, 0], EE_pos[:, 1], c=fz_log, cmap="viridis", s=5)
    plt.colorbar(sc, label="Force Z (N)")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.title("Executed Polygon Path (colored by force)")
    plt.axis("equal")
    plt.grid(True)

    plt.show()

print("End of Program")

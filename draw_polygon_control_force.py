print("Start of Program")

# Imports
import sys
sys.path.append("/documents/github/Sheet-Metal-Deformation-Research/SM MV/")
sys.path.append("toolbox")
sys.path.append("robot_motion")

import time
import traceback
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from RobotRaconteur.Client import *
from general_robotics_toolbox import *
from utils import *                      # NOT pip
from robot_def import *                  # NOT pip
from RobotMotionController import *      # NOT pip

if 'Htransform' not in globals():
    def Htransform(R, p):
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = p
        return H

if 'H_from_RT' not in globals():
    def H_from_RT(R, T):
        return Htransform(R, T)

# Geometry
num_sides   = 7
radius      = 18.5
center_x    = 50.0
center_y    = 50.0
z_clearance = 5.0    # hover height
z_start     = 1.0    # edit

angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, num_sides, endpoint=False)
verts_xy = np.array([[center_x + radius * np.cos(a), center_y + radius * np.sin(a)] for a in angles])
loop_xy = np.vstack([verts_xy, verts_xy[0]])

# FORCE PER SIDE 
forces = [-1.0, -1.2, -1.5, -1.8, -2.0, -2.2, -2.5]
assert len(forces) == num_sides, "need exactly one force per side"

# Settings 
vd       = 2.0        # drawing speed (mm/s)
Kf       = 0.05       # force -> z gain
Z_MIN    = -3.0       # rig-frame z floor (mm)
CONTACT_TOL     = 0.1   # tolerance during first contact
CONTACT_TIMEOUT = 15.0  # abort if target force is not reached
TIMESTEP = 0.004      # 4 ms EGM interval
abb_robot_ip = '192.168.60.101'   

controller_params = {
    "force_ctrl_damping": 60.0,
    "force_epsilon": 0.1,         # below this = not touching
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1,          # N/s force ramp after first contact
    "load_speed": 5.0,            # mm/s descent speed until contact
    "unload_speed": 1.0,
    'settling_time': 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 50,
    "jogging_acc": 10,
    'force_filter_alpha': 0.9
}

# Robot, rig, sensor, controller 
Pft = np.array([-55.45, 0, 131.2])   # flange -> pen tip
np.savetxt("rig_pen.csv", Htransform(np.eye(3), Pft), delimiter=',')
robot_tb = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",
                     tool_file_path="rig_pen.csv")

rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = 2.0549 * np.pi / 180
Rz = rot(rig_pose[0:3, 2], z_theta)
Pbr = rig_pose[0:3, -1]
Rbr = rig_pose[0:3, 0:3] @ Rz

H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',')
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_ati2pentip_T = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1])).T
RR_ati_cli = RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')

mctrl = MotionController(robot_tb, H_from_RT(Rbr, Pbr), H_pentip2ati, controller_params, TIMESTEP,
                         FORCE_PROTECTION=10, RR_ati_cli=RR_ati_cli, abb_robot_ip=abb_robot_ip)
print("Robot obj created")

corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
corner_R_base = Rbr @ corner_R

# Helper functions 
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


def jog_to_rig_point(p_rig):
    q_read = read_position()
    q = robot_tb.inv(Rbr @ np.asarray(p_rig) + Pbr, corner_R_base, q_read)[0]
    mctrl.jog_joint_position_cmd(q, v=controller_params["jogging_speed"])


def tare():
    for _ in range(2):
        time.sleep(0.1)
        mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
    time.sleep(1.0)   # let the force filter settle


def read_fz():
    ft_tip = ad_ati2pentip_T @ mctrl.ft_reading
    if np.linalg.norm(ft_tip[3:]) > mctrl.FORCE_PROTECTION:
        raise RuntimeError(f"force too large: {ft_tip[3:]}")
    if time.time() - mctrl.last_ft_time > 0.1:
        raise RuntimeError("force reading lost")
    return float(ft_tip[-1])


def step(q_now, xy_cmd, dz=0.0):
    tip = deepcopy(mctrl.ipad_pose_T.inv() * robot_tb.fwd(q_now))
    tip.p[2] = max(tip.p[2] + dz, Z_MIN)
    tip.p[:2] = xy_cmd
    nxt = mctrl.ipad_pose_T * tip
    position_cmd(robot_tb.inv(nxt.p, nxt.R, q_now)[0])
    return tip.p


def make_contact(xy, fz_target):
    p = mctrl.params
    touch_t = None
    t0 = time.time()
    while True:
        if time.time() - t0 > CONTACT_TIMEOUT:
            raise RuntimeError("make_contact timed out")
        q = read_position()
        fz = read_fz()
        if abs(fz) < p['force_epsilon'] and touch_t is None:
            step(q, xy, dz=-p['load_speed'] * TIMESTEP)                  # descend
        else:
            if touch_t is None:
                touch_t = time.time()
                print("contact made")
            fz_ref = max(fz_target, -(time.time() - touch_t) * p['trapzoid_slope'])
            dz = mctrl.force_impedence_ctrl(fz_ref - fz) * TIMESTEP * Kf     # CONTROL FORCE
            step(q, xy, dz)
            if fz_ref == fz_target and abs(fz - fz_target) < CONTACT_TOL:
                print("target force reached")
                return


def draw_edge(xy0, xy1, fz_target, speed, record):
    L = np.linalg.norm(xy1 - xy0)
    u = (xy1 - xy0) / L
    s = 0.0
    while True:
        q = read_position()
        fz = read_fz()                                                      # READ FORCE
        dz = mctrl.force_impedence_ctrl(fz_target - fz) * TIMESTEP * Kf     # CONTROL FORCE
        p_rig = step(q, xy0 + u * min(s, L), dz)
        record.append(np.hstack(([time.time(), fz, fz_target], p_rig, np.degrees(q))))
        if s >= L:
            break
        s += speed * TIMESTEP       


# Main 
edge_records = []
print("Starting EGM...")
mctrl.start_egm()

try:
    # Hover above the first vertex
    jog_to_rig_point([*loop_xy[0], z_clearance])
    input("At safe approach (hovering). Press Enter...")

    # Go just above the sheet, tare while NOT touching
    jog_to_rig_point([*loop_xy[0], z_start])
    input("Just above sheet. Press Enter to tare and make contact...")
    tare()
    print("Tared. Force reading:", ad_ati2pentip_T @ mctrl.ft_reading)

    # Touch down and ramp to the first side's force
    make_contact(loop_xy[0], forces[0])

    # Draw each side at its own force
    for k in range(num_sides):
        print(f"Side {k + 1}/{num_sides}: target {forces[k]} N")
        rec = []
        try:
            draw_edge(loop_xy[k], loop_xy[k + 1], forces[k], vd, rec)
        finally:
            edge_records.append(np.array(rec))   # keep partial data if a side aborts

    # Retract
    q_now = read_position()
    tip = mctrl.ipad_pose_T.inv() * robot_tb.fwd(q_now)
    jog_to_rig_point([tip.p[0], tip.p[1], z_clearance])
    print("Polygon drawing complete.")

except (Exception, KeyboardInterrupt) as e:
    print("Error:", e)
    traceback.print_exc()
    try:   # lift the pen off the sheet before shutting EGM down
        q_now = read_position()
        tip = mctrl.ipad_pose_T.inv() * robot_tb.fwd(q_now)
        jog_to_rig_point([tip.p[0], tip.p[1], z_clearance])
    except Exception as e2:
        print("Could not lift pen:", e2)

finally:
    try:
        mctrl.stop_egm()
    except Exception as e:
        print("stop_egm warning:", e)

# Save + plot (after EGM is stopped)
edge_records = [r for r in edge_records if len(r) > 0]
if len(edge_records) == 0:
    print("No data Recorded")
else:
    record = np.vstack(edge_records)
    # columns: time, fz, fz_target, x, y, z, 6 joint angles (deg)
    np.savetxt("polygon_forcectrl_record.csv", record, delimiter=',')
    t0 = record[0, 0]

    # Force vs time, one curve per side, with its target as a dashed line
    plt.figure()
    for k, r in enumerate(edge_records):
        ln, = plt.plot(r[:, 0] - t0, r[:, 1], label=f"Side {k + 1} ({r[0, 2]} N)")
        plt.hlines(r[0, 2], r[0, 0] - t0, r[-1, 0] - t0, colors=ln.get_color(), linestyles='--', linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Force Z (N)")
    plt.title("Force vs Time per Side (dashed = target)")
    plt.legend(fontsize=7)
    plt.grid(True)

    # Achieved vs desired 
    plt.figure()
    des = [r[0, 2] for r in edge_records]
    ach = [np.mean(r[len(r) // 2:, 1]) for r in edge_records]
    plt.plot(des, ach, 'o-', label="achieved")
    plt.plot(des, des, 'k--', label="ideal")
    plt.xlabel("Desired force (N)")
    plt.ylabel("Achieved force (N)")
    plt.title("Achieved vs Desired Force")
    plt.legend()
    plt.grid(True)

    # XY path colored by z
    plt.figure()
    sc = plt.scatter(record[:, 3], record[:, 4], c=record[:, 5], cmap="viridis", s=5)
    plt.colorbar(sc, label="Z (mm)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Polygon Path (colored by Z)")
    plt.axis("equal")
    plt.grid(True)

    plt.show()

print("End of Program")
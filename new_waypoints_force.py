import sys
import time
import traceback
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RobotRaconteur.Client import *
from general_robotics_toolbox import *

sys.path.append('toolbox')
sys.path.append('robot_motion')
from robot_def import *
from utils import *
from RobotMotionController import *

if 'Htransform' not in globals():
    def Htransform(R, p):
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = p
        return H

if 'H_from_RT' not in globals():
    def H_from_RT(R, T):
        return Htransform(R, T)

# USER SETTINGS
filename = "bpp_triple_serpentine.xlsx" # EDIT
sheet_name = "line1"

# X and Y rig offsets for shifting drawing position inside the rig
rigoffset_x = 0.0
rigoffset_y = 0.0

dlam_des = 0.07             # mm between planned points (keep <= 0.1)
tool_vel = 2                # mm/s
tool_acc = 2                # mm/s^2
TIMESTEP = 0.004            # 4 ms EGM interval

# EDIT MUST MATCH ZCALIB!!!!
Pft = np.array([-55.755, 0, 130.05])
Z_THETA_DEG = 2.0549

# New thing to try and fix offset
# Leave at 0.0 on the first run. After a run, try suggested value
Z_SURFACE_TRIM = 0.0

abb_robot_ip = '192.168.60.101'
ATI_SERVICE_URL = 'rr+tcp://localhost:59823?service=ati_sensor'

# Force control parameters
FZ_DES = -1.0
KF = 0.05
Z_ENGAGE = 1.0              # mm: force control starts when planned z <= this
Z_START = 1.0               # mm: "just above sheet" height to tare at
Z_MIN = -10.0               # mm: rig-frame z floor safety limit
CONTACT_TOL = 0.5
CONTACT_TIMEOUT = 15.0
FORCE_PROTECTION = 10

controller_params = {
    "force_ctrl_damping": 20.0, # decreased from 60 to 20
    "force_epsilon": 0.1,
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1,
    "load_speed": 5.0,
    "unload_speed": 1.0,
    'settling_time': 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 10, # was 50, changed bc that seemed too high
    "jogging_acc": 10,
    'force_filter_alpha': 0.9
}

try:
    H_saved = np.loadtxt("rig_pen.csv", delimiter=',')
    if np.linalg.norm(H_saved[:3, 3] - Pft) > 1e-6:
        sys.exit(f"STOP: rig_pen.csv has Pft = {H_saved[:3, 3]} but this script uses {Pft}.\n"
                 "rig_pose.csv was built with a different tool. Make Pft match zcalib and re-run zcalib.")
except OSError:
    print("Warning: rig_pen.csv not found, cannot verify Pft against zcalib.")

# Rig kinematics
final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = np.radians(Z_THETA_DEG)
Rz = rot(final_rig_pose[0:3, 2], z_theta)
Rbr = final_rig_pose[0:3, 0:3] @ Rz
Pbr = final_rig_pose[0:3, -1] + Rbr[:, 2] * Z_SURFACE_TRIM   # move origin along rig z
rig_T = Transform(Rbr, Pbr)

# Sensor, robot, controller
H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',') # EDIT FOR WHATEVER CSV FILE
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_ati2pentip_T = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1])).T
RR_ati_cli = RRN.ConnectService(ATI_SERVICE_URL)

tool_T = Transform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", Htransform(np.eye(3), Pft), delimiter=',')
robot = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml", tool_file_path="rig_pen.csv")
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p
print("Tool offset in use (must match zcalib):", Pft)
print("Z_THETA_DEG:", Z_THETA_DEG, " Z_SURFACE_TRIM:", Z_SURFACE_TRIM)

mctrl = MotionController(robot, H_from_RT(Rbr, Pbr), H_pentip2ati, controller_params, TIMESTEP,
                         FORCE_PROTECTION=FORCE_PROTECTION, RR_ati_cli=RR_ati_cli, abb_robot_ip=abb_robot_ip)

# Path planning
data = pd.read_excel(filename, sheet_name=sheet_name, header=None).to_numpy()
waypoints = data[:, 0:3].astype(float)
waypoints[:, 0] += rigoffset_x
waypoints[:, 1] += rigoffset_y
np.savetxt("gcode_coordinates.csv", waypoints, delimiter=',')

# Hover height comes from the sheet (first row)
# Waypoint z is in the rig frame: z = 0 is the sheet surface, so do NOT add
# the hover height to the waypoints. Fix the rig z (Z_SURFACE_TRIM) instead.
startheight = waypoints[0, 2]
print(f"Hover height from sheet: {startheight} mm")

def generate_robot_joints_along_path(waypoints):
    corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T    # pen pointing down
    R_cmd = Rbr @ corner_R

    pts = []
    for i in range(len(waypoints) - 1):
        n = int(np.linalg.norm(waypoints[i + 1] - waypoints[i]) / dlam_des) + 1
        seg = np.linspace(waypoints[i], waypoints[i + 1], n)
        pts.extend(seg[:-1])
    pts.append(waypoints[-1])
    p_rig = np.array(pts)
    p_base = (Rbr @ p_rig.T).T + Pbr

    def solve(p, seed):
        sols = robot.inv(p, R_cmd, seed)
        if len(sols) == 0:
            raise RuntimeError(f"No IK solution at {p}")
        return sols[0]

    curve_js = [solve(p_base[0], np.zeros(6))]
    for i in range(1, len(p_base)):
        if i % 20000 == 0:
            print(f"IK {i} out of {len(p_base)}")
        q = solve(p_base[i], curve_js[-1])
        assert np.linalg.norm(p_base[i] - robot.fwd(q).p) < 0.001, "The inverse kinematics gave a large error"
        assert np.max(np.abs(q - curve_js[-1])) < 0.1, f"Joint jump at point {i}"
        curve_js.append(q)

    lam = np.insert(np.cumsum(np.linalg.norm(np.diff(p_rig, axis=0), axis=1)), 0, 0)
    return p_rig, np.array(curve_js), lam, R_cmd


st = time.perf_counter()
p_rig_plan, curve_js, lam_planned, R_cmd = generate_robot_joints_along_path(waypoints)
print(f"Planned {len(curve_js)} points, path length {lam_planned[-1]:.1f} mm, IK {time.perf_counter() - st:.1f} s")


def trapezoid_times(L, v, a, dt):
    if a <= 0:
        T = L / v
        t = np.append(np.arange(0, T, dt), T)
        return t, np.minimum(v * t, L)
    d_a = v ** 2 / (2 * a)
    if L >= 2 * d_a:
        t_a, t_c, v_pk = v / a, (L - 2 * d_a) / v, v
    else:
        t_a, t_c = np.sqrt(L / a), 0.0
        v_pk = a * t_a
    T = 2 * t_a + t_c
    t = np.append(np.arange(0, T, dt), T)
    s_a = 0.5 * a * t_a ** 2
    s = np.where(t < t_a, 0.5 * a * t ** 2,
        np.where(t < t_a + t_c, s_a + v_pk * (t - t_a),
                 L - 0.5 * a * (T - t) ** 2))
    return t, np.clip(s, 0, L)


def trajectory_generate(curve_js, p_rig, lam, lin_vel, lin_acc):
    t, s = trapezoid_times(lam[-1], lin_vel, lin_acc, TIMESTEP)
    traj_q = np.column_stack([np.interp(s, lam, curve_js[:, j]) for j in range(curve_js.shape[1])])
    traj_p = np.column_stack([np.interp(s, lam, p_rig[:, j]) for j in range(3)])
    return traj_q, traj_p, t


fullruntraj_q, traj_p_rig, fullruntime_bp = trajectory_generate(curve_js, p_rig_plan, lam_planned, tool_vel, tool_acc)
print(f"Trajectory ready: {len(fullruntraj_q)} steps ({len(fullruntraj_q) * TIMESTEP:.1f} s)")

np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=',')
np.savetxt("fullruntime_bp.csv", fullruntime_bp, delimiter=',')
np.savetxt("curve_js.csv", curve_js, delimiter=',')
np.savetxt("lam_planned.csv", lam_planned, delimiter=',')
np.savetxt("pos_check.csv", traj_p_rig, delimiter=',')
print("Path data saved")

# Execution helpers
t_log, fz_log, fzdes_log, zoff_log, planz_log, EE_pos, q_log = [], [], [], [], [], [], []
t_start = time.time()


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


def tare():
    for _ in range(2):
        time.sleep(0.1)
        mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
    time.sleep(1.0)
    print("Tared. Force reading:", ad_ati2pentip_T @ mctrl.ft_reading)


def read_fz():
    ft_tip = ad_ati2pentip_T @ mctrl.ft_reading
    if np.linalg.norm(ft_tip[3:]) > mctrl.FORCE_PROTECTION:
        raise RuntimeError(f"Force too large: {ft_tip[3:]}")
    if time.time() - mctrl.last_ft_time > 0.1:
        raise RuntimeError("Force reading lost")
    return float(ft_tip[-1])


def log_sample(q, fz, fz_des, plan_z):
    p_tip = Rbr.T @ (robot.fwd(q).p - Pbr)
    t_log.append(time.time() - t_start)
    fz_log.append(fz)
    fzdes_log.append(fz_des)
    zoff_log.append(p_tip[2] - plan_z)
    planz_log.append(plan_z)
    EE_pos.append(p_tip)
    q_log.append(np.degrees(q))


def step(q_now, xy_cmd, dz=0.0):
    tip = deepcopy(rig_T.inv() * robot.fwd(q_now))
    tip.p[2] = max(tip.p[2] + dz, Z_MIN)
    tip.p[:2] = xy_cmd
    nxt = rig_T * tip
    sols = robot.inv(nxt.p, nxt.R, q_now)
    if len(sols) == 0:
        raise RuntimeError("No IK solution for corrected point")
    position_cmd(sols[0])


def make_contact(xy, fz_target, plan_z):
    p = controller_params
    touch_t = None
    t0 = time.time()
    while True:
        if time.time() - t0 > CONTACT_TIMEOUT:
            raise RuntimeError("make_contact timed out")
        q = read_position()
        fz = read_fz()
        if abs(fz) < p['force_epsilon'] and touch_t is None:
            log_sample(q, fz, 0.0, plan_z)
            step(q, xy, dz=-p['load_speed'] * TIMESTEP)
        else:
            if touch_t is None:
                touch_t = time.time()
                print("Contact made")
            fz_ref = max(fz_target, -(time.time() - touch_t) * p['trapzoid_slope'])
            log_sample(q, fz, fz_ref, plan_z)
            dz = mctrl.force_impedence_ctrl(fz_ref - fz) * TIMESTEP * KF
            step(q, xy, dz)
            if fz_ref == fz_target and abs(fz - fz_target) < CONTACT_TOL:
                print("Target force reached")
                return


# Execute
aborted = False
print("Starting EGM...")
mctrl.start_egm()

try:
    # Jog to the first planned point (hover height from the sheet)
    mctrl.jog_joint_position_cmd(curve_js[0], v=controller_params["jogging_speed"])
    input(f"At start (hovering at z={startheight} mm). Check the pen is about that high. Press Enter...")

    # If the path starts on the sheet, go just above it before taring
    if traj_p_rig[0, 2] <= Z_ENGAGE:
        p_above = traj_p_rig[0].copy()
        p_above[2] = Z_START
        q_above = robot.inv(Rbr @ p_above + Pbr, R_cmd, read_position())[0]
        mctrl.jog_joint_position_cmd(q_above, v=controller_params["jogging_speed"])
        input(f"Just above sheet (z={Z_START} mm). Press Enter to tare and make contact...")
    tare()

    in_contact = False
    t_start = time.time()

    for i in range(len(fullruntraj_q)):
        p_rig = traj_p_rig[i]

        # If Z is low enough to engage the sheet, use force control
        if p_rig[2] <= Z_ENGAGE:
            if not in_contact:
                tare()
                make_contact(p_rig[:2], FZ_DES, p_rig[2])
                in_contact = True
            read_q = read_position()
            fz_now = read_fz()
            dz = mctrl.force_impedence_ctrl(FZ_DES - fz_now) * TIMESTEP * KF
            log_sample(read_q, fz_now, FZ_DES, p_rig[2])
            step(read_q, p_rig[:2], dz)
        else:
            # Hover / retract move
            if in_contact:
                mctrl.jog_joint_position_cmd(fullruntraj_q[i], v=controller_params["jogging_speed"])
                in_contact = False
            read_q = read_position()
            fz_now = read_fz()
            log_sample(read_q, fz_now, 0.0, p_rig[2])
            position_cmd(fullruntraj_q[i])

    print("Drawing complete.")

except (Exception, KeyboardInterrupt) as e:
    aborted = True
    print("Error:", e)
    traceback.print_exc()

finally:
    if aborted:
        try:
            q_now = read_position()
            tip = robot.fwd(q_now)
            q_lift = robot.inv(tip.p + 10 * Rbr[:, 2], tip.R, q_now)[0]
            mctrl.jog_joint_position_cmd(q_lift, v=controller_params["jogging_speed"])
            print("Aborted - pen lifted.")
        except Exception as e2:
            print("Could not lift pen:", e2)
    try:
        mctrl.stop_egm()
    except Exception as e:
        print("stop_egm warning:", e)

# Save + plot
if len(fz_log) == 0:
    print("No data Recorded")
else:
    t_arr = np.array(t_log)
    fz_arr = np.array(fz_log)
    EE_arr = np.array(EE_pos)
    q_arr = np.array(q_log)

    np.savetxt("FTData.csv", np.column_stack([t_arr, fz_arr, fzdes_log, zoff_log]), delimiter=',')
    np.savetxt("JointData.csv", np.column_stack([t_arr, q_arr]), delimiter=',')
    np.savetxt("path_force.csv", np.column_stack([t_arr, fz_arr, EE_arr]), delimiter=',')
    np.savetxt("curvejsexedata.csv", np.radians(q_arr), delimiter=',')
    print("Data saved!")

    # Z calibration check: where (rig z) was the pen when force was 1 N?
    at_force = np.abs(fz_arr - FZ_DES) < 0.2
    if np.any(at_force):
        z_m = float(np.median(EE_arr[at_force, 2]))
        print(f"\nRig z at {abs(FZ_DES)} N contact: {z_m:.3f} mm (should be near 0)")
        print(f"Suggested Z_SURFACE_TRIM = {Z_SURFACE_TRIM + z_m:.3f}  (current {Z_SURFACE_TRIM})")
    else:
        print("\nNo samples at the target force, so no Z_SURFACE_TRIM suggestion.")

    plt.figure()
    plt.plot(t_arr, fz_arr, label="measured Fz")
    plt.plot(t_arr, fzdes_log, '--', label="desired Fz")
    plt.xlabel("Time (s)")
    plt.ylabel("Force Z (N)")
    plt.title("Force vs Time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t_arr, EE_arr[:, 2], label="tip z (rig frame)")
    plt.plot(t_arr, planz_log, '--', label="planned z")
    plt.xlabel("Time (s)")
    plt.ylabel("Z (mm)")
    plt.title("Pen z vs Time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    sc = plt.scatter(EE_arr[:, 0], EE_arr[:, 1], c=fz_arr, cmap="viridis", s=5)
    plt.colorbar(sc, label="Force Z (N)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Executed Path (colored by force)")
    plt.axis("equal")
    plt.grid(True)

    plt.show()

print("End of Program")
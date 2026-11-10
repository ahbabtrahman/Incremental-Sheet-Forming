print("Start of Program")

import sys
sys.path.append("/documents/github/Sheet-Metal-Deformation-Research/SM MV/")
sys.path.append("toolbox")
sys.path.append("robot_motion")

import time
import traceback
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from robot_def import *                  # NOT pip
from utils import *                      # NOT pip

# User settings
ROBOTSTUDIO_MODE = True      # True = RobotStudio test, False = real robot

ROBOTSTUDIO_URL = "http://127.0.0.1:80"
REAL_ROBOT_IP = '192.168.60.101'
ATI_SERVICE_URL = 'rr+tcp://localhost:59823?service=ati_sensor'

# Geometry (waypoints from Excel)
filename   = "bpp_triple_serpentine.xlsx"
sheet_name = "line1"
offset_x   = 10.0    # shift X so geometry is centered within rig
offset_y   = 10.0    # shift Y so geometry is centered within rig
z_start    = 1.0     # just above the sheet, where we tare
lower_speed = 2.0    # mm/s, speed of the move down from hover to z_start

# FORCE (one constant force for the whole path)
force = -1.5

# Settings
vd = 2.0        # drawing speed (mm/s)
Kf = 0.05       # force -> z gain
Z_MIN = -3.0       # rig-frame z floor (mm)
CONTACT_TOL = 0.1   # tolerance during first contact
CONTACT_TIMEOUT = 15.0  # abort if target force is not reached
TIMESTEP = 0.004      # 4 ms EGM interval

# Virtual sheet (RobotStudio only)
SIM_SURFACE_Z = -0.30    # mm, rig-frame z where the pen first touches (0 = planned surface)
SIM_STIFFNESS = 3.0      # N per mm of penetration
SIM_NOISE = 0.002         # N, gaussian noise on the simulated sensor
FZ_ABORT = 10.0          # N: abort if |Fz| exceeds this (simulation)

controller_params = {
    "force_ctrl_damping": 20.0,
    "force_epsilon": 0.1,         # below this = not touching
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1,          # N/s force ramp after first contact
    "load_speed": 5.0,            # mm/s descent speed until contact
    "unload_speed": 1.0,
    'settling_time': 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 10,
    "jogging_acc": 10,
    'force_filter_alpha': 0.9
}


def Htransform(R, p):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p
    return H

def H_from_RT(R, T):
    return Htransform(R, T)


# Rig kinematics
try:
    rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
except OSError:
    if not ROBOTSTUDIO_MODE:
        raise
    print("WARNING: rig_pose.csv not found - using a default rig pose for RobotStudio testing")
    rig_pose = Htransform(np.eye(3), np.array([400.0, -50.0, 300.0]))

z_theta = 2.0549 * np.pi / 180
Rz = rot(rig_pose[0:3, 2], z_theta)
Pbr = rig_pose[0:3, -1]
Rbr = rig_pose[0:3, 0:3] @ Rz
ipad_pose_T = Transform(Rbr, Pbr)     # rig frame -> robot base frame

# Robot model
Pft = np.array([-55.45, 0, 131.2])   # flange -> pen tip
tool_T = Transform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", Htransform(np.eye(3), Pft), delimiter=',')
robot_tb = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml",
                     tool_file_path="rig_pen.csv")
robot_tb.R_tool = tool_T.R
robot_tb.p_tool = tool_T.p

corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
corner_R_base = Rbr @ corner_R


# Backends
class RealBackend:
    def __init__(self):
        import RobotRaconteur as RR
        from RobotRaconteur.Client import RRN
        from RobotMotionController import MotionController
        self.RR = RR
        H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',')
        H_ati2pentip = np.linalg.inv(H_pentip2ati)
        self.ad_T = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1])).T   # sensor wrench -> pen-tip wrench
        RR_ati_cli = RRN.ConnectService(ATI_SERVICE_URL)
        self.mctrl = MotionController(robot_tb, H_from_RT(Rbr, Pbr), H_pentip2ati, controller_params, TIMESTEP,
                                      FORCE_PROTECTION=10, RR_ati_cli=RR_ati_cli,
                                      abb_robot_ip=REAL_ROBOT_IP)

    def start(self):
        self.mctrl.start_egm()

    def stop(self):
        self.mctrl.stop_egm()

    def read_position(self):
        state = None
        for _ in range(20):
            res, state = self.mctrl.egm.receive_from_robot(timeout=0.1)
            if not res:
                self.mctrl.stop_egm()
                self.mctrl.start_egm()
                print("Communication Lost, Reconnecting")
            else:
                break
        return np.radians(state.joint_angles)

    def position_cmd(self, q):
        self.mctrl.egm.send_to_robot(np.degrees(q))

    def jog(self, q, v):
        self.mctrl.jog_joint_position_cmd(np.array(q), v=v)

    def tare(self):
        for _ in range(2):
            time.sleep(0.1)
            self.mctrl.RR_ati_cli.setf_param("set_tare", self.RR.VarValue(True, "bool"))
        time.sleep(1.0)   # let the force filter settle
        print("Tared. Force reading:", self.ad_T @ self.mctrl.ft_reading)

    def read_fz(self):
        ft_tip = self.ad_T @ self.mctrl.ft_reading
        if np.linalg.norm(ft_tip[3:]) > self.mctrl.FORCE_PROTECTION:
            raise RuntimeError(f"force too large: {ft_tip[3:]}")
        if time.time() - self.mctrl.last_ft_time > 0.1:
            raise RuntimeError("force reading lost")
        return float(ft_tip[-1])

    def force_ctrl(self, f_err):
        return self.mctrl.force_impedence_ctrl(f_err)


class SimBackend:
    def __init__(self):
        import abb_motion_program_exec as abb
        from abb_robot_client.egm import EGM
        self.abb = abb
        self.EGM = EGM
        mm = abb.egm_minmax(-1e-3, 1e-3)
        self.egm_config = abb.EGMJointTargetConfig(mm, mm, mm, mm, mm, mm, 1000, 1000)
        self.client = abb.MotionProgramExecClient(base_url=ROBOTSTUDIO_URL)
        self.egm = None
        self.q_last = None
        self.rng = np.random.default_rng(0)

    def _launch(self):
        mp = self.abb.MotionProgram(egm_config=self.egm_config)
        mp.EGMRunJoint(10, 0.05, 0.05)
        self.client.execute_motion_program(mp, wait=False)

    def start(self):
        self._launch()
        self.egm = self.EGM()
        print("EGM is running (RobotStudio)")

    def stop(self):
        self.client.stop_egm()

    def read_position(self):
        for _ in range(20):
            res, state = self.egm.receive_from_robot(timeout=0.1)
            if res:
                self.q_last = np.radians(state.joint_angles)
                return self.q_last
            print("Communication Lost, Reconnecting...")
            self.client.stop_egm()
            self.client.stop_motion_program()
            self._launch()
        raise RuntimeError("EGM communication lost")

    def position_cmd(self, q):
        self.egm.send_to_robot(np.degrees(q))

    def jog(self, q_target, v):
        q0 = self.read_position()
        dq = np.asarray(q_target) - q0
        dist = np.linalg.norm(robot_tb.fwd(q_target).p - robot_tb.fwd(q0).p)
        T = max(dist / v, np.max(np.abs(dq)) / np.radians(30), 1.0)
        n = int(T / TIMESTEP)
        for k in range(1, n + 1):
            alpha = 0.5 - 0.5 * np.cos(np.pi * k / n)
            self.read_position()
            self.position_cmd(q0 + alpha * dq)

    def tare(self):
        print("Simulation: no sensor to tare")

    def read_fz(self):
        z_tip = (Rbr.T @ (robot_tb.fwd(self.q_last).p - Pbr))[2]
        penetration = max(0.0, SIM_SURFACE_Z - z_tip)
        fz = -SIM_STIFFNESS * penetration + self.rng.normal(0, SIM_NOISE)
        if abs(fz) > FZ_ABORT:
            raise RuntimeError(f"force too large: {fz:.2f} N")
        return float(fz)

    def force_ctrl(self, f_err):
        return controller_params["force_ctrl_damping"] * f_err


def calc_lam_js(curve_js, robot):
    # curve_js is the list of joint angles (rad), Nx6
    curve_p = []   # flange/tool position
    for i in range(len(curve_js)):
        curve_p.append(robot.fwd(curve_js[i]).p)
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))   # executed path length
    lam = np.insert(lam, 0, 0)
    return lam


# Grab waypoints from Excel, shift X,Y, save CSVs
data = (pd.read_excel(filename, sheet_name=sheet_name, header=None)).to_numpy()
waypoints = np.array(data[:, 0:3], dtype=float)
waypoints[:, 0] += offset_x
waypoints[:, 1] += offset_y
np.savetxt("gcode_coordinates.csv", waypoints, delimiter=',')

# Sheet layout: first row = hover point (z = 10), last row = hover point (z = 10),
# every row in between is on the sheet (z = 0)
hover_start = waypoints[0]      # x, y, z of the approach point
hover_end   = waypoints[-1]     # x, y, z of the retract point
draw_pts    = waypoints[1:-1]   # points that are actually drawn
if np.any(np.abs(draw_pts[:, 2]) > 1e-6):
    print("WARNING: some drawn rows have z != 0; z is ignored (force control sets z)")
loop_xy = draw_pts[:, :2]
num_sides = len(loop_xy) - 1

lam_planned = np.cumsum(np.linalg.norm(np.diff(loop_xy, axis=0), axis=1))   # path length
lam_planned = np.insert(lam_planned, 0, 0)
np.savetxt("lam_planned.csv", lam_planned, delimiter=',')
print(f"{num_sides} segments, path length {lam_planned[-1]:.2f} mm, "
      f"hover z start/end = {hover_start[2]} / {hover_end[2]} mm")

# Connect
backend = SimBackend() if ROBOTSTUDIO_MODE else RealBackend()
print(f"Mode: {'RobotStudio' if ROBOTSTUDIO_MODE else 'real'}")
print("Robot obj created")


# Helper functions
def jog_to_rig_point(p_rig, v=None):
    if v is None:
        v = controller_params["jogging_speed"]
    q_read = backend.read_position()
    q = robot_tb.inv(Rbr @ np.asarray(p_rig) + Pbr, corner_R_base, q_read)[0]
    backend.jog(q, v)


def step(q_now, xy_cmd, dz=0.0):
    tip = deepcopy(ipad_pose_T.inv() * robot_tb.fwd(q_now))
    tip.p[2] = max(tip.p[2] + dz, Z_MIN)
    tip.p[:2] = xy_cmd
    nxt = ipad_pose_T * tip
    backend.position_cmd(robot_tb.inv(nxt.p, nxt.R, q_now)[0])
    return tip.p


def make_contact(xy, fz_target):
    p = controller_params
    touch_t = None
    t0 = time.time()
    while True:
        if time.time() - t0 > CONTACT_TIMEOUT:
            raise RuntimeError("make_contact timed out")
        q = backend.read_position()
        fz = backend.read_fz()
        if abs(fz) < p['force_epsilon'] and touch_t is None:
            step(q, xy, dz=-p['load_speed'] * TIMESTEP)                  # descend
        else:
            if touch_t is None:
                touch_t = time.time()
                print("contact made")
            fz_ref = max(fz_target, -(time.time() - touch_t) * p['trapzoid_slope'])
            dz = backend.force_ctrl(fz_ref - fz) * TIMESTEP * Kf         # CONTROL FORCE
            step(q, xy, dz)
            if fz_ref == fz_target and abs(fz - fz_target) < CONTACT_TOL:
                print("target force reached")
                return


def draw_edge(xy0, xy1, fz_target, speed, record):
    L = np.linalg.norm(xy1 - xy0)
    if L == 0:          # repeated waypoint, nothing to draw
        return
    u = (xy1 - xy0) / L
    s = 0.0
    while True:
        q = backend.read_position()
        fz = backend.read_fz()                                              # READ FORCE
        dz = backend.force_ctrl(fz_target - fz) * TIMESTEP * Kf             # CONTROL FORCE
        p_rig = step(q, xy0 + u * min(s, L), dz)
        record.append(np.hstack(([time.time(), fz, fz_target], p_rig, np.degrees(q))))
        if s >= L:
            break
        s += speed * TIMESTEP


# Main
edge_records = []
print("Starting EGM...")
backend.start()

try:
    # 1) Hover at the first row of the sheet (x, y, z = hover)
    jog_to_rig_point(hover_start)
    input("At safe approach (hovering). Press Enter to lower...")

    # 2) Go just above the sheet
    jog_to_rig_point([*loop_xy[0], z_start], v=lower_speed)
    input("Just above sheet. Press Enter to tare and make contact...")

    # 3) Tare while NOT touching, then touch down
    backend.tare()

    # Touch down and ramp to the force
    make_contact(loop_xy[0], force)

    # Draw each segment at the same force
    for k in range(num_sides):
        print(f"Segment {k + 1}/{num_sides}: target {force} N")
        rec = []
        try:
            draw_edge(loop_xy[k], loop_xy[k + 1], force, vd, rec)
        finally:
            edge_records.append(np.array(rec))   # keep partial data if a segment aborts

    # Retract: lift straight up, then go to the last row (hover point)
    q_now = backend.read_position()
    tip = ipad_pose_T.inv() * robot_tb.fwd(q_now)
    jog_to_rig_point([tip.p[0], tip.p[1], hover_end[2]])
    jog_to_rig_point(hover_end)
    print("Drawing complete.")

except (Exception, KeyboardInterrupt) as e:
    print("Error:", e)
    traceback.print_exc()
    try:   # lift the pen off the sheet before shutting EGM down
        q_now = backend.read_position()
        tip = ipad_pose_T.inv() * robot_tb.fwd(q_now)
        jog_to_rig_point([tip.p[0], tip.p[1], hover_start[2]])
    except Exception as e2:
        print("Could not lift pen:", e2)

finally:
    try:
        backend.stop()
    except Exception as e:
        print("stop_egm warning:", e)

# Save + plot (after EGM is stopped)
edge_records = [r for r in edge_records if len(r) > 0]
if len(edge_records) == 0:
    print("No data Recorded")
else:
    record = np.vstack(edge_records)
    # columns: time, fz, fz_target, x, y, z, 6 joint angles (deg)
    np.savetxt("gcode_forcectrl_record.csv", record, delimiter=',')
    t_arr = record[:, 0] - record[0, 0]
    fz_arr = record[:, 1]
    fzdes_log = record[:, 2]
    EE_arr = record[:, 3:6]                  # measured tip position (rig frame)

    print("All done! Saving data...")

    runtime = t_arr
    ft_record = np.column_stack((fz_arr, fzdes_log))   # measured Fz, desired Fz
    qdata = record[:, 6:12]                            # actual joint angles (deg)
    curve_js_exe = np.radians(qdata)                   # executed joint angles (rad)
    lam_exe = calc_lam_js(curve_js_exe, robot_tb)      # executed path length

    # save force data
    forcedata = np.column_stack((runtime, ft_record))
    np.savetxt("FTData.csv", forcedata, delimiter=',')

    # save actual joint data
    jointdata = np.column_stack((runtime, qdata))
    np.savetxt("JointData.csv", jointdata, delimiter=',')

    # save curve_js_exe
    np.savetxt("curvejsexedata.csv", curve_js_exe, delimiter=',')

    print("Data saved!")
    planz_log = np.zeros(len(record))        # planned z = sheet surface (rig z = 0)

    plt.figure()
    plt.plot(t_arr, fz_arr, label="simulated Fz" if ROBOTSTUDIO_MODE else "measured Fz")
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
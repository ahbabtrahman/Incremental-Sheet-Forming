# Generates a path from an Excel sheet, then draws it with z force control 
print("Start of Program")
 
import sys
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
sys.path.append('toolbox')
sys.path.append('robot_motion')
from robot_def import *
from utils import *
 
# USER SETTINGS 
MODE = "real"          # "robotstudio" or "real"
USE_FORCE_CONTROL = True      # False = just play the planned path (no z correction)
 
ROBOTSTUDIO_URL = "http://127.0.0.1:80"
REAL_ROBOT_IP = '192.168.60.101'
ATI_SERVICE_URL = 'rr+tcp://localhost:59823?service=ati_sensor'
 
filename = "bpp_triple_serpentine.xlsx"     # EDIT
sheet_name = "line1"                        # EDIT
drawing_origin = np.array([10.0, 10.0, 0.0])  # rig-frame mm: G-code (0,0,0) lands here. z=0 is the sheet surface
 
dlam_des = 0.07    # mm between planned points. Try not to set higher than 0.1, curves tend to disappear
tool_vel = 4       # mm/s
tool_acc = 2       # mm/s^2
 
# Tool offset (flange -> pen tip). Should match what was used in calibration
Pft = np.array([-55.755, 0, 130.05])
 
# Force control
FZ_DES = -1.0        # N desired pen-tip Z force. Negative = pressing down
KF = 0.1             # force-control gain on top of force_ctrl_damping
Z_ENGAGE = 1.0       # mm: force control starts once planned z is at/below this height
Z_CORR_MIN = -2.0    # mm: max the pen may be pushed below the planned z (safety clamp)
FORCE_PROTECTION = 5 # real robot: torque magnitude abort limit
FZ_ABORT = 10.0      # N: abort if |Fz| exceeds this (used in simulation)
 
# Virtual sheet (RobotStudio only)
SIM_SURFACE_Z = -0.30    # mm, rig-frame z where the pen first touches (0 = planned surface). Nonzero tests the contact search
SIM_STIFFNESS = 3.0      # N per mm of penetration
SIM_NOISE = 0.01         # N, gaussian noise on the simulated sensor
 
TIMESTEP = 0.004   # 4 ms EGM interval
 
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
 
assert MODE in ("robotstudio", "real"), "MODE must be 'robotstudio' or 'real'"
 
 
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
    if MODE == "real":
        raise
    print("WARNING: rig_pose.csv not found - using a default rig pose for RobotStudio testing")
    rig_pose = Htransform(np.eye(3), np.array([400.0, -50.0, 300.0]))
 
# z correction (rotation about the rig z axis), same as the calibration/run scripts
z_theta = 2.0549 * np.pi / 180
Rz = rot(rig_pose[0:3, 2], z_theta)
Pbr = rig_pose[0:3, -1]
Rbr = rig_pose[0:3, 0:3] @ Rz
 
# Robot model 
tool_T = Transform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", Htransform(np.eye(3), Pft), delimiter=',')
robot = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml", tool_file_path="rig_pen.csv")
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p
 
 
# Path planning 
def plan_path(waypoints, dlam):
    corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T   # pen pointing down (pi about y)
    R_cmd = Rbr @ corner_R
 
    pts = []
    for i in range(len(waypoints) - 1):
        n = int(np.linalg.norm(waypoints[i + 1] - waypoints[i]) / dlam) + 1
        seg = np.linspace(waypoints[i], waypoints[i + 1], n)
        pts.extend(seg[:-1])
    pts.append(waypoints[-1])
    p_rig = np.array(pts)
    p_base = (Rbr @ p_rig.T).T + Pbr
 
    def solve(p, seed):
        sols = robot.inv(p, R_cmd, seed)
        if len(sols) == 0:
            raise RuntimeError(f"No IK solution at {p} (is the point reachable?)")
        return sols[0]
 
    curve_js = [solve(p_base[0], np.zeros(6))]
    for i in range(1, len(p_base)):
        if i % 20000 == 0:
            print(f"IK {i} out of {len(p_base)}")
        q = solve(p_base[i], curve_js[-1])
        assert np.linalg.norm(p_base[i] - robot.fwd(q).p) < 0.001, "The inverse kinematics gave a large error"
        assert np.max(np.abs(q - curve_js[-1])) < 0.1, f"Joint jump at point {i} (possible configuration flip)"
        curve_js.append(q)
 
    lam = np.insert(np.cumsum(np.linalg.norm(np.diff(p_rig, axis=0), axis=1)), 0, 0)
    return p_rig, np.array(curve_js), lam, R_cmd
 
 
def trapezoid_times(L, v, a, dt):
    if a <= 0:
        T = L / v
        t = np.append(np.arange(0, T, dt), T)
        return t, np.minimum(v * t, L)
    d_a = v ** 2 / (2 * a)
    if L >= 2 * d_a:
        t_a, t_c, v_pk = v / a, (L - 2 * d_a) / v, v
    else:                                    # too short to reach full speed: triangular profile
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
        self.mctrl = MotionController(robot, rig_pose, H_pentip2ati, controller_params, TIMESTEP,
                                      FORCE_PROTECTION=FORCE_PROTECTION, RR_ati_cli=RR_ati_cli,
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
        dist = np.linalg.norm(robot.fwd(q_target).p - robot.fwd(q0).p)
        T = max(dist / v, np.max(np.abs(dq)) / np.radians(30), 1.0)
        n = int(T / TIMESTEP)
        for k in range(1, n + 1):
            alpha = 0.5 - 0.5 * np.cos(np.pi * k / n)
            self.read_position()
            self.position_cmd(q0 + alpha * dq)
 
    def tare(self):
        print("Simulation: no sensor to tare")
 
    def read_fz(self):
        z_tip = (Rbr.T @ (robot.fwd(self.q_last).p - Pbr))[2]
        penetration = max(0.0, SIM_SURFACE_Z - z_tip)
        fz = -SIM_STIFFNESS * penetration + self.rng.normal(0, SIM_NOISE)
        if abs(fz) > FZ_ABORT:
            raise RuntimeError(f"force too large: {fz:.2f} N")
        return float(fz)
 
 
# Build the path
data = pd.read_excel(filename, sheet_name=sheet_name, header=None).to_numpy()
waypoints = data[:, 0:3].astype(float) + drawing_origin
np.savetxt("gcode_coordinates_new.csv", waypoints, delimiter=',')
 
st = time.perf_counter()
p_rig_plan, curve_js, lam_planned, R_cmd = plan_path(waypoints, dlam_des)
print(f"Planned {len(curve_js)} points, path length {lam_planned[-1]:.1f} mm, IK {time.perf_counter() - st:.1f} s")
 
fullruntraj_q, traj_p_rig, fullruntime_bp = trajectory_generate(curve_js, p_rig_plan, lam_planned, tool_vel, tool_acc)
print(f"Trajectory ready: {len(fullruntraj_q)} steps ({len(fullruntraj_q) * TIMESTEP:.1f} s)")
 
np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=',')
np.savetxt("fullruntime_bp.csv", fullruntime_bp, delimiter=',')
np.savetxt("curve_js.csv", curve_js, delimiter=',')
np.savetxt("lam_planned.csv", lam_planned, delimiter=',')
np.savetxt("pos_check.csv", traj_p_rig, delimiter=',')
 
# Execute
backend = RealBackend() if MODE == "real" else SimBackend()
print(f"Mode: {MODE}, force control: {USE_FORCE_CONTROL}")
 
t_log, fz_log, fzdes_log, zcorr_log, EE_pos, q_log = [], [], [], [], [], []
aborted = False
 
print("Starting EGM...")
backend.start()
 
try:
    # Go to the first point of the path (already at hover height from the G-code)
    backend.jog(curve_js[0], controller_params["jogging_speed"])
    if MODE == "real":
        input("At start (hovering). Press Enter to tare and draw...")
    backend.tare()
 
    z_corr = 0.0       # mm offset added to the planned rig-frame z
    touch_t = None     # time of first contact
    load_speed = controller_params['load_speed']
    t0 = time.time()
 
    for i in range(len(fullruntraj_q)):
        read_q = backend.read_position()
        fz_now = backend.read_fz()
        p_rig = traj_p_rig[i]
        this_fz_des = 0.0
 
        if USE_FORCE_CONTROL:
            if p_rig[2] <= Z_ENGAGE:
                # Force control on rig z; x/y follow the plan
                if touch_t is None and abs(fz_now) < controller_params['force_epsilon']:
                    z_corr -= load_speed * TIMESTEP            # no contact yet: keep descending
                else:
                    if touch_t is None:
                        touch_t = time.time()
                        print("contact made")
                    # trapezoidal force ramp up to FZ_DES (FZ_DES is negative)
                    this_fz_des = max(FZ_DES, -(time.time() - touch_t) * controller_params["trapzoid_slope"])
                    f_err = this_fz_des - fz_now
                    v_des = controller_params["force_ctrl_damping"] * f_err
                    z_corr += v_des * TIMESTEP * KF
                z_corr = max(z_corr, Z_CORR_MIN)
            else:
                # Hovering / retracting: release contact and relax the correction
                touch_t = None
                z_corr = min(0.0, z_corr + load_speed * TIMESTEP)
 
            p_cmd_rig = p_rig.copy()
            p_cmd_rig[2] += z_corr
            sols = robot.inv(Rbr @ p_cmd_rig + Pbr, R_cmd, read_q)
            if len(sols) == 0:
                raise RuntimeError("No IK solution for corrected point")
            q_cmd = sols[0]
        else:
            q_cmd = fullruntraj_q[i]
 
        # Log the measured tip position (rig frame)
        t_log.append(time.time() - t0)
        fz_log.append(fz_now)
        fzdes_log.append(this_fz_des)
        zcorr_log.append(z_corr)
        EE_pos.append(Rbr.T @ (robot.fwd(read_q).p - Pbr))
        q_log.append(np.degrees(read_q))
 
        backend.position_cmd(q_cmd)
 
    print("Drawing complete.")
 
except (Exception, KeyboardInterrupt) as e:
    aborted = True
    print("Error:", e)
    traceback.print_exc()
 
finally:
    if aborted:
        try:   # lift straight up off the sheet
            q_now = backend.read_position()
            tip = robot.fwd(q_now)
            q_lift = robot.inv(tip.p + 10 * Rbr[:, 2], tip.R, q_now)[0]
            backend.jog(q_lift, controller_params["jogging_speed"])
            print("Aborted - pen lifted.")
        except Exception as e2:
            print("Could not lift pen:", e2)
    try:
        backend.stop()
    except Exception as e:
        print("stop_egm warning:", e)
 
# Save + plot
if len(fz_log) == 0:
    print("No data Recorded")
else:
    t_log = np.array(t_log)
    fz_log = np.array(fz_log)
    fzdes_log = np.array(fzdes_log)
    zcorr_log = np.array(zcorr_log)
    EE_pos = np.array(EE_pos)
    q_log = np.array(q_log)
 
    np.savetxt("FTData.csv", np.column_stack([t_log, fz_log, fzdes_log, zcorr_log]), delimiter=',')
    np.savetxt("JointData.csv", np.column_stack([t_log, q_log]), delimiter=',')
    np.savetxt("path_force.csv", np.column_stack([t_log, fz_log, EE_pos]), delimiter=',')
    np.savetxt("curvejsexedata.csv", np.radians(q_log), delimiter=',')
    print("Data saved!")
 
    plt.figure()
    plt.plot(t_log, fz_log, label="measured Fz" if MODE == "real" else "simulated Fz")
    plt.plot(t_log, fzdes_log, '--', label="desired Fz")
    plt.xlabel("Time (s)")
    plt.ylabel("Force Z (N)")
    plt.title("Force vs Time")
    plt.legend()
    plt.grid(True)
 
    plt.figure()
    plt.plot(t_log, EE_pos[:, 2], label="tip z (rig frame)")
    plt.plot(t_log, traj_p_rig[:len(t_log), 2], '--', label="planned z")
    if MODE == "robotstudio":
        plt.axhline(SIM_SURFACE_Z, color='k', linestyle=':', label="virtual sheet")
    plt.xlabel("Time (s)")
    plt.ylabel("Z (mm)")
    plt.title("Pen z vs Time")
    plt.legend()
    plt.grid(True)
 
    plt.figure()
    plt.plot(t_log, zcorr_log)
    plt.xlabel("Time (s)")
    plt.ylabel("z correction (mm)")
    plt.title("Force-control z correction")
    plt.grid(True)
 
    plt.figure()
    sc = plt.scatter(EE_pos[:, 0], EE_pos[:, 1], c=fz_log, cmap="viridis", s=5)
    plt.colorbar(sc, label="Force Z (N)")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.title("Executed Path (colored by force)")
    plt.axis("equal")
    plt.grid(True)
 
    plt.show()
 
print("End of Program")
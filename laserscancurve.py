import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from matplotlib import pyplot as plt

from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox

import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM


# Settings
MODE = "robotstudio"    # "robotstudio" or "real"
PATH_MODE = "arc"       # "arc" or "excel"
SCAN_MODE = 2           # 1 = fixed orientation; 2 = follows path tangent

ENABLE_ROBOT_MOTION = True
SHOW_PLAN_PLOT = True

ROBOTSTUDIO_URL = "http://127.0.0.1:80"
REAL_ROBOT_URL = "http://192.168.60.101:80"

ENABLE_SCANNER = True

TIMESTEP = 0.004        
TOOL_VEL = 5.0          # Used on straights
MAX_JOINT_SPEED_DEG = 3.0   # deg/s used on curves
SPEED_SMOOTH_MM = 3.0       # speed changes gradually over about this much path
YAW_RATE_MAX_DEG_PER_MM = 20.0   # max tool yaw change per mm of path
YAW_AXIS_ONLY = True             # U-turns does NOT spin the tool 180 deg (keep try as to not twist wires)
JOG_SPEED = 5.0        # move from current pose to path start

DENSITY = 0.05          # Cartesian spacing for IK (mm)

# Arc path settings
ARC_RADIUS = 35.0
ARC_ORIGIN = np.array([10.0, 10.0, 0.0])
ARC_ANGLE_DEG = 180.0
ARC_HEIGHT = 20.0
ARC_NUM_WAYPOINTS = 60

# Excel path settings
EXCEL_FILENAME = "bpp_triple_serpentine.xlsx"
EXCEL_SHEET = "line1"
DRAWING_ORIGIN = np.array([10.0, 10.0, 0.0])
SCAN_HEIGHT = 20.0      # mm above the sheet (check what height laser needs, idk)

if MODE not in ("robotstudio", "real"):
    raise ValueError("MODE must be 'robotstudio' or 'real'.")

ABB_BASE_URL = ROBOTSTUDIO_URL if MODE == "robotstudio" else REAL_ROBOT_URL
MTI_SERVICE_URL = "rr+tcp://localhost:60830/?service=MTI2D"
MTI_EXPOSURE = "25"

PFT = np.array([-55.45, 0.0, 131.2])   # flange -> tool tip
RIG_POSE_FILENAME = "rig_pose.csv"

LASER_OFFSET_X = 0.0
LASER_OFFSET_Y = 0.0
LASER_OFFSET_Z = 0.0    # leave 0 unless the laser is also higher/lower than the tip
LASER_OFFSET = np.array([LASER_OFFSET_X, LASER_OFFSET_Y, LASER_OFFSET_Z])

# laser X offset is along the path and Y is sideways
# laser Y offset is along the path and X is sideways
LASER_FOLLOW_AXIS = "x"

RIG_Z_ADJUSTMENT_DEG = 2.0549
IK_JUMP_WARNING_DEG = 2.0 # dont want next to each other angles to be this far apart

if LASER_FOLLOW_AXIS not in ("x", "y"):
    raise ValueError("LASER_FOLLOW_AXIS must be 'x' or 'y'.")


def calc_lam_js(curve_js, robot):
    curve_p = np.array([fwdkin(robot, q).p for q in curve_js])
    seg = np.linalg.norm(np.diff(curve_p, axis=0), axis=1)
    return np.insert(np.cumsum(seg), 0, 0.0)

        
def trajectory_generate(curve_js, robot, lin_vel, timestep=0.004, max_joint_speed_deg=3.0, smooth_mm=3.0):
    q = np.asarray(curve_js, dtype=float)

    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("curve_js must contain at least two joint positions.")
    if lin_vel <= 0 or timestep <= 0 or max_joint_speed_deg <= 0:
        raise ValueError("lin_vel, timestep and max_joint_speed_deg must be positive.")

    lam = calc_lam_js(q, robot)

    keep = np.concatenate(([True], np.diff(lam) > 1e-9))
    lam = lam[keep]
    q = q[keep]

    if len(lam) < 2 or lam[-1] < 1e-9:
        raise ValueError("Trajectory has no measurable Cartesian path length.")

    dlam = np.diff(lam)
    t_nom = dlam / lin_vel                                    # s per segment at full speed
    t_joint = np.max(np.abs(np.diff(q, axis=0)), axis=1) / np.radians(max_joint_speed_deg)

    # Speed factor per segment: 1 = full lin_vel, <1 = slowed for joint speed.
    f = np.minimum(1.0, t_nom / np.maximum(t_joint, 1e-12))
    f = np.maximum(f, 1e-6)

    # Min filter then moving average: smooth, and never above the limit.
    w = max(1, int(round(smooth_mm / np.mean(dlam))))
    f = minimum_filter1d(f, size=2 * w + 1, mode="nearest")
    f = uniform_filter1d(f, size=2 * w + 1, mode="nearest")

    t_cum = np.insert(np.cumsum(t_nom / f), 0, 0.0)
    total_time = float(t_cum[-1])

    num_intervals = max(1, int(np.ceil(total_time / timestep)))
    time_array = np.arange(num_intervals + 1, dtype=float) * timestep
    t_query = np.minimum(time_array, total_time)

    traj_q = np.column_stack([np.interp(t_query, t_cum, q[:, k]) for k in range(q.shape[1])])
    traj_q[0] = q[0]
    traj_q[-1] = q[-1]

    return np.asarray(traj_q), time_array


def generate_arc_waypoints(radius, origin, angle_deg, height, num_pts):
    half_angle = np.radians(angle_deg) / 2.0
    angles = np.linspace(-half_angle, half_angle, num_pts)
    return np.array([
        [origin[0] + radius * np.cos(t),
         origin[1] + radius * np.sin(t),
         origin[2] + height]
        for t in angles
    ], dtype=float)


def load_waypoints_from_excel(filename, sheet_name, drawing_origin, scan_height):
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    waypoints = df.iloc[:, 0:3].to_numpy(dtype=float) + np.asarray(drawing_origin, dtype=float)
    waypoints[:, 2] = drawing_origin[2] + scan_height
    np.savetxt("gcode_coordinates_new.csv", waypoints, delimiter=",")
    return waypoints


def smooth_cartesian_path(waypoints, spacing):
    waypoints = np.asarray(waypoints, dtype=float)

    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("Waypoints must have shape (N, 3).")
    if len(waypoints) < 2:
        raise ValueError("At least two Cartesian waypoints are required.")

    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    wp_lam = np.insert(np.cumsum(seg), 0, 0.0)

    keep = np.concatenate(([True], np.diff(wp_lam) > 1e-9))
    wp_lam = wp_lam[keep]
    waypoints = waypoints[keep]

    if len(wp_lam) < 2:
        raise ValueError("All input waypoints are coincident.")

    path_spline = CubicSpline(wp_lam, waypoints, axis=0, bc_type="natural")

    lam_samples = np.append(np.arange(0.0, wp_lam[-1], spacing), wp_lam[-1])
    curve_p = path_spline(lam_samples)
    return curve_p, lam_samples, path_spline


def linear_cartesian_path(waypoints, spacing):
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError("Waypoints must have shape (N>=2, 3).")

    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = int(np.linalg.norm(b - a) / spacing) + 1
        pts.extend(np.linspace(a, b, n)[:-1])
    pts.append(waypoints[-1])
    curve_p = np.asarray(pts)

    seg = np.linalg.norm(np.diff(curve_p, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    curve_p = curve_p[keep]
    lam = np.insert(np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1)), 0, 0.0)
    return curve_p, lam, None


def calculate_path_orientations(curve_p, lam_samples, path_spline, scan_mode):
    n = len(curve_p)

    if scan_mode == 1:
        yaw = np.full(n, np.pi)
    elif scan_mode == 2:
        tangent = np.gradient(curve_p, lam_samples, axis=0)
        theta = np.arctan2(tangent[:, 1], tangent[:, 0])
        if LASER_FOLLOW_AXIS == "x":
            pass                          # tool X along the path tangent (original behavior)
        elif LASER_FOLLOW_AXIS == "y":
            theta = theta + np.pi / 2.0   # tool Y along the path tangent
        else:
            raise ValueError("LASER_FOLLOW_AXIS must be 'x' or 'y'.")

        yaw = np.empty(n)
        yaw[0] = theta[0]
        max_rate = np.radians(YAW_RATE_MAX_DEG_PER_MM)

        for i in range(1, n):
            target = theta[i]
            if YAW_AXIS_ONLY:
                # Find the equivalent angle (target + k*pi) closest to yaw[i-1]
                # by wrapping the angular difference into [-pi/2, pi/2]
                diff = target - yaw[i - 1]
                # Wrap to [-pi, pi]
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                if diff > np.pi / 2.0:
                    diff -= np.pi
                elif diff < -np.pi / 2.0:
                    diff += np.pi
                target = yaw[i - 1] + diff

            max_step = max_rate * (lam_samples[i] - lam_samples[i - 1])
            # Smoothly rate-limit the step towards the target
            step = np.clip(target - yaw[i - 1], -max_step, max_step)
            yaw[i] = yaw[i - 1] + step
    else:
        raise ValueError("SCAN_MODE must be 1 or 2.")

    print(f"Tool yaw range: {np.degrees(yaw.min()):.1f} to {np.degrees(yaw.max()):.1f} deg")
    curve_R = []
    z_tool = np.array([0.0, 0.0, -1.0])
    for a in yaw:
        x_tool = np.array([np.cos(a), np.sin(a), 0.0])
        y_tool = np.array([np.sin(a), -np.cos(a), 0.0])
        curve_R.append(np.column_stack((x_tool, y_tool, z_tool)))

    return np.asarray(curve_R)


def make_joint_solution_continuous(q_new, q_previous):
    q_new = np.asarray(q_new, dtype=float).copy()
    q_previous = np.asarray(q_previous, dtype=float)
    return q_new + 2.0 * np.pi * np.round((q_previous - q_new) / (2.0 * np.pi))


def report_ik_jumps(curve_js, warning_deg=2.0):
    changes_deg = np.abs(np.degrees(np.diff(curve_js, axis=0)))
    if len(changes_deg) == 0:
        return

    print("\nMaximum adjacent IK joint changes:")
    for j, v in enumerate(np.max(changes_deg, axis=0)):
        print(f"   Joint {j + 1}: {v:.4f} deg")

    rows, cols = np.where(changes_deg > warning_deg)
    if len(rows):
        print(f"\nWARNING: {len(rows)} adjacent joint changes exceed {warning_deg:.1f} deg.")
        for r, c in zip(rows[:20], cols[:20]):
            print(f"   Between samples {r} and {r + 1}, joint {c + 1}: {changes_deg[r, c]:.3f} deg")
        if len(rows) > 20:
            print("   Additional warnings omitted.")
    else:
        print(f"No adjacent joint changes exceed {warning_deg:.1f} deg.")


with open("ABB_1200_5_90_robot_default_config.yml", "r") as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

# TCP = laser point: tool tip position plus the laser offset (tool frame)
tool_T = Transform(np.eye(3), PFT + LASER_OFFSET)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p
print(f"TCP (laser) in flange frame: {tool_T.p}")
print(f"Tool axis following the path: {LASER_FOLLOW_AXIS.upper()}")


class EGMLink:
    def __init__(self, cond_time=10.0):
        self.cond_time = float(cond_time)
        mm = abb.egm_minmax(-1e-3, 1e-3)
        self.cfg = abb.EGMJointTargetConfig(mm, mm, mm, mm, mm, mm, 1000, 1000)
        self.client = abb.MotionProgramExecClient(base_url=ABB_BASE_URL)
        self.egm = None

    def _launch(self):
        mp = abb.MotionProgram(egm_config=self.cfg)
        mp.EGMRunJoint(self.cond_time, 0.05, 0.05)
        self.client.execute_motion_program(mp, wait=False)

    def start(self):
        self._launch()
        self.egm = EGM()
        print("EGM is running")

    def stop(self):
        try:
            self.client.stop_egm()
        except Exception as e:
            print("stop_egm warning:", e)

    def read(self, reconnect=True):
        for _ in range(20):
            res, state = self.egm.receive_from_robot(timeout=0.1)
            if res:
                return np.radians(state.joint_angles)
            if reconnect:
                print("Communication Lost, Reconnecting...")
                self.client.stop_egm()
                self.client.stop_motion_program()
                self._launch()
        raise TimeoutError("EGM communication lost")

    def cmd(self, q):
        self.egm.send_to_robot(np.degrees(np.asarray(q, dtype=float)))

    def jog(self, q_target, v=50.0):
        q0 = self.read()
        dq = np.asarray(q_target) - q0
        dist = np.linalg.norm(fwdkin(robot, q_target).p - fwdkin(robot, q0).p)
        T = max(dist / v, np.max(np.abs(dq)) / np.radians(30), 1.0)
        n = int(T / TIMESTEP)
        for k in range(1, n + 1):
            alpha = k / n
            self.read(reconnect=False)
            self.cmd(q0 + alpha * dq)


if PATH_MODE == "arc":
    corner_p_wp = generate_arc_waypoints(
        ARC_RADIUS, ARC_ORIGIN, ARC_ANGLE_DEG, ARC_HEIGHT, ARC_NUM_WAYPOINTS
    )
elif PATH_MODE == "excel":
    corner_p_wp = load_waypoints_from_excel(EXCEL_FILENAME, EXCEL_SHEET, DRAWING_ORIGIN, SCAN_HEIGHT)
else:
    raise ValueError("Invalid PATH_MODE. Use 'arc' or 'excel'.")

if PATH_MODE == "excel":
    curve_p, lam_samples, path_spline = linear_cartesian_path(corner_p_wp, DENSITY)
else:
    curve_p, lam_samples, path_spline = smooth_cartesian_path(corner_p_wp, DENSITY)
print(f"Generated smooth Cartesian path: {len(curve_p)} samples, length {lam_samples[-1]:.2f} mm")
z_spread = float(curve_p[:, 2].max() - curve_p[:, 2].min())
print(f"Scan height (rig z): {curve_p[0, 2]:.2f} mm, variation {z_spread:.4f} mm")
if z_spread > 1e-3:
    raise RuntimeError("Scan height is not constant along the path.")


curve_R = calculate_path_orientations(curve_p, lam_samples, path_spline, SCAN_MODE)


try:
    final_rig_pose = np.loadtxt(RIG_POSE_FILENAME, delimiter=",")
except OSError:
    if MODE == "real":
        raise
    print("WARNING: rig_pose.csv not found - using default rig pose for RobotStudio")
    final_rig_pose = np.eye(4)
    final_rig_pose[:3, 3] = [400.0, -50.0, 300.0]

z_theta = np.radians(RIG_Z_ADJUSTMENT_DEG)
Rz = rot(final_rig_pose[0:3, 2], z_theta)

Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz

curve_p_base = np.asarray([Rbr @ p + Pbr for p in curve_p])
curve_R_base = np.asarray([Rbr @ R for R in curve_R])


# Inverse kinematics 
print("\nCalculating inverse kinematics...")
ik_start_time = time.perf_counter()

joint_lo = np.asarray(robot.joint_lower_limit, dtype=float)
joint_hi = np.asarray(robot.joint_upper_limit, dtype=float)


def pick_continuous_solution(solutions, q_ref):
    best_q, best_cost = None, np.inf
    for sol in solutions:
        q = make_joint_solution_continuous(sol, q_ref)
        cost = np.max(np.abs(q - q_ref))
        if cost < best_cost:
            best_q, best_cost = q, cost
    return best_q


def shift_into_limits(curve_js):
    q = np.asarray(curve_js, dtype=float).copy()
    for jn in range(6):
        qmin, qmax = q[:, jn].min(), q[:, jn].max()
        best_shift, best_margin = 0.0, -np.inf
        for k in range(-4, 5):
            sft = 2.0 * np.pi * k
            margin = min(qmin + sft - joint_lo[jn], joint_hi[jn] - (qmax + sft))
            if margin > best_margin:
                best_shift, best_margin = sft, margin
        if best_margin < 0:
            raise RuntimeError(
                f"Joint {jn + 1} needs travel from {np.degrees(qmin):.0f} to {np.degrees(qmax):.0f} deg "
                f"({np.degrees(qmax - qmin):.0f} deg total) but its limits are "
                f"{np.degrees(joint_lo[jn]):.0f} to {np.degrees(joint_hi[jn]):.0f} deg. "
                "Reduce the tool yaw sweep (SCAN_MODE = 1) or change the path/start."
            )
        q[:, jn] += best_shift
    return q


q_init_solutions = robot6_sphericalwrist_invkin(
    robot, Transform(curve_R_base[0], curve_p_base[0]), np.zeros(6)
)
if len(q_init_solutions) == 0:
    raise RuntimeError("No IK solution found for the first path point (unreachable or outside joint limits).")

curve_js = [pick_continuous_solution(q_init_solutions, np.zeros(6))]

for i in range(1, len(curve_p_base)):
    solutions = robot6_sphericalwrist_invkin(
        robot, Transform(curve_R_base[i], curve_p_base[i]), curve_js[-1]
    )
    if len(solutions) == 0:
        raise RuntimeError(
            f"No IK solution at path sample {i} (path position {lam_samples[i]:.2f} mm): "
            "the target pose is unreachable or outside the joint limits."
        )
    curve_js.append(pick_continuous_solution(solutions, curve_js[-1]))

curve_js = shift_into_limits(np.asarray(curve_js))

# Verify Cartesian position with forward kinematics
max_pos_err = max(
    np.linalg.norm(curve_p_base[k] - fwdkin(robot, q).p) for k, q in enumerate(curve_js)
)
if max_pos_err > 0.05:
    raise RuntimeError(f"IK position error up to {max_pos_err:.4f} mm")

print("Planned joint ranges (deg):")
for jn in range(6):
    print(f"   Joint {jn + 1}: {np.degrees(curve_js[:, jn].min()):.1f} to "
          f"{np.degrees(curve_js[:, jn].max()):.1f}")

lam_planned = calc_lam_js(curve_js, robot)

print(
    f"Planned {len(curve_js)} IK points, FK path length {lam_planned[-1]:.2f} mm, "
    f"IK time {time.perf_counter() - ik_start_time:.2f} s"
)
report_ik_jumps(curve_js, IK_JUMP_WARNING_DEG)

max_jump_deg = float(np.max(np.abs(np.degrees(np.diff(curve_js, axis=0)))))
if ENABLE_ROBOT_MOTION and max_jump_deg > IK_JUMP_WARNING_DEG:
    raise RuntimeError(
        f"Refusing to move: largest adjacent joint change is {max_jump_deg:.2f} deg "
        f"(limit {IK_JUMP_WARNING_DEG:.1f}). The path/orientation is not safe to execute. "
        "Try SCAN_MODE = 1 or a smoother path."
    )

_lo = getattr(robot, "joint_lower_limit", None)
_hi = getattr(robot, "joint_upper_limit", None)
if ENABLE_ROBOT_MOTION and _lo is not None and _hi is not None:
    if np.any(curve_js < np.asarray(_lo)) or np.any(curve_js > np.asarray(_hi)):
        raise RuntimeError("Planned joint path exceeds the robot joint limits.")


fullruntraj_q, fullruntime = trajectory_generate(
    curve_js, robot, lin_vel=TOOL_VEL, timestep=TIMESTEP,
    max_joint_speed_deg=MAX_JOINT_SPEED_DEG, smooth_mm=SPEED_SMOOTH_MM
)

print(
    f"\nTrajectory generated: {len(fullruntraj_q)} commands, "
    f"duration {fullruntime[-1]:.2f} s, interval {TIMESTEP:.4f} s"
)
qd_max = np.degrees(np.max(np.abs(np.diff(fullruntraj_q, axis=0)), axis=0)) / TIMESTEP
print("Max planned joint speeds (deg/s): " +
      ", ".join(f"J{k + 1} {v:.2f}" for k, v in enumerate(qd_max)))

np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=",")
np.savetxt("fullruntime.csv", fullruntime, delimiter=",")
np.savetxt("curve_js.csv", curve_js, delimiter=",")
np.savetxt("lam_planned.csv", lam_planned, delimiter=",")
np.savetxt("curve_p_base.csv", curve_p_base, delimiter=",")

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(fullruntime, np.degrees(fullruntraj_q[:, i]), label=f"Joint {i + 1}")
plt.xlabel("Time (s)")
plt.ylabel("Joint angle (degrees)")
plt.title("Planned smooth joint trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("planned_joint_trajectory.png", dpi=120)
if SHOW_PLAN_PLOT:
    plt.show()
else:
    plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: 2D Drawn Path with Tool Orientation Vectors
ax1.plot(curve_p[:, 0], curve_p[:, 1], 'b-', label='Laser Path')
ax1.scatter(curve_p[0, 0], curve_p[0, 1], color='green', s=50, label='Start Point')
ax1.scatter(curve_p[-1, 0], curve_p[-1, 1], color='red', s=50, label='End Point')

# Plot orientation vectors along the path
step = max(1, len(curve_p) // 500)  # Sample x quiver arrows along the trajectory
x_coords = curve_p[::step, 0]
y_coords = curve_p[::step, 1]
axis_col = 0 if LASER_FOLLOW_AXIS == "x" else 1
u_vecs = curve_R[::step, 0, axis_col]  # followed tool axis (X component)
v_vecs = curve_R[::step, 1, axis_col]  # followed tool axis (Y component)

ax1.quiver(x_coords, y_coords, u_vecs, v_vecs, color='orange', scale=20,
           label=f'Tool {LASER_FOLLOW_AXIS.upper()}-axis (follows path)')
ax1.set_xlabel("X (mm)")
ax1.set_ylabel("Y (mm)")
ax1.set_title(f"Laser Path & Tool Orientation (SCAN_MODE = {SCAN_MODE})")
ax1.set_aspect('equal', 'box')
ax1.grid(True)
ax1.legend()

# Subplot 2: Joint 6 Orientation vs Path Length
j6_angles_deg = np.degrees(curve_js[:, 5])
ax2.plot(lam_planned, j6_angles_deg, 'g-', linewidth=2, label='Joint 6 Angle')
ax2.set_xlabel("Path Length (mm)")
ax2.set_ylabel("Joint 6 Angle (degrees)")
mode_str = "Stationary (Fixed Orientation)" if SCAN_MODE == 1 else "Normal to Path (Following Tangent)"
ax2.set_title(f"Joint 6 Behavior (Option {SCAN_MODE}: {mode_str})")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("drawn_path_and_j6_orientation.png", dpi=120)
plt.show()


if not ENABLE_ROBOT_MOTION:
    print(
        "\nRobot motion is DISABLED. Trajectory files saved. Review the plots, "
        "joint limits and workspace before setting ENABLE_ROBOT_MOTION = True."
    )

else:
    # Scanner (optional). Imports are lazy so RobotStudio-only runs don't need them.
    mti_client = None
    if ENABLE_SCANNER:
        from RobotRaconteur.Client import RRN
        print("Connecting to MTI scanner...")
        mti_client = RRN.ConnectService(MTI_SERVICE_URL)
        mti_client.setExposureTime(MTI_EXPOSURE)
    else:
        print("Scanner capture disabled.")

    print(f"\nConnecting to ABB EGM ({MODE})...")
    # Jog (>= 1 s, usually a few s) + scan duration + generous margin
    link = EGMLink(cond_time=fullruntime[-1] + 120.0)
    EE_pos, mti_recording, curve_js_exe = [], [], []

    try:
        link.start()

        print("Moving to path start...")
        link.jog(curve_js[0], v=JOG_SPEED)
        print("At path starting position. Beginning scan.")

        for i, q_command in enumerate(fullruntraj_q):
            read_q = link.read(reconnect=False)
            curve_js_exe.append(read_q)
            EE_pos.append(Rbr.T @ (fwdkin(robot, read_q).p - Pbr))

            if mti_client is not None:
                mti_recording.append(deepcopy(np.array([
                    mti_client.lineProfile.X_data,
                    mti_client.lineProfile.Z_data
                ])))

            link.cmd(q_command)

            if i % 500 == 0:
                print(f"Executed {i}/{len(fullruntraj_q)} commands")

        print("Path execution completed.")

    except (Exception, KeyboardInterrupt):
        try:
            link.client.stop_motion_program()
        except Exception:
            pass
        raise

    finally:
        link.stop()

    curve_js_exe = np.asarray(curve_js_exe)
    EE_pos = np.asarray(EE_pos)
    lam_exe = calc_lam_js(curve_js_exe, robot)

    np.savetxt("EE_pos.csv", EE_pos, delimiter=",")
    np.savetxt("curve_js_executed.csv", curve_js_exe, delimiter=",")

    if mti_client is not None:
        import open3d as o3d

        pcd_all = o3d.geometry.PointCloud()
        q_out_exe = curve_js_exe[:, :6]

        for i in range(min(len(mti_recording), len(q_out_exe))):
            scanner_T = fwdkin(robot, q_out_exe[i])   # laser frame (TCP includes LASER_OFFSET)
            profile = np.asarray(mti_recording[i])

            if profile.ndim != 2 or profile.shape[0] < 2:
                continue

            scan_points = np.insert(profile, 1, np.zeros(len(profile[0])), axis=0)
            scan_points[0] *= -1.0
            scan_points = scan_points.T
            scan_points = np.matmul(scanner_T.R, scan_points.T).T + scanner_T.p

            scan_points = scan_points[np.all(np.isfinite(scan_points), axis=1)]
            if len(scan_points) == 0:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan_points)
            pcd_all += pcd

        o3d.visualization.draw_geometries([pcd_all])
    else:
        print("Scanner disabled")

    plt.figure(figsize=(12, 7))
    for i in range(6):
        plt.plot(lam_planned, np.degrees(curve_js[:, i]), label=f"Joint {i + 1} planned")
        if len(curve_js_exe) > 0:
            plt.plot(lam_exe, np.degrees(curve_js_exe[:, i]), linestyle="--",
                     label=f"Joint {i + 1} executed")
    plt.legend()
    plt.xlabel("Path length (mm)")
    plt.ylabel("Joint angle (degrees)")
    plt.title("Planned vs executed joint trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("End of Program")
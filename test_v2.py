import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM

TIMESTEP = 0.004  # 4 ms EGM timestep

# ---------------------------------------------------------------
# PATH — rig frame (mm)
# Straight → arc corner → straight
# ---------------------------------------------------------------
Z          = 25.0
ARC_RADIUS = 10.0

P_START  = np.array([70.0, 120.0, Z])
P_CORNER = np.array([20.0, 120.0, Z])
P_END    = np.array([20.0,  70.0, Z])

# ---------------------------------------------------------------
# SPEED
# ---------------------------------------------------------------
MAX_VEL  = 5.0
tool_vel = 1.0   # mm/s
tool_acc = 1.0   # mm/s^2
jog_vel  = 5.0   # mm/s
jog_acc  = 5.0   # mm/s^2

tool_vel = min(tool_vel, MAX_VEL)
jog_vel  = min(jog_vel,  MAX_VEL)

# ---------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------
J6_CAL_OFFSET    = -6.94   # degrees — corrected 2026-06-26
SCANNER_LAG_MM   = 12.7    # scanner trails pen by this distance in travel direction

# ---------------------------------------------------------------
# ROBOT DEFINITION
# ---------------------------------------------------------------
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f:
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([-25.45, 7.27, 131.2])
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# ---------------------------------------------------------------
# RIG KINEMATICS
# ---------------------------------------------------------------
final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = 2.0549 * np.pi / 180
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)
Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz

# ---------------------------------------------------------------
# BUILD DENSE PATH: straight → arc → straight
# ---------------------------------------------------------------
dlam_des  = 0.02
corner_R  = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

d1 = (P_CORNER - P_START)[:2];  d1 = d1 / np.linalg.norm(d1)
d2 = (P_END    - P_CORNER)[:2]; d2 = d2 / np.linalg.norm(d2)

arc_p1 = P_CORNER[:2] - ARC_RADIUS * d1
arc_p2 = P_CORNER[:2] + ARC_RADIUS * d2

perp_d1 = np.array([-d1[1], d1[0]])
arc_center_xy = arc_p1 + ARC_RADIUS * perp_d1

theta1 = np.arctan2(arc_p1[1] - arc_center_xy[1], arc_p1[0] - arc_center_xy[0])
theta2 = np.arctan2(arc_p2[1] - arc_center_xy[1], arc_p2[0] - arc_center_xy[0])
if theta2 < theta1:
    theta2 += 2 * np.pi

seg1_end = np.append(arc_p1, Z)
n1 = max(int(np.linalg.norm(seg1_end - P_START) / dlam_des) + 1, 2)
seg1 = np.linspace(P_START, seg1_end, n1)

arc_len = ARC_RADIUS * abs(theta2 - theta1)
n_arc = max(int(arc_len / dlam_des) + 1, 2)
arc_theta = np.linspace(theta1, theta2, n_arc)
arc_pts = np.column_stack([
    arc_center_xy[0] + ARC_RADIUS * np.cos(arc_theta),
    arc_center_xy[1] + ARC_RADIUS * np.sin(arc_theta),
    np.full(n_arc, Z),
])

seg2_start = np.append(arc_p2, Z)
n2 = max(int(np.linalg.norm(P_END - seg2_start) / dlam_des) + 1, 2)
seg2 = np.linspace(seg2_start, P_END, n2)

curve_p_rig = np.vstack([seg1[:-1], arc_pts[:-1], seg2])

curve_p = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R = np.tile(Rbr @ corner_R, (len(curve_p), 1, 1))

# ---------------------------------------------------------------
# SCANNER ANGLES — scanner-as-follower
#
# j6 is set from the travel direction at the scanner's actual position
# (SCANNER_LAG_MM behind the pen along the path). On straights this is
# identical to the pen's direction; through arcs j6 transitions smoothly
# and the scanner stays perpendicular to the channel beneath it.
# ---------------------------------------------------------------
travel_dirs = np.diff(curve_p[:, :2], axis=0)
travel_dirs = np.vstack([travel_dirs, travel_dirs[-1]])
travel_angles = np.arctan2(travel_dirs[:, 1], travel_dirs[:, 0])

# Arc-length along path
lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
lam = np.insert(lam, 0, 0)

scanner_angles = np.zeros(len(curve_p))
for i in range(len(curve_p)):
    lag_idx = np.searchsorted(lam, max(lam[i] - SCANNER_LAG_MM, 0))
    scanner_angles[i] = np.radians(J6_CAL_OFFSET - 90.0) - travel_angles[lag_idx]

# Wrap to [-pi, pi] first (formula can produce values like -277° for -X travel),
# then unwrap to ensure smooth transitions with no jumps > 180°
scanner_angles = (scanner_angles + np.pi) % (2 * np.pi) - np.pi
scanner_angles = np.unwrap(scanner_angles)

# ---------------------------------------------------------------
# INVERSE KINEMATICS — iterative j6 compensation
# ---------------------------------------------------------------
terminate_threshold = 0.001
IK_MAX_ITER = 5

print("IK (iterative j6 compensation)...")
curve_js = []
q_seed = np.zeros(6)

for i in range(len(curve_p)):
    if i % 5000 == 0 or i == len(curve_p) - 1:
        print(f"  {i}/{len(curve_p)-1}")

    j6_des   = scanner_angles[i]
    p_target = curve_p[i].copy()

    q = robot6_sphericalwrist_invkin(robot, Transform(curve_R[i], p_target), q_seed)[0]
    for _ in range(IK_MAX_ITER):
        q[5] = j6_des
        p_actual = fwdkin(robot, q).p
        error = curve_p[i] - p_actual
        if np.linalg.norm(error) < terminate_threshold:
            break
        p_target = p_target + error
        q = robot6_sphericalwrist_invkin(robot, Transform(curve_R[i], p_target), q)[0]

    q[5] = j6_des
    curve_js.append(q)
    q_seed = q

curve_js = np.array(curve_js)

tip_errors = [np.linalg.norm(fwdkin(robot, curve_js[i]).p - curve_p[i])
              for i in range(0, len(curve_p), max(1, len(curve_p) // 20))]
print(f"Tip error — max: {max(tip_errors):.4f} mm, mean: {np.mean(tip_errors):.4f} mm")

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def calc_lam_js(curve_js):
    pts = [fwdkin(robot, q).p for q in curve_js]
    lam = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return np.insert(lam, 0, 0)


def trajectory_generate(curve_js, lin_vel, lin_acc):
    lam = calc_lam_js(curve_js)
    if len(lam) > 2 and lin_acc > 0:
        time_bp = np.zeros_like(lam)
        acc = lin_acc
        vel = 0
        for i in range(len(lam)):
            if vel >= lin_vel:
                time_bp[i] = time_bp[i-1] + (lam[i] - lam[i-1]) / lin_vel
            else:
                time_bp[i] = np.sqrt(2 * lam[i] / acc)
                vel = acc * time_bp[i]
        time_bp_half = []
        vel = 0
        for i in range(len(lam)-1, -1, -1):
            if vel >= lin_vel or i <= len(lam) / 2:
                break
            time_bp_half.append(np.sqrt(2 * (lam[-1] - lam[i]) / acc))
            vel = acc * time_bp_half[-1]
        time_bp_half = np.array(time_bp_half)[::-1]
        time_bp_half = time_bp_half * -1 + time_bp_half[0]
        time_bp[-len(time_bp_half):] = (
            time_bp[-len(time_bp_half)-1] + time_bp_half
            + (lam[-len(time_bp_half)] - lam[-len(time_bp_half)-1]) / lin_vel
        )
    else:
        time_bp = lam / lin_vel

    num_steps = int(time_bp[-1] / TIMESTEP)
    traj_q = []
    for step in range(num_steps):
        current_time = step * TIMESTEP
        for i in range(len(time_bp)-1):
            if time_bp[i] <= current_time < time_bp[i+1]:
                seg = i
                break
        frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
        traj_q.append(frac * curve_js[seg+1] + (1 - frac) * curve_js[seg])
    return np.array(traj_q)


def read_position(egm):
    for _ in range(20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if res:
            break
        print("Communication lost, retrying...")
    return np.radians(state.joint_angles)


def position_cmd(q, egm):
    egm.send_to_robot(np.degrees(q))


# ---------------------------------------------------------------
# TRAJECTORY GENERATION
# ---------------------------------------------------------------
fullruntraj_q = trajectory_generate(curve_js, lin_vel=tool_vel, lin_acc=tool_acc)
print(f"Trajectory generated: {len(fullruntraj_q)} steps")

# ---------------------------------------------------------------
# EGM SETUP & EXECUTION
# ---------------------------------------------------------------
mm_egm = abb.egm_minmax(-1e-3, 1e-3)
egm_config = abb.EGMJointTargetConfig(mm_egm, mm_egm, mm_egm, mm_egm, mm_egm, mm_egm, 1000, 1000)
mp = abb.MotionProgram(egm_config=egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80")
client.execute_motion_program(mp, wait=False)
egm = EGM()

print("EGM running. Jogging to start (70, 120, 5)...")

q_start = read_position(egm)

# Wrap j6 delta to [-pi, pi] so the jog always takes the shortest rotation
q_jog_target = curve_js[0].copy()
j6_delta = q_jog_target[5] - q_start[5]
j6_delta = (j6_delta + np.pi) % (2 * np.pi) - np.pi
q_jog_target[5] = q_start[5] + j6_delta

q_approach = np.linspace(q_start, q_jog_target, num=100)
traj_approach = trajectory_generate(q_approach, lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_approach:
    read_position(egm)
    position_cmd(q, egm)

print("At start. Holding for stability...")
for _ in range(125):   # 0.5 s
    read_position(egm)
    position_cmd(curve_js[0], egm)

print("Starting stroke (70,120,5) → arc → (20,70,5)...")
for q in fullruntraj_q:
    read_position(egm)
    position_cmd(q, egm)

print("Stroke complete.")

try:
    client.stop_egm()
except Exception as e:
    print("stop_egm warning:", e)

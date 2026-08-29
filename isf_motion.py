import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM

TIMESTEP = 0.004  # 4 ms EGM timestep

# ===============================================================
# EDIT HERE — Path definition in rig frame (mm)
# Straight → arc corner → straight
# ===============================================================
Z          = 25.0   # forming depth (mm)
ARC_RADIUS = 10.0   # corner arc radius (mm)

P_START  = np.array([100.0, 100.0, Z])   # start of first straight
P_CORNER = np.array([ 50.0, 100.0, Z])   # geometric corner (arc replaces the sharp turn here)
P_END    = np.array([ 50.0,  50.0, Z])   # end of second straight

# ===============================================================
# EDIT HERE — Speed
# ===============================================================
MAX_VEL  = 5.0   # mm/s — hard ceiling applied to ALL robot motion
tool_vel = 1.0   # mm/s — forming speed (must be ≤ MAX_VEL)
tool_acc = 1.0   # mm/s^2
jog_vel  = 5.0   # mm/s — approach speed (must be ≤ MAX_VEL)
jog_acc  = 5.0   # mm/s^2

tool_vel = min(tool_vel, MAX_VEL)
jog_vel  = min(jog_vel,  MAX_VEL)

# ===============================================================
# CALIBRATION — Joint 6 offset (degrees)
# J6_CAL_OFFSET is the joint 6 angle at which the scanner line
# is perpendicular to the -Y travel direction (scanner along X axis).
# Positive j6 = clockwise rotation.
#
# ⚠️ Re-measure this value any time the scanner is removed and
#    reinstalled — the mounting position can shift.
# To re-calibrate:
#   1. Jog the robot so it is moving in the -Y direction.
#   2. Rotate joint 6 until the scanner line is parallel to X axis.
#   3. Read joint 6 angle from the teach pendant and enter it here.
# ===============================================================
J6_CAL_OFFSET = 10.76   # degrees — measured 2026-06-25, scanner perpendicular to -Y travel

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def calc_lam_js(curve_js, robot):
    curve_p = [fwdkin(robot, q).p for q in curve_js]
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
    lam = np.insert(lam, 0, 0)
    return lam


def trajectory_generate(curve_js, robot, lin_vel, lin_acc):
    lam = calc_lam_js(curve_js, robot)
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
        if step % 5000 == 0 or step == num_steps - 1:
            print(f"  Trajectory: {step}/{num_steps} steps")
        current_time = step * TIMESTEP
        for i in range(len(time_bp)-1):
            if time_bp[i] <= current_time < time_bp[i+1]:
                seg = i
                break
        frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
        traj_q.append(frac * curve_js[seg+1] + (1 - frac) * curve_js[seg])
    return np.array(traj_q), np.array(time_bp)


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
# ROBOT DEFINITION
# ---------------------------------------------------------------
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f:
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([-25.45, 7.27, 131.2])   # pen tip offset from flange (mm), in flange frame
# Derived 2026-06-25 from empirical j6 rotation test:
#   j6 rotated 0→90° with j1-j5 fixed; pen tip moved [-16.5, -33.6] mm in world XY.
#   Solved (R_j6_90 - R_j6_0) @ Pft_xy = [-16.5, -33.6] using FlexPendant quaternions.
# NOTE: sign of Pft_x is NEGATIVE (flange X ≈ -world X at this configuration).
#       Old value [+27.94, -0.457, 131.2] had wrong sign → IK compensated in wrong direction.
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
dlam_des = 0.02  # path resolution (mm)

corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T  # fixed EE orientation

# Unit vectors of the two straight segments
d1 = (P_CORNER - P_START)[:2];  d1 = d1 / np.linalg.norm(d1)   # -X direction
d2 = (P_END    - P_CORNER)[:2]; d2 = d2 / np.linalg.norm(d2)   # -Y direction

# Tangent points where arc meets each straight segment
arc_p1 = P_CORNER[:2] - ARC_RADIUS * d1   # end of first straight  = (60, 100)
arc_p2 = P_CORNER[:2] + ARC_RADIUS * d2   # start of second straight = (50,  90)

# Arc center: offset from arc_p1 inward (perpendicular to d1, toward inside of turn)
# Rotate d1 by +90° CCW: (x,y) → (-y, x). For d1=(-1,0) → (0,-1), pointing to inside ✓
perp_d1 = np.array([-d1[1], d1[0]])
arc_center_xy = arc_p1 + ARC_RADIUS * perp_d1   # (60,100) + 10*(0,-1) = (60, 90)

# Arc angles from center to each tangent point
theta1 = np.arctan2(arc_p1[1] - arc_center_xy[1], arc_p1[0] - arc_center_xy[0])  # 90°
theta2 = np.arctan2(arc_p2[1] - arc_center_xy[1], arc_p2[0] - arc_center_xy[0])  # 180°
# Ensure we sweep counterclockwise (increasing angle) through the turn
if theta2 < theta1:
    theta2 += 2 * np.pi

# Segment 1: straight P_START → arc_p1
seg1_end = np.append(arc_p1, Z)
seg1_len = np.linalg.norm(seg1_end - P_START)
n1 = max(int(seg1_len / dlam_des) + 1, 2)
seg1 = np.linspace(P_START, seg1_end, n1)

# Arc segment
arc_len = ARC_RADIUS * abs(theta2 - theta1)
n_arc = max(int(arc_len / dlam_des) + 1, 2)
arc_theta = np.linspace(theta1, theta2, n_arc)
arc_pts = np.column_stack([
    arc_center_xy[0] + ARC_RADIUS * np.cos(arc_theta),
    arc_center_xy[1] + ARC_RADIUS * np.sin(arc_theta),
    np.full(n_arc, Z),
])

# Segment 2: straight arc_p2 → P_END
seg2_start = np.append(arc_p2, Z)
seg2_len = np.linalg.norm(P_END - seg2_start)
n2 = max(int(seg2_len / dlam_des) + 1, 2)
seg2 = np.linspace(seg2_start, P_END, n2)

# Concatenate, dropping duplicate junction points
curve_p_rig = np.vstack([seg1[:-1], arc_pts[:-1], seg2])

# Transform to base frame
curve_p = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R = np.tile(Rbr @ corner_R, (len(curve_p), 1, 1))

# ---------------------------------------------------------------
# SCANNER ANGLES — computed from path tangent in base frame XY
# Formula: j6 = (J6_CAL_OFFSET - 90°) - travel_angle
# At -Y travel (-90°): j6 = (10.76-90) - (-90) = 10.76° ✓
# At -X travel (-180°): j6 = (10.76-90) - (-180) = 100.76° ✓
# ---------------------------------------------------------------
travel_dirs = np.diff(curve_p[:, :2], axis=0)
travel_dirs = np.vstack([travel_dirs, travel_dirs[-1]])
travel_angles = np.arctan2(travel_dirs[:, 1], travel_dirs[:, 0])
scanner_angles = np.radians(J6_CAL_OFFSET - 90.0) - travel_angles

# ---------------------------------------------------------------
# INVERSE KINEMATICS — iterative j6 compensation
#
# Rotating j6 to align the scanner moves the pen tip (27.94 mm radial
# offset from j6 axis). A one-shot analytic correction fails: shifting
# the IK target also shifts the wrist center, which changes the natural
# j6 the IK produces, introducing a new residual error.
#
# Fix: iterate. After each IK solve + j6 override, compute the actual
# tip error and add it back into the IK target for the next solve.
# The map (IK target -> tip with j6 override) has Jacobian ~I near any
# well-conditioned config, so this converges to sub-threshold in 2-3 steps.
# ---------------------------------------------------------------
terminate_threshold = 0.001
IK_MAX_ITER = 5   # converges in ~2 for typical ISF configs

print("IK (iterative j6 compensation)...")
curve_js = []
q_seed = np.zeros(6)

for i in range(len(curve_p)):
    if i % 5000 == 0 or i == len(curve_p) - 1:
        print(f"  {i}/{len(curve_p)-1}")

    j6_des = scanner_angles[i]
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

# Verify final tip accuracy
tip_errors = [np.linalg.norm(fwdkin(robot, curve_js[i]).p - curve_p[i])
              for i in range(0, len(curve_p), max(1, len(curve_p) // 20))]
print(f"Tip error after compensation — max: {max(tip_errors):.4f} mm, mean: {np.mean(tip_errors):.4f} mm")

# ---------------------------------------------------------------
# TRAJECTORY GENERATION
# ---------------------------------------------------------------
fullruntraj_q, _ = trajectory_generate(curve_js, robot, lin_vel=tool_vel, lin_acc=tool_acc)
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

print("EGM running. Jogging to start position...")

# Jog from current position to path start
q_start = read_position(egm)
q_approach = np.linspace(q_start, curve_js[0], num=100)
traj_approach, _ = trajectory_generate(q_approach, robot, lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_approach:
    read_position(egm)
    position_cmd(q, egm)
print("At start position. Holding for stability before stroke...")

# The two-pass IK already bakes the correct j6 and arm compensation into curve_js[0].
# The jog above delivered the robot there, so no separate settle IK is needed.
# A separate settle IK solve risks landing on a different IK branch, which would
# cause an instantaneous joint jump (jerk) when the main trajectory starts from curve_js[0].
# Instead, just hold at curve_js[0] for 0.5 s to let the robot settle.
for _ in range(125):   # 125 × 4 ms = 0.5 s
    read_position(egm)
    position_cmd(curve_js[0], egm)

print("Scanner in position. Starting main trajectory...")

# Main trajectory
for q in fullruntraj_q:
    read_position(egm)
    position_cmd(q, egm)

print("Trajectory complete.")

try:
    client.stop_egm()
except Exception as e:
    print("stop_egm warning:", e)

# ═══════════════════════════════════════════════════════════════════
#  isf_motion_v4.py — Scanner movement / j6 tracking validation
#
#  Goal: fix and validate scanner (j6) behaviour with no force control.
#  The robot follows the pre-planned path at a flat Z value — no ATI
#  sensor, no z_corr, no live Z adjustment.  With force control removed,
#  any remaining issues in j6 tracking, scanner angle computation,
#  Algorithm 1 convergence, or path geometry can be isolated cleanly.
#
#  Differences from v3:
#  - No ATI connection, no force control parameters
#  - No z_corr — robot executes planned joint angles exactly
#  - Loop is minimal: joint velocity clamp only
#  - Z, scanner lag, j6 calibration can all be tuned here without
#    the force loop interfering
# ═══════════════════════════════════════════════════════════════════

import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM
import time

TIMESTEP = 0.004  # 4 ms EGM timestep

# ═══════════════════════════════════════════════════════════════════
#  PATH — rig frame (mm)
#
#  Serpentine design: 3 parallel lines in the Y direction connected by
#  two 5 mm-radius 180° arcs.
#
#    Line 1  X=150  Y_TOP → Y_BOT   (travel direction -Y)
#    Arc 1   CW semicircle at bottom  X: 150 → 140
#    Line 2  X=140  Y_BOT → Y_TOP   (travel direction +Y)
#    Arc 2   CCW semicircle at top   X: 140 → 130
#    Line 3  X=130  Y_TOP → Y_BOT   (travel direction -Y)
#    Lift + scanner tail at end of Line 3
#
#  Because Lines 1 and 3 both travel in -Y, j6 starts and ends at the
#  same angle, preventing cable wrap accumulation.
# ═══════════════════════════════════════════════════════════════════
Z            = -8.75
Z_SAFE       = 20.0
Z_END        = 25.0
ARC_RADIUS   =  2.5   # mm — radius of the 180° connecting arcs
Z_SCAN_LIFT  =  1.0   # mm — pen rises above Z at end of pass
SCAN_TAIL_MM = 12.5   # mm — scanner tail length past the last line end

X_L1  = 150.0   # X of line 1
X_L2  = X_L1 - 2 * ARC_RADIUS   # 140.0
X_L3  = X_L2 - 2 * ARC_RADIUS   # 130.0
Y_TOP = 115.0
Y_BOT =  85.0

# ═══════════════════════════════════════════════════════════════════
#  SPEED
# ═══════════════════════════════════════════════════════════════════
MAX_VEL    = 5.0
tool_vel   = 1.0    # mm/s
corner_vel = 0.15   # mm/s — slow enough for j6 to track perpendicular at ≤5°/s through 2.5 mm arcs
tool_acc   = 1.0    # mm/s²
jog_vel    = 5.0    # mm/s
jog_acc    = 1.0    # mm/s²

MAX_JOINT_VEL_DEG_S = np.array([150.0, 150.0, 150.0, 150.0, 150.0, 5.0])

tool_vel   = min(tool_vel,   MAX_VEL)
corner_vel = min(corner_vel, MAX_VEL)
jog_vel    = min(jog_vel,    MAX_VEL)

# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════
J6_CAL_OFFSET  = 10.76   # degrees
SCANNER_LAG_MM = 12.7     # mm  (0.5 inch)

# ═══════════════════════════════════════════════════════════════════
#  ROBOT DEFINITION
# ═══════════════════════════════════════════════════════════════════
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f:
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([0.0, 0.0, 131.2])
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# ═══════════════════════════════════════════════════════════════════
#  RIG KINEMATICS
# ═══════════════════════════════════════════════════════════════════
final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = 2.0549 * np.pi / 180
vz  = final_rig_pose[0:3, 2]
Rz  = rot(vz, z_theta)
Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz

# ═══════════════════════════════════════════════════════════════════
#  BUILD PATH: serpentine (3 lines + 2 arcs) → lift → scanner tail
#
#  Arc 1 (bottom, CW): incoming -Y, outgoing +Y, center (145, Y_BOT)
#    theta: 0 → -π  (dips below Y_BOT by ARC_RADIUS)
#  Arc 2 (top, CCW): incoming +Y, outgoing -Y, center (135, Y_TOP)
#    theta: 0 → +π  (rises above Y_TOP by ARC_RADIUS)
#
#  EE orientation corner_R keeps tool -Z into the sheet throughout.
# ═══════════════════════════════════════════════════════════════════
dlam_des = 0.02
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

n_line = max(int((Y_TOP - Y_BOT) / dlam_des) + 1, 2)
n_arc  = max(int(np.pi * ARC_RADIUS / dlam_des) + 1, 2)

# Line 1: X_L1, Y_TOP → Y_BOT  (direction -Y)
line1 = np.column_stack([np.full(n_line, X_L1),
                         np.linspace(Y_TOP, Y_BOT, n_line),
                         np.full(n_line, Z)])

# Arc 1: CW 180° at bottom connecting Line 1 → Line 2
C1        = np.array([X_L1 - ARC_RADIUS, Y_BOT])
arc1_t    = np.linspace(0, -np.pi, n_arc)
arc1      = np.column_stack([C1[0] + ARC_RADIUS * np.cos(arc1_t),
                              C1[1] + ARC_RADIUS * np.sin(arc1_t),
                              np.full(n_arc, Z)])

# Line 2: X_L2, Y_BOT → Y_TOP  (direction +Y)
line2 = np.column_stack([np.full(n_line, X_L2),
                         np.linspace(Y_BOT, Y_TOP, n_line),
                         np.full(n_line, Z)])

# Arc 2: CCW 180° at top connecting Line 2 → Line 3
C2        = np.array([X_L2 - ARC_RADIUS, Y_TOP])
arc2_t    = np.linspace(0, np.pi, n_arc)
arc2      = np.column_stack([C2[0] + ARC_RADIUS * np.cos(arc2_t),
                              C2[1] + ARC_RADIUS * np.sin(arc2_t),
                              np.full(n_arc, Z)])

# Line 3: X_L3, Y_TOP → Y_BOT  (direction -Y, same as Line 1)
line3 = np.column_stack([np.full(n_line, X_L3),
                         np.linspace(Y_TOP, Y_BOT, n_line),
                         np.full(n_line, Z)])

# Lift at end of Line 3
P_END     = np.array([X_L3, Y_BOT, Z])
P_LIFTED  = np.array([X_L3, Y_BOT, Z + Z_SCAN_LIFT])
n_lift    = max(int(Z_SCAN_LIFT / dlam_des) + 1, 2)
lift_seg  = np.linspace(P_END, P_LIFTED, n_lift)

# Scanner tail continues in -Y direction (same as Line 3 exit direction)
d_last    = np.array([0.0, -1.0])
P_SCAN_END = np.array([X_L3 + d_last[0] * SCAN_TAIL_MM,
                       Y_BOT + d_last[1] * SCAN_TAIL_MM,
                       Z + Z_SCAN_LIFT])
n_tail    = max(int(SCAN_TAIL_MM / dlam_des) + 1, 2)
tail_seg  = np.linspace(P_LIFTED, P_SCAN_END, n_tail)

# Concatenate (drop duplicate junction points)
curve_p_rig = np.vstack([line1[:-1], arc1[:-1],
                          line2[:-1], arc2[:-1],
                          line3[:-1], lift_seg[:-1], tail_seg])

is_arc_waypoint = np.array(
    [False] * (len(line1)    - 1) +
    [True]  * (len(arc1)     - 1) +
    [False] * (len(line2)    - 1) +
    [True]  * (len(arc2)     - 1) +
    [False] * (len(line3)    - 1) +
    [False] * (len(lift_seg) - 1) +
    [False] * len(tail_seg),
    dtype=bool
)

curve_p = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R = np.tile(Rbr @ corner_R, (len(curve_p), 1, 1))

print(f"Path: {len(curve_p)} waypoints  "
      f"(line1={len(line1)-1}, arc1={len(arc1)-1}, "
      f"line2={len(line2)-1}, arc2={len(arc2)-1}, "
      f"line3={len(line3)-1}, lift={len(lift_seg)-1}, tail={len(tail_seg)})")

# ═══════════════════════════════════════════════════════════════════
#  SCANNER OFFSET IN FLANGE FRAME
# ═══════════════════════════════════════════════════════════════════
P_SCAN_FLANGE = np.array([0.0, SCANNER_LAG_MM, 0.0])

# ═══════════════════════════════════════════════════════════════════
#  CHANNEL TANGENT VECTORS (base frame)
# ═══════════════════════════════════════════════════════════════════
_td    = np.diff(curve_p[:, :2], axis=0)
_td    = np.vstack([_td, _td[-1]])
_norms = np.linalg.norm(_td, axis=1, keepdims=True).clip(min=1e-9)
channel_tangents = (_td / _norms).astype(float)

# ═══════════════════════════════════════════════════════════════════
#  IK + ALGORITHM 1 SCANNER ANGLES
# ═══════════════════════════════════════════════════════════════════
print("IK + Algorithm 1 scanner angles...")
curve_js = []
raw_j6   = []
q_seed   = np.zeros(6)

_d0      = curve_p[1, :2] - curve_p[0, :2] if len(curve_p) > 1 else np.array([1.0, 0.0])
_j6_seed = float(np.radians(J6_CAL_OFFSET - 90.0) - np.arctan2(_d0[1], _d0[0]))

for i in range(len(curve_p)):
    if i % 5000 == 0 or i == len(curve_p) - 1:
        print(f"  {i}/{len(curve_p) - 1}")

    q    = robot6_sphericalwrist_invkin(robot, Transform(curve_R[i], curve_p[i]), q_seed)[0]
    q[5] = _j6_seed

    for _it in range(10):
        T      = fwdkin(robot, q)
        p_scan = T.p[:2] + (T.R @ P_SCAN_FLANGE)[:2]

        dists   = np.linalg.norm(curve_p[:, :2] - p_scan, axis=1)
        n_idx   = int(np.argmin(dists))
        tangent = channel_tangents[n_idx]

        j6_tgt = float(np.radians(J6_CAL_OFFSET - 90.0)
                       - np.arctan2(tangent[1], tangent[0]))

        delta = (j6_tgt - q[5] + np.pi) % (2 * np.pi) - np.pi
        if abs(delta) < np.radians(0.05):
            q[5] = j6_tgt
            break
        q[5] += 0.5 * delta

    raw_j6.append(q[5])
    _j6_seed = q[5]
    curve_js.append(q)
    q_seed = q

curve_js = np.array(curve_js)

# ═══════════════════════════════════════════════════════════════════
#  J6 PROFILE: arc-length lag-follower
#
#  The pen leads.  The scanner follows — it must always point at the
#  pen's position SCANNER_LAG_MM (12.7 mm) ago along the path.
#
#  For each waypoint i:
#    1. Walk back 12.7 mm along the path in arc length → lag_pos
#    2. tangent = unit vector from lag_pos to pen tip (forward dir)
#    3. j6 = J6_CAL_OFFSET_rad - π/2 - arctan2(ty, tx)
#       → scanner points opposite to tangent, i.e. toward lag_pos
#
#  This is perpendicular to the channel at the lag position by
#  construction (scanner ⊥ channel direction at the measurement point).
#
#  On straight lines: lag_pos is exactly 12.7 mm behind, j6 is
#  constant.  Through arc transitions: lag_pos walks along the actual
#  path arc, so j6 changes smoothly over a window of
#  (arc_length + SCANNER_LAG_MM) centred on the corner — the rotation
#  starts before the arc and finishes after it.
# ═══════════════════════════════════════════════════════════════════

# Cumulative arc length along the rig-frame path (XY only, Z is flat)
_seg_lens = np.linalg.norm(np.diff(curve_p_rig[:, :2], axis=0), axis=1)
arc_len   = np.concatenate([[0.0], np.cumsum(_seg_lens)])

# For each waypoint, find the path point SCANNER_LAG_MM behind in arc length
lag_pos_rig = np.empty_like(curve_p_rig)
_d0_dir = curve_p_rig[1, :2] - curve_p_rig[0, :2]
_d0_dir = _d0_dir / np.linalg.norm(_d0_dir)
for i in range(len(curve_p_rig)):
    target = arc_len[i] - SCANNER_LAG_MM
    if target <= 0:
        # Before lag catches up: extend backward behind the path start
        lag_pos_rig[i, :2] = curve_p_rig[0, :2] + _d0_dir * target  # target < 0 → behind
        lag_pos_rig[i, 2]  = curve_p_rig[0, 2]
    else:
        j       = int(np.searchsorted(arc_len, target, side='right')) - 1
        j       = min(j, len(arc_len) - 2)
        seg_len = arc_len[j + 1] - arc_len[j]
        frac    = (target - arc_len[j]) / seg_len if seg_len > 1e-9 else 0.0
        lag_pos_rig[i] = curve_p_rig[j] + frac * (curve_p_rig[j + 1] - curve_p_rig[j])

lag_pos_world = np.array([Rbr @ p + Pbr for p in lag_pos_rig])

# Compute j6 from lag direction
raw_j6 = np.empty(len(curve_p))
for i in range(len(curve_p)):
    delta = curve_p[i, :2] - lag_pos_world[i, :2]   # lag → pen (forward)
    norm  = np.linalg.norm(delta)
    if norm < 1e-9:
        raw_j6[i] = np.radians(J6_CAL_OFFSET)
    else:
        tx, ty   = delta / norm
        raw_j6[i] = np.radians(J6_CAL_OFFSET - 90.0) - np.arctan2(ty, tx)

# Unwrap for continuity, shift so first value = J6_CAL_OFFSET
scanner_angles = np.unwrap(raw_j6)
_shift = np.radians(J6_CAL_OFFSET) - scanner_angles[0]
_shift = np.round(_shift / (2 * np.pi)) * (2 * np.pi)
scanner_angles += _shift

for i in range(len(curve_js)):
    curve_js[i][5] = scanner_angles[i]

j6_min_deg = np.degrees(scanner_angles.min())
j6_max_deg = np.degrees(scanner_angles.max())
print(f"J6 command range: {j6_min_deg:.1f}° to {j6_max_deg:.1f}°  (robot limit ±400°)")
if abs(scanner_angles.min()) > np.radians(390) or abs(scanner_angles.max()) > np.radians(390):
    raise RuntimeError(f"J6 out of range: [{j6_min_deg:.1f}°, {j6_max_deg:.1f}°]")

# Pen tip on channel.  Verify scanner lands at the lag position.
scan_errors = []
for i in range(0, len(curve_p), max(1, len(curve_p) // 20)):
    T      = fwdkin(robot, curve_js[i])
    p_scan = T.p + T.R @ P_SCAN_FLANGE
    scan_errors.append(np.linalg.norm(p_scan[:2] - lag_pos_world[i, :2]))
print(f"Scanner-to-lag error — max: {max(scan_errors):.4f} mm, mean: {np.mean(scan_errors):.4f} mm")

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def calc_lam_js(qs):
    pts = np.array([fwdkin(robot, q).p for q in qs])
    lam = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return np.insert(lam, 0, 0)


def trajectory_generate(qs, lin_vel, lin_acc):
    lam = calc_lam_js(qs)
    if len(lam) > 2 and lin_acc > 0:
        time_bp = np.zeros_like(lam)
        vel = 0.0
        for i in range(len(lam)):
            if vel >= lin_vel:
                time_bp[i] = time_bp[i-1] + (lam[i] - lam[i-1]) / lin_vel
            else:
                time_bp[i] = np.sqrt(2 * lam[i] / lin_acc)
                vel = lin_acc * time_bp[i]
        half = []
        vel  = 0.0
        for i in range(len(lam) - 1, -1, -1):
            if vel >= lin_vel or i <= len(lam) / 2:
                break
            half.append(np.sqrt(2 * (lam[-1] - lam[i]) / lin_acc))
            vel = lin_acc * half[-1]
        half = np.array(half)[::-1]
        half = half * -1 + half[0]
        time_bp[-len(half):] = (
            time_bp[-len(half) - 1] + half
            + (lam[-len(half)] - lam[-len(half) - 1]) / lin_vel
        )
    else:
        time_bp = lam / lin_vel

    num_steps = int(time_bp[-1] / TIMESTEP)
    traj = []
    for step in range(num_steps):
        t = step * TIMESTEP
        for i in range(len(time_bp) - 1):
            if time_bp[i] <= t < time_bp[i + 1]:
                seg = i
                break
        frac = (t - time_bp[seg]) / (time_bp[seg + 1] - time_bp[seg])
        traj.append(frac * qs[seg + 1] + (1 - frac) * qs[seg])
    return np.array(traj), time_bp


def trajectory_generate_continuous(qs, vel_targets, lin_acc):
    pts = np.array([fwdkin(robot, q).p for q in qs])
    ds  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    n   = len(qs)

    v_fwd    = np.zeros(n)
    for i in range(1, n):
        v_achievable = np.sqrt(max(v_fwd[i-1]**2 + 2.0 * lin_acc * ds[i-1], 0.0))
        v_fwd[i] = min(vel_targets[i], v_achievable)

    v_bwd     = np.zeros(n)
    for i in range(n - 2, -1, -1):
        v_achievable = np.sqrt(max(v_bwd[i+1]**2 + 2.0 * lin_acc * ds[i], 0.0))
        v_bwd[i] = min(v_fwd[i], v_achievable)

    v = np.minimum(v_fwd, v_bwd)

    time_bp = np.zeros(n)
    for i in range(1, n):
        v_avg = 0.5 * (v[i-1] + v[i])
        time_bp[i] = time_bp[i-1] + (ds[i-1] / v_avg if v_avg > 1e-9 else 0.0)

    num_steps = int(time_bp[-1] / TIMESTEP)
    traj = []
    for step in range(num_steps):
        t   = step * TIMESTEP
        idx = int(np.searchsorted(time_bp, t, side='right')) - 1
        idx = min(max(idx, 0), n - 2)
        dt  = time_bp[idx + 1] - time_bp[idx]
        frac = (t - time_bp[idx]) / dt if dt > 1e-9 else 0.0
        traj.append((1.0 - frac) * qs[idx] + frac * qs[idx + 1])

    print(f"Continuous trajectory: {num_steps} steps  ({time_bp[-1]:.1f} s total)")
    return np.array(traj), time_bp


def read_position(egm):
    for _ in range(20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if res:
            break
        print("Communication lost, retrying...")
    return np.radians(state.joint_angles)


def position_cmd(q, egm):
    egm.send_to_robot(np.degrees(q))


def clamp_joint_vel(traj_q, max_vel_deg_s):
    if len(traj_q) == 0:
        return traj_q
    max_delta = np.radians(max_vel_deg_s) * TIMESTEP
    out = traj_q.copy().astype(float)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        out[i] = out[i - 1] + np.clip(delta, -max_delta, max_delta)
    return out


# ═══════════════════════════════════════════════════════════════════
#  TRAJECTORY GENERATION
# ═══════════════════════════════════════════════════════════════════
vel_targets   = np.where(is_arc_waypoint, corner_vel, tool_vel).astype(float)
fullruntraj_q, _ = trajectory_generate_continuous(curve_js, vel_targets, tool_acc)
fullruntraj_q    = clamp_joint_vel(fullruntraj_q, MAX_JOINT_VEL_DEG_S)

# ═══════════════════════════════════════════════════════════════════
#  HOVER POSITIONS
# ═══════════════════════════════════════════════════════════════════
def hover_ik(p_rig, j6_target, R_target, q_seed_init):
    p_robot = Rbr @ p_rig + Pbr
    q = robot6_sphericalwrist_invkin(robot, Transform(R_target, p_robot), q_seed_init)[0]
    q[5] = j6_target
    return q

p_hover_start_rig = np.array([curve_p_rig[0][0],  curve_p_rig[0][1],  Z_SAFE])
p_hover_end_rig   = np.array([curve_p_rig[-1][0], curve_p_rig[-1][1], Z_END])

q_hover_start = hover_ik(p_hover_start_rig, curve_js[0][5],  curve_R[0],  curve_js[0])
q_hover_end   = hover_ik(p_hover_end_rig,   curve_js[-1][5], curve_R[-1], curve_js[-1])

# ═══════════════════════════════════════════════════════════════════
#  EGM SETUP & EXECUTION
# ═══════════════════════════════════════════════════════════════════
mm_egm     = abb.egm_minmax(-1e-3, 1e-3)
egm_config = abb.EGMJointTargetConfig(
    mm_egm, mm_egm, mm_egm, mm_egm, mm_egm, mm_egm, 1000, 1000
)
mp = abb.MotionProgram(egm_config=egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)
client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80")
client.execute_motion_program(mp, wait=False)
egm = EGM()

print("EGM running. Jogging to start position...")
q_start = read_position(egm)

# Real-time j6 clamp — applied in every loop, every phase
max_delta_rad = np.radians(MAX_JOINT_VEL_DEG_S) * TIMESTEP
q_prev_cmd    = q_start.copy()

def send_clamped(q_target):
    """Send q_target through the real-time velocity clamp and update q_prev_cmd."""
    global q_prev_cmd
    q_adj      = q_prev_cmd + np.clip(q_target - q_prev_cmd, -max_delta_rad, max_delta_rad)
    q_prev_cmd = q_adj.copy()
    position_cmd(q_adj, egm)

# Phase 1: pre-rotate j6 to calibration angle before any Z movement
q_pre_j6     = q_start.copy()
q_pre_j6[5]  = np.radians(J6_CAL_OFFSET)
j6_delta_rad = abs(q_pre_j6[5] - q_start[5])
j6_vel_rad   = np.radians(MAX_JOINT_VEL_DEG_S[5])
n_j6_steps   = max(int(j6_delta_rad * np.pi / (2.0 * j6_vel_rad * TIMESTEP)) + 1, 20)
t_norm       = np.linspace(0.0, 1.0, n_j6_steps)
smooth_t     = 0.5 * (1.0 - np.cos(np.pi * t_norm))
traj_j6      = q_start + smooth_t[:, np.newaxis] * (q_pre_j6 - q_start)
print(f"Phase 1: rotating j6 by {np.degrees(j6_delta_rad):.1f} deg "
      f"in {n_j6_steps} steps ({n_j6_steps * TIMESTEP:.1f} s) ...")
for q in traj_j6:
    read_position(egm)
    send_clamped(q)

print("Settling (0.5 s)...")
for _ in range(125):
    read_position(egm)
    send_clamped(q_pre_j6)

# Phase 2: jog to hover
traj_approach, _ = trajectory_generate(np.linspace(q_pre_j6, q_hover_start, 500),
                                        lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_approach:
    read_position(egm)
    send_clamped(q)

print("Settling at hover (0.5 s)...")
for _ in range(125):
    read_position(egm)
    send_clamped(q_hover_start)

print(f"Hovering at Z={Z_SAFE} mm. Plunging to Z={Z} mm...")

# Phase 3: plunge
traj_plunge, _ = trajectory_generate(np.linspace(q_hover_start, curve_js[0], 200),
                                      lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_plunge:
    read_position(egm)
    send_clamped(q)

print("At start. Holding 0.5 s...")
for _ in range(125):
    read_position(egm)
    send_clamped(curve_js[0])

J6_TOL = np.radians(0.5)
j6_converged = False
for _ in range(500):
    q_actual = read_position(egm)
    send_clamped(curve_js[0])
    j6_err = abs(q_actual[5] - curve_js[0][5])
    if j6_err < J6_TOL:
        j6_converged = True
        break

if not j6_converged:
    print(f"WARNING: j6 did not converge — error: {np.degrees(j6_err):.2f}°")
else:
    print(f"J6 converged. Pre-stroke error: {np.degrees(j6_err):.2f}°")

# ═══════════════════════════════════════════════════════════════════
#  FORMING + SCAN PASS — no force control, flat Z
# ═══════════════════════════════════════════════════════════════════
print("Starting trajectory (no force control)...")
j6_max_err_rad = 0.0

for i, q_cmd in enumerate(fullruntraj_q):
    q_actual = read_position(egm)
    send_clamped(q_cmd)

    j6_err = abs(q_actual[5] - q_prev_cmd[5])
    if j6_err > j6_max_err_rad:
        j6_max_err_rad = j6_err

print(f"Trajectory complete. Max j6 tracking error: {np.degrees(j6_max_err_rad):.2f}°")

# Retract
print(f"Retracting to Z={Z_END} mm...")
traj_retract, _ = trajectory_generate(np.linspace(q_prev_cmd, q_hover_end, 200),
                                       lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_retract:
    read_position(egm)
    send_clamped(q)

print("Retract complete. Holding 1 s...")
for _ in range(250):
    read_position(egm)
    send_clamped(q_hover_end)

try:
    client.stop_egm()
except Exception as e:
    print("stop_egm warning:", e)

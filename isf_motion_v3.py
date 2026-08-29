# ═══════════════════════════════════════════════════════════════════
#  isf_motion_v3.py — Force-controlled ISF forming pass
#
#  What this file does:
#  - Plans a straight → arc → straight path in rig frame
#  - Uses Algorithm 1 to set j6 so the scanner always faces the channel
#  - Runs the forming pass under closed-loop force control via the ATI
#    F/T sensor: a P-controller adjusts z_corr (a live Z offset) each
#    4 ms EGM step to maintain F_TARGET = -1.5 N against the sheet
#  - Rate-limits z_corr to Z_CORR_RATE mm/s to prevent runaway descent
#  - Force control is active only between step 5 and N-6 to avoid noisy
#    start/end data
#  - At the end of the forming pass, z_corr is zeroed and the pen lifts
#    to Z = -8 mm so the scanner tail can measure without pen contact
# ═══════════════════════════════════════════════════════════════════

import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM
from RobotRaconteur.Client import *
import RobotRaconteur as RR
import time

TIMESTEP = 0.004  # 4 ms EGM timestep

# ═══════════════════════════════════════════════════════════════════
#  PATH — rig frame (mm)
#  Test path: straight (-X) → arc → straight (-Y)
#  Same geometry as test_v2.py.  Z = 25 (no contact — safe test height).
# ═══════════════════════════════════════════════════════════════════
Z            = -9.0
Z_SAFE       = 20.0   # hover height on approach (10 mm above Z)
Z_END        = 25.0   # retract height after run
ARC_RADIUS   = 10.0
Z_SCAN_LIFT  =  1.0   # mm — lifts pen from Z=-9 to Z=-8 mm at end of pass
SCAN_TAIL_MM = 12.5   # mm — distance to travel past P_END at lifted height
                      #       (matches scanner lag ~12.7 mm)

P_START  = np.array([150.0, 115.0, Z])
P_CORNER = np.array([100.0, 115.0, Z])
P_END    = np.array([100.0,  65.0, Z])

# ═══════════════════════════════════════════════════════════════════
#  SPEED
# ═══════════════════════════════════════════════════════════════════
MAX_VEL    = 5.0
tool_vel   = 1.0    # mm/s — straight segment speed
corner_vel = 0.3    # mm/s — arc speed
tool_acc   = 1.0    # mm/s² — forming trajectory acceleration limit
jog_vel    = 5.0    # mm/s — approach / retract speed
jog_acc    = 1.0    # mm/s²

# Maximum joint velocities — safety clamp applied to every trajectory phase.
# j6 is capped low (30°/s) to protect the scanner bracket.
# IRB1200 mechanical limits are ~180-400 deg/s; these are conservative test values.
MAX_JOINT_VEL_DEG_S = np.array([150.0, 150.0, 150.0, 150.0, 150.0, 5.0])

# ═══════════════════════════════════════════════════════════════════
#  FORCE CONTROL
#
#  ATI convention: force.z is negative when the pen presses into the
#  sheet (reaction force compresses the sensor).  If you see the pen
#  lifting instead of pressing, flip FORCE_SIGN to -1.
#
#  KF: proportional gain — mm of Z correction per N of force error
#      per 4 ms step.  0.005 is conservative; raise if response is
#      sluggish.  Existing scripts use 60 * 0.05 * 0.004 = 0.012.
# ═══════════════════════════════════════════════════════════════════
FORCE_SIGN        = -1        # raw ATI force.z is positive when pressing; flip to negative
F_TARGET          = -1.5     # N  (negative = pressing into surface)
KF                =  0.005   # mm / (N · step)  proportional gain
Z_CORR_RATE       =  0.1     # mm/s — max rate z_corr may change (caps descent/lift speed)
Z_CORR_MAX        = 15.0     # mm  max Z correction magnitude
FORCE_LIMIT       = 15.0     # N  magnitude — abort if exceeded
LIFTOFF_THRESHOLD =  0.4     # N  |force| below this → pen truly off surface

tool_vel   = min(tool_vel,   MAX_VEL)
corner_vel = min(corner_vel, MAX_VEL)
jog_vel    = min(jog_vel,    MAX_VEL)

# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
#
#  J6_CAL_OFFSET: j6 angle at which the scanner beam is perpendicular
#  to the channel when the robot travels in the -Y direction.
#  Measured 2026-06-25.  Re-verify after new EE installation.
#
#  SCANNER_LAG_MM: pen-to-scanner distance along the travel direction.
#  0.5 inch = 12.7 mm.  j6 is set from the travel direction at the
#  scanner's position (this distance behind the pen along the path).
# ═══════════════════════════════════════════════════════════════════
J6_CAL_OFFSET  = -10.45 #10.76   # degrees
SCANNER_LAG_MM = 12.7    # mm  (0.5 inch)

# ═══════════════════════════════════════════════════════════════════
#  ROBOT DEFINITION
#
#  Hardware update 2026-07-03: pen repositioned to sit directly below
#  the j6 rotation axis.  Pft XY = 0 — j6 rotation no longer moves
#  the pen tip.  No iterative Pft correction needed in IK.
#  Z component (tool length) estimated at 131.2 mm — re-verify.
# ═══════════════════════════════════════════════════════════════════
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as f:
    robot = rr_rox.load_robot_info_yaml_to_robot(f)

Pft = np.array([0.0, 0.0, 131.2])   # pen on j6 axis — XY = 0
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
#  BUILD PATH: straight → arc → straight
# ═══════════════════════════════════════════════════════════════════
dlam_des = 0.02   # waypoint spacing (mm)
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

d1 = (P_CORNER - P_START)[:2];  d1 = d1 / np.linalg.norm(d1)
d2 = (P_END    - P_CORNER)[:2]; d2 = d2 / np.linalg.norm(d2)

arc_p1 = P_CORNER[:2] - ARC_RADIUS * d1
arc_p2 = P_CORNER[:2] + ARC_RADIUS * d2

perp_d1       = np.array([-d1[1], d1[0]])
arc_center_xy = arc_p1 + ARC_RADIUS * perp_d1

theta1 = np.arctan2(arc_p1[1] - arc_center_xy[1], arc_p1[0] - arc_center_xy[0])
theta2 = np.arctan2(arc_p2[1] - arc_center_xy[1], arc_p2[0] - arc_center_xy[0])
if theta2 < theta1:
    theta2 += 2 * np.pi

# Segment 1 — straight
seg1_end = np.append(arc_p1, Z)
n1  = max(int(np.linalg.norm(seg1_end - P_START) / dlam_des) + 1, 2)
seg1 = np.linspace(P_START, seg1_end, n1)

# Arc
arc_len   = ARC_RADIUS * abs(theta2 - theta1)
n_arc     = max(int(arc_len / dlam_des) + 1, 2)
arc_theta = np.linspace(theta1, theta2, n_arc)
arc_pts   = np.column_stack([
    arc_center_xy[0] + ARC_RADIUS * np.cos(arc_theta),
    arc_center_xy[1] + ARC_RADIUS * np.sin(arc_theta),
    np.full(n_arc, Z),
])

# Segment 2 — straight
seg2_start = np.append(arc_p2, Z)
n2   = max(int(np.linalg.norm(P_END - seg2_start) / dlam_des) + 1, 2)
seg2 = np.linspace(seg2_start, P_END, n2)

# Lift segment — pen rises Z_SCAN_LIFT mm at the end of the forming path.
# XY stays fixed at P_END so the pen clears the surface without disturbing
# the channel, then the scanner tail carries it forward.
P_LIFTED    = np.array([P_END[0], P_END[1], Z + Z_SCAN_LIFT])
n_lift      = max(int(Z_SCAN_LIFT / dlam_des) + 1, 2)
lift_seg    = np.linspace(P_END, P_LIFTED, n_lift)

# Scanner tail — continue SCAN_TAIL_MM along the last segment direction
# at the lifted Z so the scanner fully covers the channel end before retract.
P_SCAN_END  = np.array([P_END[0] + d2[0] * SCAN_TAIL_MM,
                        P_END[1] + d2[1] * SCAN_TAIL_MM,
                        Z + Z_SCAN_LIFT])
n_tail      = max(int(SCAN_TAIL_MM / dlam_des) + 1, 2)
tail_seg    = np.linspace(P_LIFTED, P_SCAN_END, n_tail)

# Number of forming waypoints (seg1 + arc + seg2, before lift/tail)
n_forming_waypts = (len(seg1) - 1) + (len(arc_pts) - 1) + (len(seg2) - 1)

# Concatenate — drop duplicate junction points
curve_p_rig = np.vstack([seg1[:-1], arc_pts[:-1], seg2[:-1], lift_seg[:-1], tail_seg])

# Arc mask — True at arc waypoints, False at all others
is_arc_waypoint = np.array(
    [False] * (len(seg1)     - 1) +
    [True]  * (len(arc_pts)  - 1) +
    [False] * (len(seg2)     - 1) +
    [False] * (len(lift_seg) - 1) +
    [False] * len(tail_seg),
    dtype=bool
)

# Transform to robot base frame
curve_p = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R = np.tile(Rbr @ corner_R, (len(curve_p), 1, 1))

print(f"Path: {len(curve_p)} waypoints  "
      f"(seg1={len(seg1)-1}, arc={len(arc_pts)-1}, seg2={len(seg2)-1}, "
      f"lift={len(lift_seg)-1}, tail={len(tail_seg)})")

# ═══════════════════════════════════════════════════════════════════
#  SCANNER OFFSET IN FLANGE FRAME
#
#  The scanner is SCANNER_LAG_MM from the pen in the post-j6 flange
#  frame Y direction.  Derived from calibration: with j6=J6_CAL_OFFSET
#  and corner_R orientation, the scanner is SCANNER_LAG_MM in the +Y
#  rig direction from the pen.  corner_R maps rig-Y to flange-Y, so
#  the offset in the (post-j6) flange frame is simply [0, L, 0].
#
#  As j6 rotates, fwdkin(robot, q).R rotates this vector in world
#  space — the scanner sweeps a circle of radius SCANNER_LAG_MM
#  around the pen tip.
# ═══════════════════════════════════════════════════════════════════
P_SCAN_FLANGE = np.array([0.0, SCANNER_LAG_MM, 0.0])   # mm, post-j6 flange frame

# ═══════════════════════════════════════════════════════════════════
#  CHANNEL TANGENT VECTORS (base frame) — for Algorithm 1
# ═══════════════════════════════════════════════════════════════════
_td    = np.diff(curve_p[:, :2], axis=0)
_td    = np.vstack([_td, _td[-1]])
_norms = np.linalg.norm(_td, axis=1, keepdims=True).clip(min=1e-9)
channel_tangents = (_td / _norms).astype(float)   # (N, 2) unit vectors, base frame

# ═══════════════════════════════════════════════════════════════════
#  INVERSE KINEMATICS + ALGORITHM 1 SCANNER ANGLES
#
#  Algorithm 1 — explicit scanner position tracking:
#  1. Use fwdkin to compute the scanner's ACTUAL world XY position
#     given the current j6 estimate (not the arc-length approximation).
#  2. Find the nearest channel waypoint to the scanner's world position.
#  3. Compute j6 for perpendicularity at that nearest point.
#  4. Iterate (typically 3-6 steps) because j6 determines scanner
#     position, which determines the nearest point, which determines j6.
#
#  This is exact on straights and more accurate than arc-length lag
#  at arc entries/exits where the lagged-position approximation breaks
#  down most.  Damped update (alpha=0.5) ensures convergence for arc
#  radii >= ~25mm; tighter arcs converge more slowly but still settle.
# ═══════════════════════════════════════════════════════════════════
print("IK + Algorithm 1 scanner angles...")
curve_js = []
raw_j6   = []     # raw j6 values (per-waypoint converged, pre-global-unwrap)
q_seed   = np.zeros(6)

# Initial j6 seed from first travel direction
_d0      = curve_p[1, :2] - curve_p[0, :2] if len(curve_p) > 1 else np.array([1.0, 0.0])
_j6_seed = float(np.radians(J6_CAL_OFFSET - 90.0) - np.arctan2(_d0[1], _d0[0]))

for i in range(len(curve_p)):
    if i % 5000 == 0 or i == len(curve_p) - 1:
        print(f"  {i}/{len(curve_p) - 1}")

    # Solve j1-j5 (Pft XY=0 means j6 doesn't affect pen tip position)
    q    = robot6_sphericalwrist_invkin(robot, Transform(curve_R[i], curve_p[i]), q_seed)[0]
    q[5] = _j6_seed   # seed from previous waypoint

    # Algorithm 1: iterate j6 to nearest-channel-point perpendicularity
    for _it in range(10):
        T      = fwdkin(robot, q)
        p_scan = T.p[:2] + (T.R @ P_SCAN_FLANGE)[:2]   # scanner world XY

        # Nearest channel waypoint (base frame)
        dists   = np.linalg.norm(curve_p[:, :2] - p_scan, axis=1)
        n_idx   = int(np.argmin(dists))
        tangent = channel_tangents[n_idx]

        # j6 for perpendicularity at nearest point
        j6_tgt = float(np.radians(J6_CAL_OFFSET - 90.0)
                       - np.arctan2(tangent[1], tangent[0]))

        # Damped update — wrap delta to [-π, π] for stability near ±π
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

# Global unwrap + normalize — same as arc-length lag version.
# Ensures smooth j6 transitions across waypoints and keeps values within ±400°.
raw_j6_arr     = np.array(raw_j6)
scanner_angles = np.unwrap(raw_j6_arr)
shift          = -np.floor((scanner_angles[0] + np.pi) / (2 * np.pi)) * 2 * np.pi
scanner_angles += shift

for i in range(len(curve_js)):
    curve_js[i][5] = scanner_angles[i]

j6_min_deg = np.degrees(scanner_angles.min())
j6_max_deg = np.degrees(scanner_angles.max())
print(f"J6 command range: {j6_min_deg:.1f}° to {j6_max_deg:.1f}°  (robot limit ±400°)")
if abs(scanner_angles.min()) > np.radians(390) or abs(scanner_angles.max()) > np.radians(390):
    raise RuntimeError(f"J6 out of range: [{j6_min_deg:.1f}°, {j6_max_deg:.1f}°]")

tip_errors = [np.linalg.norm(fwdkin(robot, curve_js[i]).p - curve_p[i])
              for i in range(0, len(curve_p), max(1, len(curve_p) // 20))]
print(f"Tip error — max: {max(tip_errors):.4f} mm, mean: {np.mean(tip_errors):.4f} mm")

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def calc_lam_js(qs):
    pts = np.array([fwdkin(robot, q).p for q in qs])
    lam = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return np.insert(lam, 0, 0)


def trajectory_generate(qs, lin_vel, lin_acc):
    """Trapezoidal velocity profile — used for jog / plunge / retract."""
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
    """
    Continuous forming trajectory — single pass, no mid-path stops.

    Uses a forward-backward velocity planning pass to respect per-waypoint
    speed targets (tool_vel on straights, corner_vel on arcs) while
    obeying the acceleration limit.  The robot accelerates from rest at
    the start and decelerates to rest at the end; it never stops between
    straight and arc segments.

    Parameters
    ----------
    qs          : (N, 6) array of joint waypoints
    vel_targets : (N,)   per-waypoint target speed (mm/s)
    lin_acc     : float  acceleration limit (mm/s²)
    """
    pts = np.array([fwdkin(robot, q).p for q in qs])
    ds  = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # segment lengths
    n   = len(qs)

    # Forward pass: max speed achievable at each waypoint from rest
    v_fwd    = np.zeros(n)
    for i in range(1, n):
        v_achievable = np.sqrt(max(v_fwd[i-1]**2 + 2.0 * lin_acc * ds[i-1], 0.0))
        v_fwd[i] = min(vel_targets[i], v_achievable)

    # Backward pass: max speed still allowing deceleration to rest at end
    v_bwd     = np.zeros(n)
    for i in range(n - 2, -1, -1):
        v_achievable = np.sqrt(max(v_bwd[i+1]**2 + 2.0 * lin_acc * ds[i], 0.0))
        v_bwd[i] = min(v_fwd[i], v_achievable)

    v = np.minimum(v_fwd, v_bwd)   # final velocity profile

    # Build time breakpoints via trapezoidal integration
    time_bp = np.zeros(n)
    for i in range(1, n):
        v_avg = 0.5 * (v[i-1] + v[i])
        time_bp[i] = time_bp[i-1] + (ds[i-1] / v_avg if v_avg > 1e-9 else 0.0)

    # Interpolate to 4 ms EGM timesteps
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
    """
    Rate-limit a trajectory array.
    No joint moves faster than max_vel_deg_s between consecutive 4 ms EGM steps.
    Each joint is clamped independently — safety backstop, not a path planner.
    """
    if len(traj_q) == 0:
        return traj_q
    max_delta = np.radians(max_vel_deg_s) * TIMESTEP   # max rad per 4 ms step
    out = traj_q.copy().astype(float)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        out[i] = out[i - 1] + np.clip(delta, -max_delta, max_delta)
    return out


# ═══════════════════════════════════════════════════════════════════
#  TRAJECTORY GENERATION — continuous, per-waypoint speed
# ═══════════════════════════════════════════════════════════════════
vel_targets   = np.where(is_arc_waypoint, corner_vel, tool_vel).astype(float)
fullruntraj_q, time_bp_full = trajectory_generate_continuous(curve_js, vel_targets, tool_acc)
fullruntraj_q               = clamp_joint_vel(fullruntraj_q, MAX_JOINT_VEL_DEG_S)

# Step index where the lift segment begins (end of forming pass)
i_scan_start = int(time_bp_full[n_forming_waypts] / TIMESTEP)

# ═══════════════════════════════════════════════════════════════════
#  HOVER POSITIONS — Z_SAFE on approach, Z_END on departure
# ═══════════════════════════════════════════════════════════════════
def hover_ik(p_rig, j6_target, R_target, q_seed_init):
    """IK for a hover point defined in rig frame. Pen on j6 axis — no Pft correction."""
    p_robot = Rbr @ p_rig + Pbr
    q = robot6_sphericalwrist_invkin(robot, Transform(R_target, p_robot), q_seed_init)[0]
    q[5] = j6_target
    return q

p_hover_start_rig = np.array([curve_p_rig[0][0],  curve_p_rig[0][1],  Z_SAFE])
p_hover_end_rig   = np.array([curve_p_rig[-1][0], curve_p_rig[-1][1], Z_END])

q_hover_start = hover_ik(p_hover_start_rig, curve_js[0][5],  curve_R[0],  curve_js[0])
q_hover_end   = hover_ik(p_hover_end_rig,   curve_js[-1][5], curve_R[-1], curve_js[-1])

# ═══════════════════════════════════════════════════════════════════
#  PRECOMPUTE CARTESIAN POSITIONS FOR FORCE CONTROL
#
#  Avoids calling fwdkin inside the 4 ms EGM loop.  The planned
#  joint angles → Cartesian positions and rotations are cached here.
#  During execution, z_corr offsets p in rig +Z, and IK re-solves
#  per step (closed-form, ~0.1 ms).
# ═══════════════════════════════════════════════════════════════════
rig_z_base = Rbr[:, 2]   # rig +Z direction expressed in robot base frame

print("Precomputing planned Cartesian positions for force control...")
plan_p    = np.array([fwdkin(robot, q).p for q in fullruntraj_q])
plan_R_fc = np.array([fwdkin(robot, q).R for q in fullruntraj_q])
plan_j6   = fullruntraj_q[:, 5]

# ═══════════════════════════════════════════════════════════════════
#  ATI FORCE SENSOR CONNECTION
# ═══════════════════════════════════════════════════════════════════
print("Connecting to ATI force sensor...")
ati_cli     = RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')
wrench_wire = ati_cli.wrench_sensor_value.Connect()
time.sleep(0.3)   # let wire stabilise

def read_fz():
    """Return ATI force Z (raw sensor frame).  Returns 0.0 on read failure."""
    try:
        return float(FORCE_SIGN * wrench_wire.InValue['force']['z'])
    except Exception:
        return 0.0

print("ATI connected.")

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

# Phase 1: pre-rotate j6 to initial scanner angle — arm stays fixed.
# IMPORTANT: Pft XY=0 means j6 rotation produces ~0 Cartesian displacement,
# so calc_lam_js returns ~0 and trajectory_generate produces an empty array.
# Compute step count from j6 angular velocity instead.
#
# Uses a cosine (ease-in/ease-out) profile so j6 starts and ends at v=0.
# A linspace would jump immediately to full speed — felt as a jerk.
# For a cosine profile the peak velocity = pi/2 × (delta / T), so the
# number of steps for a given peak velocity is T/dt = pi×delta/(2×v_max×dt).
q_pre_j6     = q_start.copy()
q_pre_j6[5]  = curve_js[0][5]
j6_delta_rad = abs(q_pre_j6[5] - q_start[5])
j6_vel_rad   = np.radians(MAX_JOINT_VEL_DEG_S[5])   # rad/s peak velocity limit
n_j6_steps   = max(int(j6_delta_rad * np.pi / (2.0 * j6_vel_rad * TIMESTEP)) + 1, 20)
t_norm       = np.linspace(0.0, 1.0, n_j6_steps)
smooth_t     = 0.5 * (1.0 - np.cos(np.pi * t_norm))          # 0→1 ease-in/out
traj_j6      = q_start + smooth_t[:, np.newaxis] * (q_pre_j6 - q_start)
traj_j6      = clamp_joint_vel(traj_j6, MAX_JOINT_VEL_DEG_S)  # safety backstop
print(f"Phase 1: rotating j6 by {np.degrees(j6_delta_rad):.1f} deg "
      f"in {n_j6_steps} steps ({n_j6_steps * TIMESTEP:.1f} s) ...")
for q in traj_j6:
    read_position(egm)
    position_cmd(q, egm)

# Settle hold after j6 pre-rotation — lets the servo reach the commanded
# angle before the arm starts moving.
print("Settling (0.5 s)...")
for _ in range(125):
    read_position(egm)
    position_cmd(q_pre_j6, egm)

# Phase 2: jog arm to hover height above path start — j6 already at target.
traj_approach, _ = trajectory_generate(np.linspace(q_pre_j6, q_hover_start, 500),
                                        lin_vel=jog_vel, lin_acc=jog_acc)
traj_approach = clamp_joint_vel(traj_approach, MAX_JOINT_VEL_DEG_S)
for q in traj_approach:
    read_position(egm)
    position_cmd(q, egm)

# Settle hold after arm jog.
print("Settling at hover (0.5 s)...")
for _ in range(125):
    read_position(egm)
    position_cmd(q_hover_start, egm)

print(f"Hovering at Z={Z_SAFE} mm. Plunging to Z={Z} mm...")

# Phase 3: plunge to forming depth.
traj_plunge, _ = trajectory_generate(np.linspace(q_hover_start, curve_js[0], 200),
                                      lin_vel=jog_vel, lin_acc=jog_acc)
traj_plunge = clamp_joint_vel(traj_plunge, MAX_JOINT_VEL_DEG_S)
for q in traj_plunge:
    read_position(egm)
    position_cmd(q, egm)

print("At start. Holding 0.5 s...")
for _ in range(125):
    read_position(egm)
    position_cmd(curve_js[0], egm)

# Tare ATI at current robot pose (arm weight compensated).
print("Taring ATI force sensor...")
time.sleep(0.1)
ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
time.sleep(0.1)
ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
print(f"ATI tared. Pre-contact fz = {read_fz():.3f} N")

# J6 convergence check — confirms servo reaches commanded angle before stroke.
# If max tracking error at end of run is also small, cable tension is not an issue.
J6_TOL = np.radians(0.5)
j6_converged = False
for _ in range(500):
    q_actual = read_position(egm)
    position_cmd(curve_js[0], egm)
    j6_err = abs(q_actual[5] - curve_js[0][5])
    if j6_err < J6_TOL:
        j6_converged = True
        break

if not j6_converged:
    print(f"WARNING: j6 did not converge — error: {np.degrees(j6_err):.2f}°")
else:
    print(f"J6 converged. Pre-stroke error: {np.degrees(j6_err):.2f}°")

print(f"Starting forming trajectory (force control, F_target={F_TARGET:.1f} N)...")
print(f"  Force active: steps 5 to N-6  |  Scan tail starts at step {i_scan_start}")
z_corr             = 0.0    # mm — live Z correction (+ = lift, - = push deeper)
z_corr_final       = None   # frozen once pen lifts off into scanner tail
contact_established = False  # True once pen has made contact at least once
j6_max_err_rad     = 0.0
q_prev_cmd         = curve_js[0].copy()
max_delta_rad      = np.radians(MAX_JOINT_VEL_DEG_S) * TIMESTEP
N_steps            = len(fullruntraj_q)

for i, q_cmd in enumerate(fullruntraj_q):
    q_actual = read_position(egm)

    # Read contact force (sign-normalised: negative = pressing into surface)
    f_z = read_fz()

    print(f"  step {i:05d}  fz={f_z:.3f} N  z_corr={z_corr:.4f} mm")

    # Force safety — stop if magnitude too large
    if abs(f_z) > FORCE_LIMIT:
        print(f"FORCE LIMIT: {f_z:.2f} N at step {i} — stopping trajectory")
        break

    # Track first contact
    if abs(f_z) > LIFTOFF_THRESHOLD:
        if not contact_established:
            print(f"  Contact at step {i}. fz = {f_z:.3f} N, z_corr = {z_corr:.3f} mm")
        contact_established = True

    # --- Z correction update ---
    # Scan tail: zero z_corr so the robot actually reaches the planned -8 mm
    # lift height, regardless of how much correction accumulated during forming.
    if i >= i_scan_start:
        z_corr = 0.0

    # Force controller active only between step 5 and 5 before the end —
    # avoids noisy start/end data and prevents spurious corrections.
    elif z_corr_final is None and 5 <= i <= N_steps - 6:
        if contact_established and abs(f_z) < LIFTOFF_THRESHOLD:
            z_corr_final = z_corr
            print(f"  Liftoff at step {i}. Freezing z_corr = {z_corr:.3f} mm")
        else:
            f_err  = F_TARGET - f_z
            dz     = np.clip(KF * f_err,
                             -Z_CORR_RATE * TIMESTEP,
                              Z_CORR_RATE * TIMESTEP)
            z_corr = float(np.clip(z_corr + dz, -Z_CORR_MAX, Z_CORR_MAX))
    # else: outside active window or z_corr frozen — hold current value

    # Apply correction: shift planned Cartesian position in rig +Z direction
    p_adj = plan_p[i] + z_corr * rig_z_base
    q_adj = robot6_sphericalwrist_invkin(
                robot, Transform(plan_R_fc[i], p_adj), q_cmd
            )[0]
    q_adj[5] = plan_j6[i]   # preserve Algorithm-1 scanner angle

    # Per-step joint velocity clamp (safety backstop)
    q_adj = q_prev_cmd + np.clip(q_adj - q_prev_cmd, -max_delta_rad, max_delta_rad)
    q_prev_cmd = q_adj.copy()

    position_cmd(q_adj, egm)

    j6_err = abs(q_actual[5] - q_adj[5])
    if j6_err > j6_max_err_rad:
        j6_max_err_rad = j6_err

print(f"Trajectory complete. Max j6 tracking error: {np.degrees(j6_max_err_rad):.2f}°")
print(f"Final z_corr = {z_corr:.3f} mm  (z_corr_final = {z_corr_final})")

# Retract to Z_END.  Start retract from actual last commanded position (q_prev_cmd)
# rather than nominal curve_js[-1] because force control shifts Z.
print(f"Retracting to Z={Z_END} mm...")
traj_retract, _ = trajectory_generate(np.linspace(q_prev_cmd, q_hover_end, 200),
                                       lin_vel=jog_vel, lin_acc=jog_acc)
traj_retract = clamp_joint_vel(traj_retract, MAX_JOINT_VEL_DEG_S)
for q in traj_retract:
    read_position(egm)
    position_cmd(q, egm)

print("Retract complete. Holding 1 s...")
for _ in range(250):
    read_position(egm)
    position_cmd(q_hover_end, egm)

try:
    client.stop_egm()
except Exception as e:
    print("stop_egm warning:", e)

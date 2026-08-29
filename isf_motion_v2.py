import numpy as np
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM

TIMESTEP = 0.004  # 4 ms EGM timestep

# ═══════════════════════════════════════════════════════════════════
#  SERPENTINE PATH — proof of concept (rig frame, mm)
#
#  4 passes within the (50,50)–(100,100) working square:
#
#    x=55                             x=95
#     |                                |
#  y=95  →→→→→→→→→→→→→→→→→→→→→→→→→→→  pass 0  (−X)
#         ╮ left arc (sweeps west)
#  y=85  ←←←←←←←←←←←←←←←←←←←←←←←←←  pass 1  (+X)
#                          right arc ╯ (sweeps east)
#  y=75  →→→→→→→→→→→→→→→→→→→→→→→→→→→  pass 2  (−X)
#         ╮ left arc (sweeps west)
#  y=65  ←←←←←←←←←←←←←←←←←←←←←←←←←  pass 3  (+X)
#
#  To extend to a full serpentine: add more Y values to Y_PASSES.
#  Each left arc  rotates j6 +180°.
#  Each right arc rotates j6 −180°.
#  Alternating arcs cancel, so j6 oscillates and never accumulates.
# ═══════════════════════════════════════════════════════════════════

Z          = -4.0   # forming depth (mm)
Z_SAFE     = 10.0   # transit/hover height above sheet (mm) — used on approach
Z_END      = 15.0   # final retract height after forming (mm)
ARC_RADIUS =  5.0   # U-turn radius (mm) — must equal half the Y spacing between passes

X_LEFT   = 35.0    # left  edge of straight passes (mm) — shifted −20 mm from 55.0
X_RIGHT  = 75.0    # right edge of straight passes (mm) — shifted −20 mm from 95.0

# One Y value per pass, top to bottom. Spacing must equal 2 * ARC_RADIUS.
Y_PASSES = [95.0, 85.0]

# Sanity check: spacing must match arc diameter
spacings = np.diff(Y_PASSES)
if not np.allclose(np.abs(spacings), 2 * ARC_RADIUS):
    raise ValueError(
        f"Y_PASSES spacing ({np.abs(spacings)}) must equal 2 * ARC_RADIUS ({2*ARC_RADIUS})"
    )

# ═══════════════════════════════════════════════════════════════════
#  SPEED
# ═══════════════════════════════════════════════════════════════════
MAX_VEL    = 5.0
tool_vel   = 1.0    # mm/s — straight pass speed
corner_vel = 0.3    # mm/s — arc/corner speed (slower for better geometry)
tool_acc   = 1.0
jog_vel    = 5.0    # mm/s — approach speed (capped to MAX_VEL)
jog_acc    = 1.0    # mm/s² — gentle accel/decel to avoid jerk on approach

tool_vel   = min(tool_vel,   MAX_VEL)
corner_vel = min(corner_vel, MAX_VEL)
jog_vel    = min(jog_vel,    MAX_VEL)

# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION — two-point
#
#  The scanner mounting has ~11.6° of gravity-induced play that
#  manifests asymmetrically between the two travel directions.
#  Measured 2026-06-30 by jogging j6 until the laser line was
#  visually perpendicular to the channel on each pass direction.
#
#    NEG_X: robot traveling in -X (right→left passes)  j6 ≈ +101.67°
#    POS_X: robot traveling in +X (left→right passes)  j6 ≈  -89.93°
#
#  Arc waypoints interpolate smoothly between the two values using
#  a cosine ease so the scanner is never abruptly mis-aimed.
# ═══════════════════════════════════════════════════════════════════
J6_CAL_OFFSET_NEG_X = 11.67   # degrees — perpendicular on -X (right→left) passes
J6_CAL_OFFSET_POS_X =  0.07   # degrees — perpendicular on +X (left→right) passes

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
        for i in range(len(time_bp) - 1):
            if time_bp[i] <= current_time < time_bp[i + 1]:
                seg = i
                break
        frac = (current_time - time_bp[seg]) / (time_bp[seg + 1] - time_bp[seg])
        traj_q.append(frac * curve_js[seg + 1] + (1 - frac) * curve_js[seg])
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

Pft = np.array([-25.45, 7.27, 131.2])   # pen tip offset in flange frame (mm)
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# ---------------------------------------------------------------
# RIG KINEMATICS
# ---------------------------------------------------------------
final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')
z_theta = 2.0549 * np.pi / 180
vz  = final_rig_pose[0:3, 2]
Rz  = rot(vz, z_theta)
Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz

# ---------------------------------------------------------------
# BUILD DENSE SERPENTINE PATH
# ---------------------------------------------------------------
dlam_des = 0.02   # path resolution (mm)
corner_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T

segments    = []
seg_is_arc  = []   # parallel list: True = arc, False = straight

for i, y in enumerate(Y_PASSES):

    # ── Straight pass ──────────────────────────────────────────
    # Even passes go right→left (−X); odd passes go left→right (+X)
    if i % 2 == 0:
        x_start, x_end = X_RIGHT, X_LEFT
    else:
        x_start, x_end = X_LEFT, X_RIGHT

    p_seg_start = np.array([x_start, y, Z])
    p_seg_end   = np.array([x_end,   y, Z])
    n_seg = max(int(np.linalg.norm(p_seg_end - p_seg_start) / dlam_des) + 1, 2)
    segments.append(np.linspace(p_seg_start, p_seg_end, n_seg))
    seg_is_arc.append(False)

    # ── U-turn arc to next pass (skip after last pass) ─────────
    if i >= len(Y_PASSES) - 1:
        continue

    y_next   = Y_PASSES[i + 1]
    y_center = (y + y_next) / 2.0

    if i % 2 == 0:
        # Arrived at X_LEFT traveling −X.  Turn to +X via a CCW arc
        # that sweeps the WEST side (x < X_LEFT).
        #
        #   Entry (θ = 90°):  tangent = (−sin90, cos90) = (−1, 0) = −X ✓
        #   Exit  (θ = 270°): tangent = (−sin270,cos270)= (+1, 0) = +X ✓
        #
        arc_cx = X_LEFT
        theta1, theta2 = np.pi / 2, 3.0 * np.pi / 2   # CCW: θ increases

    else:
        # Arrived at X_RIGHT traveling +X.  Turn to −X via a CW arc
        # that sweeps the EAST side (x > X_RIGHT).
        #
        #   Entry (θ = 90°):  CW tangent = (+sin90, −cos90) = (+1, 0) = +X ✓
        #   Exit  (θ = −90°): CW tangent = (+sin(-90),−cos(-90))=(−1,0) = −X ✓
        #
        arc_cx = X_RIGHT
        theta1, theta2 = np.pi / 2, -np.pi / 2         # CW: θ decreases

    n_arc  = max(int(ARC_RADIUS * abs(theta2 - theta1) / dlam_des) + 1, 2)
    thetas = np.linspace(theta1, theta2, n_arc)
    arc = np.column_stack([
        arc_cx + ARC_RADIUS * np.cos(thetas),
        y_center + ARC_RADIUS * np.sin(thetas),
        np.full(n_arc, Z),
    ])
    segments.append(arc)
    seg_is_arc.append(True)

# Concatenate — drop duplicate junction points between segments.
# Build boolean mask (arc vs straight) and per-waypoint cal_offset in the same pass.
# Arc offsets use cosine ease-in/ease-out so the scanner transitions smoothly.
curve_p_rig_parts = []
is_arc_waypoint   = []
cal_offset_parts  = []
straight_count    = 0

for k, (seg, is_arc) in enumerate(zip(segments, seg_is_arc)):
    pts = seg[:-1] if k < len(segments) - 1 else seg
    n   = len(pts)
    curve_p_rig_parts.append(pts)
    is_arc_waypoint.extend([is_arc] * n)

    if not is_arc:
        # Even straights travel -X; odd straights travel +X.
        offset = J6_CAL_OFFSET_NEG_X if straight_count % 2 == 0 else J6_CAL_OFFSET_POS_X
        cal_offset_parts.append(np.full(n, offset))
        straight_count += 1
    else:
        # Arc: interpolate calibration offset across the turn.
        # Entry = offset of the straight just finished; exit = offset of the next straight.
        if (straight_count - 1) % 2 == 0:          # came from -X straight → next is +X
            o_start, o_end = J6_CAL_OFFSET_NEG_X, J6_CAL_OFFSET_POS_X
        else:                                        # came from +X straight → next is -X
            o_start, o_end = J6_CAL_OFFSET_POS_X, J6_CAL_OFFSET_NEG_X
        t        = np.linspace(0.0, 1.0, n)
        smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))  # cosine ease-in/ease-out: 0 → 1
        cal_offset_parts.append(o_start + smooth_t * (o_end - o_start))

curve_p_rig     = np.vstack(curve_p_rig_parts)
is_arc_waypoint = np.array(is_arc_waypoint)
cal_offset_arr  = np.concatenate(cal_offset_parts)

# Transform to base frame
curve_p = np.array([Rbr @ p + Pbr for p in curve_p_rig])
curve_R = np.tile(Rbr @ corner_R, (len(curve_p), 1, 1))

print(f"Path built: {len(curve_p)} waypoints across {len(Y_PASSES)} passes.")

# ---------------------------------------------------------------
# SCANNER ANGLES — from path tangent at each waypoint
# ---------------------------------------------------------------
travel_dirs   = np.diff(curve_p[:, :2], axis=0)
travel_dirs   = np.vstack([travel_dirs, travel_dirs[-1]])
travel_angles = np.arctan2(travel_dirs[:, 1], travel_dirs[:, 0])

# np.unwrap is REQUIRED here, not just for the cable check.
# arctan2 returns values in (-π, +π]. At the start of each left arc the
# finite-diff travel vector has a tiny -y component, so arctan2 returns ≈ -π
# instead of +π (same physical direction, different branch). Without unwrap
# this creates a 2π jump in scanner_angles → trajectory_generate interpolates
# across it → j6 spins a full 360° extra at every left arc, winding the cable.
# np.unwrap removes 2π jumps (avoids spurious 360° spins at left arc entries).
# The normalize step then shifts the whole array so scanner_angles[0] lands in
# (-π, +π], keeping all j6 values within the ABB IRB1200's ±400° joint limit.
# Without the normalize, unwrapped values reach -439° and the robot clamps.
scanner_angles = np.unwrap(np.radians(cal_offset_arr - 90.0) - travel_angles)
shift = -np.floor((scanner_angles[0] + np.pi) / (2 * np.pi)) * 2 * np.pi
scanner_angles += shift

j6_min_deg = np.degrees(scanner_angles.min())
j6_max_deg = np.degrees(scanner_angles.max())
print(f"J6 command range: {j6_min_deg:.1f}° to {j6_max_deg:.1f}°  (robot limit: ±400°)")
if abs(scanner_angles.min()) > np.radians(390) or abs(scanner_angles.max()) > np.radians(390):
    raise RuntimeError(f"J6 out of range: [{j6_min_deg:.1f}°, {j6_max_deg:.1f}°]")

# Cable safety check
j6_unwrapped = scanner_angles   # already unwrapped above
j6_range_deg = np.degrees(j6_unwrapped.max() - j6_unwrapped.min())
j6_net_deg   = np.degrees(j6_unwrapped[-1] - j6_unwrapped[0])
print(f"J6 sweep range : {j6_range_deg:.1f}°  (cable safe if < 300°)")
print(f"J6 net rotation: {j6_net_deg:.1f}°")
if j6_range_deg > 300.0:
    raise RuntimeError("J6 exceeds cable safety limit — check path geometry.")

# ---------------------------------------------------------------
# INVERSE KINEMATICS — iterative j6 compensation
# ---------------------------------------------------------------
terminate_threshold = 0.001
IK_MAX_ITER         = 5

print("IK (iterative j6 compensation)...")
curve_js = []
q_seed   = np.zeros(6)

for i in range(len(curve_p)):
    if i % 5000 == 0 or i == len(curve_p) - 1:
        print(f"  {i}/{len(curve_p) - 1}")

    j6_des   = scanner_angles[i]
    p_target = curve_p[i].copy()

    q = robot6_sphericalwrist_invkin(robot, Transform(curve_R[i], p_target), q_seed)[0]
    for _ in range(IK_MAX_ITER):
        q[5]     = j6_des
        p_actual = fwdkin(robot, q).p
        error    = curve_p[i] - p_actual
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
# HOVER POSITIONS — Z_SAFE on approach, Z_END on departure
# Same XY as path start/end, same j6 as first/last forming waypoint.
# ---------------------------------------------------------------
def hover_ik(p_rig, j6_target, R_target, q_seed_init):
    """Iterative IK for a hover point defined in rig frame."""
    p_robot = Rbr @ p_rig + Pbr
    q = robot6_sphericalwrist_invkin(robot, Transform(R_target, p_robot), q_seed_init)[0]
    for _ in range(IK_MAX_ITER):
        q[5] = j6_target
        p_actual = fwdkin(robot, q).p
        error = p_robot - p_actual
        if np.linalg.norm(error) < terminate_threshold:
            break
        q = robot6_sphericalwrist_invkin(robot, Transform(R_target, p_robot + error), q)[0]
    q[5] = j6_target
    return q

p_hover_start_rig = np.array([curve_p_rig[0][0],  curve_p_rig[0][1],  Z_SAFE])
p_hover_end_rig   = np.array([curve_p_rig[-1][0], curve_p_rig[-1][1], Z_END])

q_hover_start = hover_ik(p_hover_start_rig, curve_js[0][5],  curve_R[0],  curve_js[0])
q_hover_end   = hover_ik(p_hover_end_rig,   curve_js[-1][5], curve_R[-1], curve_js[-1])
print(f"Hover start: Z={Z_SAFE} mm above path start  |  Retract end: Z={Z_END} mm above path end")

# ---------------------------------------------------------------
# TRAJECTORY GENERATION — straight passes at tool_vel, arcs at corner_vel
# ---------------------------------------------------------------
traj_chunks = []
i = 0
while i < len(curve_js):
    cur_type = is_arc_waypoint[i]
    j = i + 1
    while j < len(curve_js) and is_arc_waypoint[j] == cur_type:
        j += 1
    chunk = curve_js[i:j]
    vel   = corner_vel if cur_type else tool_vel
    label = "arc" if cur_type else "straight"
    print(f"  Trajectory [{label}]: {len(chunk)} pts @ {vel} mm/s")
    if len(chunk) > 1:
        chunk_traj, _ = trajectory_generate(chunk, robot, lin_vel=vel, lin_acc=tool_acc)
        traj_chunks.append(chunk_traj)
    i = j

fullruntraj_q = np.vstack(traj_chunks)
print(f"Trajectory: {len(fullruntraj_q)} total steps")

# ---------------------------------------------------------------
# EGM SETUP & EXECUTION
# ---------------------------------------------------------------
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

q_start    = read_position(egm)

# Phase 1: rotate j6 only to the initial scanner angle while holding arm fixed.
# This avoids a large j6 swing being mixed into the arm jog, which causes jerk.
q_pre_j6 = q_start.copy()
q_pre_j6[5] = curve_js[0][5]
q_approach_j6 = np.linspace(q_start, q_pre_j6, num=500)
traj_j6, _ = trajectory_generate(q_approach_j6, robot, lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_j6:
    read_position(egm)
    position_cmd(q, egm)

# Phase 2: jog arm to hover position (Z_SAFE above start) — j6 already at target angle.
q_approach = np.linspace(q_pre_j6, q_hover_start, num=500)
traj_approach, _ = trajectory_generate(
    q_approach, robot, lin_vel=jog_vel, lin_acc=jog_acc
)
for q in traj_approach:
    read_position(egm)
    position_cmd(q, egm)

print(f"Hovering at Z={Z_SAFE} mm. Plunging to forming depth Z={Z} mm...")

# Plunge: descend from hover height to forming depth.
q_plunge = np.linspace(q_hover_start, curve_js[0], num=200)
traj_plunge, _ = trajectory_generate(q_plunge, robot, lin_vel=jog_vel, lin_acc=jog_acc)
for q in traj_plunge:
    read_position(egm)
    position_cmd(q, egm)

print("At start. Holding 0.5 s...")
for _ in range(125):
    read_position(egm)
    position_cmd(curve_js[0], egm)

# J6 convergence check — verify the servo reached the commanded angle
# before starting the stroke. If cable torque is causing position error,
# the actual j6 will consistently deviate here and we'll see it in the warning.
J6_TOL = np.radians(0.5)   # 0.5° tolerance
j6_converged = False
for _ in range(500):        # 2 s max at 250 Hz
    q_actual = read_position(egm)
    position_cmd(curve_js[0], egm)
    j6_err = abs(q_actual[5] - curve_js[0][5])
    if j6_err < J6_TOL:
        j6_converged = True
        break

if not j6_converged:
    print(f"WARNING: j6 did not converge — final error: {np.degrees(j6_err):.2f}°")
else:
    print(f"J6 converged. Pre-stroke error: {np.degrees(j6_err):.2f}°")

print("Starting forming trajectory...")
j6_max_err_rad = 0.0
for q_cmd in fullruntraj_q:
    q_actual = read_position(egm)
    position_cmd(q_cmd, egm)
    j6_err = abs(q_actual[5] - q_cmd[5])
    if j6_err > j6_max_err_rad:
        j6_max_err_rad = j6_err

print(f"Trajectory complete. Max j6 tracking error during stroke: {np.degrees(j6_max_err_rad):.2f}°")

# Retract: lift from end of forming path to Z_END.
print(f"Retracting to Z={Z_END} mm...")
q_retract = np.linspace(curve_js[-1], q_hover_end, num=200)
traj_retract, _ = trajectory_generate(q_retract, robot, lin_vel=jog_vel, lin_acc=jog_acc)
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

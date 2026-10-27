# DO NOT RUN OPTION 2 YET IT IS STILL A WIP

import numpy as np
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import robotraconteur as rr_rox
from matplotlib import pyplot as plt
import time
import pandas as pd

import abb_motion_program_exec as abb
from abb_robot_client.egm import EGM
from RobotRaconteur.Client import *
import open3d as o3d


# Helper functions
def calc_lam_js(curve_js, robot):
    curve_p = [fwdkin(robot, q).p for q in curve_js]
    lam = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
    return np.insert(lam, 0, 0)

def read_position(egm):
    for i in range(0, 20):
        res, state = egm.receive_from_robot(timeout=0.1)
        if not res:
            client.stop_egm()
            mm = abb.egm_minmax(-1e-3, 1e-3)
            mp = abb.MotionProgram(egm_config=egm_config)
            mp.EGMRunJoint(10, 0.05, 0.05)
            client.stop_motion_program()
            lognum = client.execute_motion_program(mp, wait=False)
            print("Communication Lost, Reconnecting")
        else:
            break
    return np.radians(state.joint_angles)

def position_cmd(q, egm):
    egm.send_to_robot(np.degrees(q))

def trajectory_generate(curve_js, robot, lin_vel, lin_acc, timestep=0.004):
    lam = calc_lam_js(curve_js, robot)
    
    if len(lam) > 2 and lin_acc > 0:
        time_bp = np.zeros_like(lam)
        acc = lin_acc
        vel = 0
        for i in range(0, len(lam)):
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
            else:
                time_bp_half.append(np.sqrt(2 * (lam[-1] - lam[i]) / acc))
                vel = acc * time_bp_half[-1]
        
        time_bp_half = np.array(time_bp_half)[::-1]
        time_bp_half = time_bp_half * -1 + time_bp_half[0]
        time_bp[-len(time_bp_half):] = time_bp[-len(time_bp_half)-1] + time_bp_half \
            + (lam[-len(time_bp_half)] - lam[-len(time_bp_half)-1]) / lin_vel
    else:
        time_bp = lam / lin_vel

    num_steps = int(time_bp[-1] / timestep)
    traj_q = []

    for step in range(num_steps):
        current_time = step * timestep
        seg = 0
        for i in range(len(time_bp) - 1):
            if current_time >= time_bp[i] and current_time < time_bp[i+1]:
                seg = i
                break
        frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
        q_des = frac * curve_js[seg+1] + (1 - frac) * curve_js[seg]
        traj_q.append(q_des)

    return np.array(traj_q), np.array(time_bp)

def generate_arc_waypoints(radius, origin, angle_deg, height, num_pts):
    half_angle = np.radians(angle_deg) / 2.0
    angles = np.linspace(-half_angle, half_angle, num_pts)
    waypoints = []
    for theta in angles:
        x = origin[0] + radius * np.cos(theta)  # Apex bulges out along X
        y = origin[1] + radius * np.sin(theta)  # Spans left-to-right along Y
        z = origin[2] + height
        waypoints.append([x, y, z])
    return np.array(waypoints)

def load_waypoints_from_excel(filename, sheet_name, drawing_origin):
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    waypoints = df.iloc[:, 0:3].to_numpy().astype(float) + drawing_origin
    np.savetxt("gcode_coordinates_new.csv", waypoints, delimiter=',')
    return waypoints


# Robot setup
mti_client = RRN.ConnectService("rr+tcp://localhost:60830/?service=MTI2D")
mti_client.setExposureTime("25")
# mti_client = None

with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

Pft = np.array([-55.45, 0, 131.2])
tool_T = Transform(np.eye(3), Pft)
robot.R_tool = tool_T.R
robot.p_tool = tool_T.p

# Configuration and scan mode
PATH_MODE = 'arc'       # Options: 'arc', 'excel'
SCAN_MODE =            # 1 = Option 1 (Straight line sweep, fixed orientation)
                        # 2 = Option 2 (Follow path geometry, normal-to-path orientation)
dlam_des = 0.05         # Point density (mm)

if PATH_MODE == 'arc':
    corner_p_wp = generate_arc_waypoints(35.0, np.array([10.0, 10.0, 0.0]), 180.0, 20.0, 60)
elif PATH_MODE == 'excel':
    corner_p_wp = load_waypoints_from_excel("bpp_triple_serpentine.xlsx", "line1", np.array([10.0, 10.0, 0.0]))
else:
    print("Invalid path mode")
    exit()

# If SCAN_MODE == 1, override waypoints with a straight sweep across the bounding box
if SCAN_MODE == 1:
    print("Scan Mode 1 Selected: Generating straight linear sweep along Y-axis...")
    x_mid = np.mean(corner_p_wp[:, 0])
    y_min, y_max = np.min(corner_p_wp[:, 1]), np.max(corner_p_wp[:, 1])
    z_max = np.max(corner_p_wp[:, 2])
    
    # Straight line sweep from min Y to max Y at a constant X mid-point
    corner_p_wp = np.array([
        [x_mid, y_min, z_max],
        [x_mid, y_max, z_max]
    ])
    
# Densify Path
curve_p = []
for i in range(corner_p_wp.shape[0] - 1):
    dist = np.linalg.norm(corner_p_wp[i+1] - corner_p_wp[i])
    n_pts = int(dist / dlam_des) + 1
    this_seg_p = np.linspace(corner_p_wp[i], corner_p_wp[i+1], n_pts)
    curve_p.extend(this_seg_p[:-1])
curve_p.append(corner_p_wp[-1])
curve_p = np.array(curve_p)

# Calculate Orientation Matrix based on Scan Mode
curve_R = []
if SCAN_MODE == 1:
    # Option 1: Fixed orientation 
    fixed_R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T
    curve_R = np.tile(fixed_R, (len(curve_p), 1, 1))
else:
    # Option 2: Dynamic Normal-to-Path orientation along tangents
    for i in range(len(curve_p)):
        if i < len(curve_p) - 1:
            tangent = curve_p[i+1] - curve_p[i]
        else:
            tangent = curve_p[i] - curve_p[i-1]
        
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent /= norm_t
        
        z_tool = np.array([0.0, 0.0, -1.0])
        y_tool = np.cross(z_tool, tangent)
        norm_y = np.linalg.norm(y_tool)
        if norm_y < 1e-6:
            y_tool = np.array([0.0, 1.0, 0.0])
        else:
            y_tool /= norm_y
            
        x_tool = np.cross(y_tool, z_tool)
        R_local = np.column_stack((x_tool, y_tool, z_tool))
        curve_R.append(R_local)
    curve_R = np.array(curve_R)


# Rig kinematics
final_rig_pose = np.loadtxt("rig_pose.csv", delimiter=',')

z_theta = 2.0549 * np.pi / 180  # Radians Z adjustment
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3, -1]
Rbr = final_rig_pose[0:3, 0:3] @ Rz

for i in range(curve_p.shape[0]):
    curve_p[i] = Rbr @ curve_p[i] + Pbr

new_curve_R = [Rbr @ R for R in curve_R]

q_init = robot6_sphericalwrist_invkin(robot, Transform(new_curve_R[0], curve_p[0]), np.zeros(6))[0]

curve_js = [q_init]
terminate_threshold = 0.001
st = time.perf_counter()

for i in range(1, len(curve_p)):
    this_q = robot6_sphericalwrist_invkin(robot, Transform(new_curve_R[i], curve_p[i]), curve_js[-1])[0]
    flange_T = fwdkin(robot, this_q)
    vd = curve_p[i] - flange_T.p
    assert np.linalg.norm(vd) < terminate_threshold, f"Inverse Kinematics error at step {i}"
    curve_js.append(this_q)

lam_planned = np.cumsum(np.linalg.norm(np.diff(curve_p, axis=0), axis=1))
lam_planned = np.insert(lam_planned, 0, 0)
print(f"Planned {len(curve_js)} points, path length {lam_planned[-1]:.1f} mm, IK time: {time.perf_counter() - st:.1f} s")


# Execution 
TIMESTEP = 0.004
tool_vel = 1   # mm/s
tool_acc = 1   # mm/s^2

fullruntraj_q, fullruntime_bp = trajectory_generate(curve_js, robot, lin_vel=tool_vel, lin_acc=tool_acc, timestep=TIMESTEP)
print(f"Trajectory ready: {len(fullruntraj_q)} steps ({len(fullruntraj_q) * TIMESTEP:.1f} s)")

np.savetxt("fullruntraj_q.csv", fullruntraj_q, delimiter=',')
np.savetxt("fullruntime_bp.csv", fullruntime_bp, delimiter=',')
np.savetxt("curve_js.csv", np.array(curve_js), delimiter=',')
np.savetxt("lam_planned.csv", lam_planned, delimiter=',')

curve_js_exe = []

mm = abb.egm_minmax(-1e-3, 1e-3)
egm_config = abb.EGMJointTargetConfig(mm, mm, mm, mm, mm, mm, 1000, 1000)
mp = abb.MotionProgram(egm_config=egm_config)
mp.EGMRunJoint(10, 0.05, 0.05)

client = abb.MotionProgramExecClient(base_url="http://192.168.60.101:80")
# client = abb.MotionProgramExecClient(base_url="http://127.0.0.1:80")
lognum = client.execute_motion_program(mp, wait=False)

egm = EGM()
print("Robot starting motion via EGM...")

q_start = read_position(egm)
q_all = np.linspace(q_start, curve_js[0], num=100)
traj_q, _ = trajectory_generate(q_all, robot, lin_vel=tool_vel, lin_acc=tool_acc, timestep=TIMESTEP)

for i in range(len(traj_q)):
    read_q = read_position(egm)
    position_cmd(traj_q[i], egm)

print("Robot positioned at start point.")

EE_pos = []
mti_recording = []

for i in range(len(fullruntraj_q)):
    read_q = read_position(egm)
    curve_js_exe.append(read_q)

    current_EEpos = fwdkin(robot, read_q).p
    coordinates_in_rig_frame = Rbr.T @ (current_EEpos - Pbr)

    mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data, mti_client.lineProfile.Z_data])))

    EE_pos.append(coordinates_in_rig_frame)
    position_cmd(fullruntraj_q[i], egm)

client.stop_egm()

curve_js_exe = np.array(curve_js_exe)
lam_exe = calc_lam_js(curve_js_exe, robot)
np.savetxt("EE_pos.csv", EE_pos, delimiter=',')


# 3D point cloud
q_out_exe = np.array(curve_js_exe)[:, :6]
pcd_all = o3d.geometry.PointCloud()

for i in range(len(mti_recording)):
    scanner_T = fwdkin(robot, q_out_exe[i])
    
    scan_points = np.insert(mti_recording[i], 1, np.zeros(len(mti_recording[i][0])), axis=0)
    scan_points[0] = scan_points[0] * -1  # Invert X axis for tool mounting frame

    scan_points = scan_points.T
    scan_points = np.transpose(np.matmul(scanner_T.R, np.transpose(scan_points))) + scanner_T.p

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan_points)
    pcd_all += pcd

o3d.visualization.draw_geometries([pcd_all])

curve_js = np.array(curve_js)
plt.figure()
for i in range(6):
    plt.plot(lam_planned, curve_js[:, i], label=f'Joint {i+1} planned')
    plt.plot(lam_exe, curve_js_exe[:, i], label=f'Joint {i+1} executed', linestyle='--')
plt.legend()
plt.xlabel('Path length (mm)')
plt.ylabel('Joint angle (rad)')
plt.title('Planned vs Executed Joint Trajectory')
plt.show()

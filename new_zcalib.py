import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from RobotRaconteur.Client import *
from general_robotics_toolbox import *
from calibration import *

sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *
sys.path.append('robot_motion')
from RobotMotionController import *


# Load in a JSON file, create new one for every new tool
CONFIG_FILE = "pen_mount_side.json"
print(f"Loading configuration from {CONFIG_FILE}...")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Extract tool geometry, dimensions, and tool-specific corners
Pft = np.array(config["Pft"])
w = config["rig_dimensions"]["w"]
h = config["rig_dimensions"]["h"]
l = config["rig_dimensions"]["l"]
thickness = config["rig_dimensions"]["thickness"]
SURFACE_CORRECTION = np.array(config["surface_correction"])

# Load the specific corners for this mount from JSON
c1 = np.array(config["corners"]["c1"])
c2 = np.array(config["corners"]["c2"])
c3 = np.array(config["corners"]["c3"])
c4 = np.array(config["corners"]["c4"])

print(f"Active Tool Profile: {config.get('tool_name')}")
print(f"Loaded Pft: {Pft}")

# Network / Connection Parameters
abb_robot_ip = '192.168.60.101'
ATI_SERVICE_URL = 'rr+tcp://localhost:59823?service=ati_sensor'
TIMESTEP = 0.004

controller_params = {
    "force_ctrl_damping": 20.0,
    "force_epsilon": 1.5,
    "moveL_speed_lin": 6.0,
    "moveL_acc_lin": 7.2,
    "moveL_speed_ang": np.radians(10),
    "trapzoid_slope": 1.0,
    "load_speed": 10.0,
    "unload_speed": 1.0,
    'settling_time': 0.2,
    "lookahead_time": 0.132,
    "jogging_speed": 25,
    "jogging_acc": 10,
    'force_filter_alpha': 0.9
}

# Initialize Sensor & Tool Definition
H_pentip2ati = np.loadtxt("probetip2ati.csv", delimiter=',') # EDIT
H_ati2pentip = np.linalg.inv(H_pentip2ati)
ad_ati2pentip = adjoint_map(Transform(H_ati2pentip[:3, :3], H_ati2pentip[:3, -1]))
ad_ati2pentip_T = ad_ati2pentip.T

RR_ati_cli = RRN.ConnectService(ATI_SERVICE_URL)

tool_T = Htransform(np.eye(3), Pft)
np.savetxt("rig_pen.csv", tool_T, delimiter=',')

robot = robot_obj('ABB_1200_5_90', "ABB_1200_5_90_robot_default_config.yml", tool_file_path="rig_pen.csv")
q_seed = np.zeros(6)

# Quaternion from teach pendant (base to flange orientation)
RoF = q2R(np.array([.01916, .00040, -.99982, -.00112]))  # Possibly edit???

# Apply tool flange offset to corner touch points
c1 = c1 + RoF @ Pft
c2 = c2 + RoF @ Pft
c3 = c3 + RoF @ Pft
c4 = c4 + RoF @ Pft

# Save raw Pbr, Rbr to .csv file for z-axis calibration
calibrate(c1, c2, c3, c4)
rig_pose = np.loadtxt("rig_pose_raw.csv", delimiter=',')

# Apply offset adjustments to pose matrix
rig_pose[:3, -1] = rig_pose[:3, -1] + rig_pose[:3, :3] @ np.array([h / 2, w / 2, 0])
R_pencil = rig_pose[:3, :3] @ Ry(np.pi)

mctrl = MotionController(robot, rig_pose, H_pentip2ati, controller_params, TIMESTEP, FORCE_PROTECTION=6, RR_ati_cli=RR_ati_cli, abb_robot_ip=abb_robot_ip)

corners_offset = np.array([[h/2 + l/2, 0, 0], [0, w/2 + l/2, 0], [-h/2 - l/2, 0, 0], [0, -w/2 - l/2, 0]])
corners = np.dot(rig_pose[:3, :3], corners_offset.T).T + np.tile(rig_pose[:3, -1], (4, 1))

# Loop four corners to get precise position based on force feedback
corners_adjusted = []
f_d = -1.0  # 10N push down target reference

mctrl.start_egm()
try:
    for corner in corners:
        z_offset = np.array([0, 0, 0])
        corner_top = corner + 20 * rig_pose[:3, -2] + z_offset
        corner_top_safe = corner + 80 * rig_pose[:3, -2] + z_offset
        
        print(f"Moving to corner top: {corner_top}")
        input("Press Enter to move to corner...")
        
        q_corner_top = robot.inv(corner_top, R_pencil, q_seed)[0]
        q_corner_top_safe = robot.inv(corner_top_safe, R_pencil, q_seed)[0]
        
        mctrl.jog_joint_position_cmd(q_corner_top_safe, v=controller_params["jogging_speed"])
        input("Press Enter to push down...")
        
        mctrl.jog_joint_position_cmd(q_corner_top, v=controller_params["jogging_speed"])
        time.sleep(0.1)
        
        # Clear sensor bias (tare)
        mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
        time.sleep(0.2)
        mctrl.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
        time.sleep(1.0)
        
        print("Current force reading:", mctrl.ft_reading)
        input("Press Enter to start force probing...")
        
        for _ in range(100):
            q_cur = mctrl.read_position()

        ft_record = mctrl.force_load_z(f_d)
        
        for _ in range(100):
            q_cur = mctrl.read_position()
            
        corners_adjusted.append(robot.fwd(q_cur).p)
        print("Adjusted corner recorded:", corners_adjusted[-1])

        mctrl.jog_joint_position_cmd(q_corner_top_safe, v=controller_params["jogging_speed"])

except (Exception, KeyboardInterrupt) as e:
    print("Error during corner calibration:", e)
finally:
    try:
        mctrl.stop_egm()
    except Exception as e:
        print("stop_egm warning:", e)

# UPDATE RIG POSE BASED ON ADJUSTED CORNERS
p_all = np.array(corners_adjusted)
np.savetxt("corners_adjusted.csv", p_all, delimiter=',')

center = np.mean(p_all, axis=0)
pca = PCA()
pca.fit(p_all)
R_temp = pca.components_.T

if R_temp[:, 0] @ center < 0:
    R_temp[:, 0] = -R_temp[:, 0]
if R_temp[:, -1] @ R_pencil[:, -1] > 0:
    R_temp[:, -1] = -R_temp[:, -1]

R_temp[:, 1] = np.cross(R_temp[:, 2], R_temp[:, 0])

# Re-center and apply surface corrections loaded from JSON config
center = center + R_temp @ np.array([-h / 2, -w / 2, -thickness])
center = center + R_temp @ SURFACE_CORRECTION

print("New rig Orientation matrix R:\n", R_temp)
print("New rig center position:\n", center)

np.savetxt("rig_pose.csv", H_from_RT(R_temp, center), delimiter=',')
print("Calibration complete. rig_pose.csv successfully updated using JSON config.")

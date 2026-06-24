import numpy as np
from copy import deepcopy
from qpsolvers import solve_qp
from general_robotics_toolbox import *
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox
from matplotlib import pyplot as plt

from traj_gen import get_trajectory

# Define the robot
with open('ABB_1200_5_90_robot_default_config.yml', 'r') as file:
    robot = rr_rox.load_robot_info_yaml_to_robot(file)

print(fwdkin(robot, np.radians([0,30,30,0,20,0])))

# Define a curve in 3D space with postion and orientation in the rbots base frame
# sample a parabolic curve in 3D space
final_l = 300 # 300 mm curve
x_sample_range = np.linspace(0, final_l, int(final_l/0.1)+1) # sample every 0.1 mm
curve_p, curve_n = get_trajectory(x_sample_range)

# fix orientation R
curve_R = []
for i in range(len(curve_p)):
    if len(curve_R)==0:
        curve_R.append(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T)
    else:
        y_axis = curve_p[i]-curve_p[i-1]
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = curve_n[i]
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        curve_R.append(np.vstack((x_axis, y_axis, z_axis)).T)

# curve position translation
curve_p = curve_p + np.array([400, 0, 325])

# visualize the 3D curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_p[:,0], curve_p[:,1], curve_p[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#### Now the curve is defined
# curve_p: an array. Each element is a 3D position vector
# curve_R: an array. Each element is a 3x3 rotation matrix
# curve_n: an array. Each element is a 3D normal vector where the z_axis of the end effector should align with
# We are then iterating through the curve and calculate the joint trajectory

## Case 1: 6 dof constraints (position and orientation)
# get the initial joint angles
q_init = robot6_sphericalwrist_invkin(robot,Transform(curve_R[0], curve_p[0]),np.zeros(6))[0]

angle_weight = 1
terminate_threshold = 0.001
alpha=1 # we can use a line search to find the best alpha. Here we use a fixed alpha.

# get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
curve_js = [q_init]
for i in range(1,len(curve_p)):

    q_all = [curve_js[-1]]
    while True:
        flange_T = fwdkin(robot, q_all[-1])
        vd = curve_p[i]-flange_T.p
        Rd = curve_R[i]@flange_T.R.T
        
        k,theta = R2rot(Rd)
        if theta<1e-10:
            omega_d = np.zeros(3)
        else:
            omega_d=np.sin(theta/2)*k # You can use any s error function here
        nu_d = np.concatenate((omega_d*angle_weight,vd))

        # print("Iteration:", len(q_all), "Error:", np.linalg.norm(vd), 'Theta:', np.degrees(theta))
        if np.linalg.norm(nu_d) < terminate_threshold:
            break

        J = robotjacobian(robot, q_all[-1])

        H = J.T @ J
        H = (H + H.T) / 2
        f = -J.T @ nu_d
        #solving quadractic programming here
        qdot = solve_qp(H, f, lb=robot.joint_lower_limit+np.ones(6)*0.0001-q_all[-1], ub=robot.joint_upper_limit-np.ones(6)*0.0001+q_all[-1], solver="quadprog")

        q_all.append(q_all[-1] + alpha*qdot)
    curve_js.append(q_all[-1])

# visualize the joint trajectory
curve_js = np.array(curve_js)
fig = plt.figure()
for i in range(6):
    plt.plot(curve_js[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle (rad)')
plt.title('Joint trajectory')
plt.show()

# visualize qdot
qdot = np.diff(curve_js, axis=0)
fig = plt.figure()
for i in range(6):
    plt.plot(qdot[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle velocity (rad/s)')
plt.title('Joint velocity')
plt.show()

# visualize qddot
qddot = np.diff(qdot, axis=0)
fig = plt.figure()
for i in range(6):
    plt.plot(qddot[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle acceleration (rad/s^2)')
plt.title('Joint acceleration')
plt.show()

## Case 2: 5 dof constraints (position + 2 dof normal vector)
# get the initial joint angles
q_init = robot6_sphericalwrist_invkin(robot,Transform(curve_R[0], curve_p[0]),np.zeros(6))[0]

terminate_threshold = 0.001
alpha=1 # we can use a line search to find the best alpha. Here we use a fixed alpha.

# get joint trajectory using 6 dof constraints (curve_p and curve_R, i.e. position and orientation)
curve_js = [q_init]
for i in range(1,len(curve_p)):

    q_all = [curve_js[-1]]
    while True:
        flange_T = fwdkin(robot, q_all[-1])
        vd = curve_p[i]-flange_T.p
        ezd = (curve_n[i]-flange_T.R[:,-1])
        
        nu_d = np.concatenate((ezd*angle_weight,vd))

        if np.linalg.norm(nu_d) < terminate_threshold:
            break

        J = robotjacobian(robot, q_all[-1])
        hatR = -np.eye(6)
        hatR[:3,:3] = hat(flange_T.R[:,-1])
        hatR[3:,:3] = hat(flange_T.R[:,-1])
        J_mod = -hatR@J

        H = J_mod.T @ J_mod + np.ones(6)*0.0001 # add a small value to make the matrix positive definite
        H = (H + H.T) / 2
        f = -J_mod.T @ nu_d
        qdot = solve_qp(H, f, lb=robot.joint_lower_limit+np.ones(6)*0.0001-q_all[-1], ub=robot.joint_upper_limit-np.ones(6)*0.0001+q_all[-1], solver="quadprog")

        q_all.append(q_all[-1] + alpha*qdot)
    curve_js.append(q_all[-1])

# visualize the joint trajectory
curve_js = np.array(curve_js)
fig = plt.figure()
for i in range(6):
    plt.plot(curve_js[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle (rad)')
plt.title('Joint trajectory')
plt.show()

# visualize qdot
qdot = np.diff(curve_js, axis=0)
fig = plt.figure()
for i in range(6):
    plt.plot(qdot[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle velocity (rad/s)')
plt.title('Joint velocity')
plt.show()

# visualize qddot
qddot = np.diff(qdot, axis=0)
fig = plt.figure()
for i in range(6):
    plt.plot(qddot[:,i], label=f'Joint {i+1}')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Joint angle acceleration (rad/s^2)')
plt.title('Joint acceleration')
plt.show()
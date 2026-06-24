
import numpy as np
from general_robotics_toolbox import *

#############################################Rig Kinematics###################################################
final_rig_pose=np.loadtxt("rig_pose.csv",delimiter=',')

#Angle adjustment about z axis, makes parallel with rig axis
z_theta = 2.0549*np.pi/180     #Radians
#current z axis
vz = final_rig_pose[0:3, 2]
Rz = rot(vz, z_theta)

Pbr = final_rig_pose[0:3,-1]
Rbr = final_rig_pose[0:3, 0:3]@Rz

# Run it on the robot
qbr = R2q(Rbr)
################################################################################################

#Create 7 lines
x_points = [10,20,30,40,50,60,70]

lines = []
for startpoints in x_points:
	segments = []
	buffer1 = np.array([startpoints, 10, 10])
	line_start = np.array([startpoints, 10, 0])
	line_end = np.array([startpoints, 90, 0])
	buffer2 = np.array([startpoints, 90, 10])
	segments.append(buffer1)
	segments.append(line_start)
	segments.append(line_end)
	segments.append(buffer2)
	seg_array = np.array(segments)
	lines.append(seg_array)

for line in lines:
	print(line)
	
lines = np.array(lines)
#Transformation to rig
for i in range(0,lines.shape[0]):
	for j in range(0,4):
		lines[i][j] = Rbr@lines[i][j] + Pbr

lines = lines.tolist()
forces = [.25, .5, .75, 1.0, 1.25, 1.5, 1.75]
print(lines[0][0])
print(len(lines))

for line in lines:
	print(line)
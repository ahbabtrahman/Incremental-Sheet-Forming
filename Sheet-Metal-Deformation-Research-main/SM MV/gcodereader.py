import numpy as np

gcode=open("countourpath.txt", 'r')
lines = gcode.readlines()
newlines = []
current_X = "X0.0 "
current_Y = " Y0.0"
current_Z = " Z0.0"
for chunk in lines:
    if 'G' in chunk or 'F' in chunk:
        chunk = chunk.replace('G0 ', '')
        chunk = chunk.replace('G1 ', '')
        chunk = chunk.replace(' F1000', '')
        chunk = chunk.replace(' F333.3', '')
    #Check if line is missing X, Y, Z coordinates information
    #Only Z coord
    if 'X' not in chunk and 'Y' not in chunk and 'Z' in chunk:
        chunk = current_X + current_Y + chunk
    #Only X Coord
    if 'Y' not in chunk and 'Z' not in chunk and 'X' in chunk:
        chunk = chunk + current_Y + current_Z
    #Only Y coord
    if 'Z' not in chunk and 'X' not in chunk and 'Y' in chunk:
        chunk = current_X + chunk + current_Z
    #Only X and Y
    if 'Z' not in chunk and 'X' in chunk and 'Y' in chunk:
        chunk = chunk + current_Z
    #Only Y and Z
    if 'X' not in chunk and 'Y' in chunk and 'Z' in chunk:
        chunk = current_X + chunk
    #Only X and Z
    if 'Y' not in chunk and 'X' in chunk and 'Z' in chunk:
        chunk = current_X + chunk + current_Z
    
    #Update current coordinates
    coords = chunk.split()
    print(coords)
    current_X = coords[0] + " "
    current_Y = " " + coords[1] + " "
    current_Z = " " + coords[2]

    #Remove XYZ and spaces
    for character in 'XYZ\n':
        chunk = chunk.replace(character, '')
        chunk = chunk.replace(" ", ",")
        chunk = chunk.replace(",,", ",")

    newlines.append(chunk)
#Save to csv
np.savetxt('gcode_coordinates.csv', newlines, delimiter=",", fmt = '%s')

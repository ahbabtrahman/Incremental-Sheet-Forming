import re
import numpy as np

hover_height = 10.0      
hover_threshold = 1.0    

flat_z = 0               
hover_at_ends = True

gcode=open("line4.nc", 'r')
raw_lines = gcode.readlines()

# Clean up: remove comments, % markers, blank lines 
lines = []                                
for l in raw_lines:
    l = re.sub(r'\(.*?\)', '', l)          # (anything in parentheses)
    l = re.sub(r'\(.*', '', l)             # unclosed "(" comment to end of line
    l = re.sub(r';.*', '', l)              # ;comments
    l = l.replace('%', '')                 # program start/end markers
    if l.strip() == '':
        continue
    lines.append(l)

#  Auto-detect format 
is_gcode = any(re.search(r'[XYZxyz]\s*[-+]?\d', l) for l in lines)
print("Detected format:", "G-code" if is_gcode else "tab-separated columns")

newlines = []

if is_gcode:
    current_X = "X0.0"
    current_Y = "Y0.0"
    current_Z = "Z0.0"

    for chunk in lines:
        # Remove any G / F / M / N words (works for any feed rate)
        chunk = re.sub(r'[GFMNgfmn]\s*[-+]?\d*\.?\d+', '', chunk)
        chunk = chunk.strip().upper()

        # Skip lines with no coordinates at all
        if 'X' not in chunk and 'Y' not in chunk and 'Z' not in chunk:
            continue

        parts = chunk.split()

        # Fill in any missing coordinates from the last known position
        # Only Z coord
        if 'X' not in chunk and 'Y' not in chunk and 'Z' in chunk:
            chunk = current_X + " " + current_Y + " " + chunk
        # Only X coord
        elif 'Y' not in chunk and 'Z' not in chunk and 'X' in chunk:
            chunk = chunk + " " + current_Y + " " + current_Z
        # Only Y coord
        elif 'Z' not in chunk and 'X' not in chunk and 'Y' in chunk:
            chunk = current_X + " " + chunk + " " + current_Z
        # Only X and Y
        elif 'Z' not in chunk and 'X' in chunk and 'Y' in chunk:
            chunk = chunk + " " + current_Z
        # Only Y and Z
        elif 'X' not in chunk and 'Y' in chunk and 'Z' in chunk:
            chunk = current_X + " " + chunk
        # Only X and Z
        elif 'Y' not in chunk and 'X' in chunk and 'Z' in chunk:
            chunk = parts[0] + " " + current_Y + " " + parts[1]
        # (X, Y and Z all present -> nothing to fill in)

        # Update current coordinates (these keep the TRUE G-code values)
        coords = chunk.split()
        current_X = coords[0]
        current_Y = coords[1]
        current_Z = coords[2]

        # Remove XYZ letters and override Z: hover height or 0
        x = coords[0].replace('X', '')
        y = coords[1].replace('Y', '')
        z = float(coords[2].replace('Z', ''))
        z_out = hover_height if z > hover_threshold else 0

        newlines.append(f"{x},{y},{z_out}")

        if len(newlines) % 1000 == 0:
            print(f"{len(newlines)} points processed...")

else:
    # Blank cells mean "same as the previous row"
    rows = []
    has_z = False
    for chunk in lines:
        chunk = chunk.rstrip('\r\n')       # keep tabs: a leading tab means X is blank
        if chunk.strip() == '':
            continue
        parts = chunk.split('\t')
        parts += [''] * (3 - len(parts))   # pad to 3 columns
        vals = [float(p) if p.strip() else None for p in parts[:3]]
        if vals[2] is not None:
            has_z = True
        rows.append(vals)

    # If the file starts with a blank, back-fill from the first value that appears
    current = [next((r[i] for r in rows if r[i] is not None), 0.0) for i in range(3)]
    if rows and any(rows[0][i] is None for i in range(2)):
        print("Warning: first row is missing X or Y, using the first known value instead.")

    for x, y, z in rows:
        x = current[0] if x is None else x
        y = current[1] if y is None else y
        z = current[2] if z is None else z
        current = [x, y, z]
        if has_z:
            z_out = hover_height if z > hover_threshold else 0   # same rule as G-code mode
        else:
            z_out = flat_z
        newlines.append(f"{x},{y},{z_out}")

        if len(newlines) % 1000 == 0:
            print(f"{len(newlines)} points processed...")

    # No Z data in the file: lift above the first and last points
    if not has_z and hover_at_ends and newlines:
        fx, fy, _ = newlines[0].split(',')
        lx, ly, _ = newlines[-1].split(',')
        newlines.insert(0, f"{fx},{fy},{hover_height}")
        newlines.append(f"{lx},{ly},{hover_height}")

# Save 
np.savetxt('line4.csv', newlines, delimiter=",", fmt = '%s')

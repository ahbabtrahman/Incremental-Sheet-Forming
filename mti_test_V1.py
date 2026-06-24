import numpy as np
import time
import os
from collections import deque
from matplotlib import pyplot as plt
from RobotRaconteur.Client import *

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MTI_SERVICE_URL = "rr+tcp://localhost:60830/?service=MTI2D"
EXPOSURE_TIME = "25"           # MTI exposure time in ms
NUM_FRAMES = 50                # size of the rolling window of frames averaged together
FRAME_DELAY = 0.02             # seconds between frames (20ms)
OUTPUT_FILENAME = "test"       # output file saved as test.png
Z_MIN_THRESHOLD = 40           # discard scanner points with Z < this value (noise filter)
X_MIN = -5                     # discard scanner points with X < this value (mm)
X_MAX = 10                     # discard scanner points with X > this value (mm)
PLOT_TITLE = "Live Snapshot"   # title shown on the profile plot
NUM_BINS = 500                 # number of X bins used to build the averaged profile

# Save to Windows Downloads folder
DOWNLOADS_FOLDER = os.path.join(os.path.expanduser("~"), "Downloads")

# ---------------------------------------------------------------
# CONNECT TO MTI SCANNER
# ---------------------------------------------------------------
print("Connecting to MTI scanner...")
try:
    mti_client = RRN.ConnectService(MTI_SERVICE_URL)
    mti_client.setExposureTime(EXPOSURE_TIME)
    print(f"Connected to MTI scanner at {MTI_SERVICE_URL}")
except Exception as e:
    print(f"ERROR: Could not connect to MTI scanner: {e}")
    print("Make sure mti2D_RR.exe is running in a separate terminal:")
    print("  .\\mti2D_RR.exe --scanner-ip-address=192.168.60.14")
    exit(1)

# ---------------------------------------------------------------
# LIVE PLOT SETUP
# ---------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], color='steelblue', linewidth=1.2)
ax.set_title(PLOT_TITLE, fontsize=13)
ax.set_xlabel("X (mm)", fontsize=11)
ax.set_ylabel("Z (mm)", fontsize=11)
ax.set_xlim(X_MIN, X_MAX)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

stop_requested = False
def on_key(event):
    global stop_requested
    if event.key == 'enter':
        stop_requested = True
fig.canvas.mpl_connect('key_press_event', on_key)

print("Streaming live profile. Click the plot window and press Enter to stop.")

# ---------------------------------------------------------------
# LIVE CAPTURE + PROCESS LOOP
# Each new frame is filtered, mirrored, and folded into a rolling
# window of the last NUM_FRAMES frames, then re-binned/averaged/
# detrended to produce the displayed profile.
# ---------------------------------------------------------------
bin_edges = np.linspace(X_MIN, X_MAX, NUM_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
frame_buffer = deque(maxlen=NUM_FRAMES)
x_plot, z_plot = np.array([]), np.array([])

while not stop_requested and plt.fignum_exists(fig.number):
    try:
        x_data = np.array(mti_client.lineProfile.X_data)
        z_data = np.array(mti_client.lineProfile.Z_data)
    except Exception as e:
        print(f"  Warning: frame capture failed: {e}")
        plt.pause(FRAME_DELAY)
        continue

    # Filter out noise below Z threshold and outside X range
    valid_mask = (z_data > Z_MIN_THRESHOLD) & (x_data > X_MIN) & (x_data < X_MAX)
    x_data = x_data[valid_mask]
    z_data = z_data[valid_mask]
    if len(x_data) == 0:
        plt.pause(FRAME_DELAY)
        continue

    # Mirror X axis (scanner convention, same as original script)
    x_data = x_data * -1
    frame_buffer.append((x_data, z_data))

    # Bin X values across the recent frames and average Z per bin
    all_x_flat = np.concatenate([f[0] for f in frame_buffer])
    all_z_flat = np.concatenate([f[1] for f in frame_buffer])
    bin_indices = np.clip(np.digitize(all_x_flat, bin_edges) - 1, 0, NUM_BINS - 1)

    z_avg = np.full(NUM_BINS, np.nan)
    for b in range(NUM_BINS):
        mask = bin_indices == b
        if mask.sum() > 0:
            z_avg[b] = np.mean(all_z_flat[mask])

    valid = ~np.isnan(z_avg)
    x_plot = bin_centers[valid]
    z_plot = z_avg[valid]

    if len(x_plot) >= 2:
        # Detrend: fit and subtract the scanner tilt (~50 deg)
        slope, intercept = np.polyfit(x_plot, z_plot, 1)
        z_plot = z_plot - (slope * x_plot + intercept)
        # Invert about the X-axis so the profile dips instead of rising
        z_plot = -z_plot

        line.set_data(x_plot, z_plot)
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(FRAME_DELAY)

plt.ioff()

# ---------------------------------------------------------------
# SAVE FINAL PROFILE PLOT
# ---------------------------------------------------------------
if len(x_plot) > 0:
    png_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Profile plot saved to: {png_path}")

plt.close(fig)
print("Done.")

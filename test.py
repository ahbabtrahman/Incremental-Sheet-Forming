import numpy as np
import time
import os
from collections import deque
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from RobotRaconteur.Client import *

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MTI_SERVICE_URL   = "rr+tcp://localhost:60830/?service=MTI2D"
EXPOSURE_TIME     = "25"
NUM_FRAMES        = 50
FRAME_DELAY       = 0.02
OUTPUT_FILENAME   = "test"
Z_MIN_THRESHOLD   = 40
X_MIN             = -5
X_MAX             = 10    # extended to capture right plateau after scanner move
NUM_BINS          = 500

PEN_TO_LASER_OFFSET_MM = 0.5 * 25.4   # 12.7 mm
X_BIN_MIN = X_MIN + PEN_TO_LASER_OFFSET_MM   #  7.7 mm
X_BIN_MAX = X_MAX + PEN_TO_LASER_OFFSET_MM   # 17.7 mm

NODATA_SENTINEL      = 0.0
PEN_X_CENTER         = 0.0 + PEN_TO_LASER_OFFSET_MM
PEN_X_HALFWIDTH      = 0.75
HAMPEL_WINDOW        = 7
HAMPEL_K             = 3.0
FRAME_QUALITY_STD_THRESHOLD = 2.0   # raised: step edge produces high residuals
SAVGOL_WINDOW        = 15
SAVGOL_POLYORDER     = 3
LOG_RAW_FRAMES       = True

# --- Z calibration reference ---
# Step-edge method: place a second 0.003" sheet on the flat base, scan the edge.
# Z_SCALE = SHEET_THICKNESS_MM / measured_step_su
# Last measured: step = 0.2712 su → Z_SCALE = 0.0762 / 0.2712 = 0.2810 mm/su
# Verified 2026-06-26: perpendicular-orientation rescan of 16 channels (Set 1 & 2,
# 3.25–5.0 N) gave depths of 0.14–0.26 mm, consistent with Z_SCALE = 0.2810.
SHEET_THICKNESS_IN   = 0.0038 * 8     # inches — 8 sheets of 0.0038" paper stacked
SHEET_THICKNESS_MM   = SHEET_THICKNESS_IN * 25.4   # 0.0762 mm

# --- Plateau windows for step measurement (physical frame, mm from pen tip) ---
# Set these to flats regions on EITHER SIDE of the sheet edge, avoiding the edge spike.
# Adjust after first look at the plot if the edge lands in a different spot.
LEFT_PLATEAU_LO  = 8.0    # mm  — base aluminum side
LEFT_PLATEAU_HI  = 11.5   # mm
RIGHT_PLATEAU_LO = 15.0   # mm  — top-of-sheet side
RIGHT_PLATEAU_HI = 17.5   # mm

if SAVGOL_WINDOW % 2 == 0:
    SAVGOL_WINDOW += 1

DOWNLOADS_FOLDER = os.path.join(os.path.expanduser("~"), "Downloads")


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def hampel(z, window=7, k=3.0):
    z = z.copy()
    h = window // 2
    for i in range(len(z)):
        seg = z[max(0, i - h): i + h + 1]
        med = np.nanmedian(seg)
        mad = 1.4826 * np.nanmedian(np.abs(seg - med))
        if mad > 0 and not np.isnan(z[i]) and np.abs(z[i] - med) > k * mad:
            z[i] = np.nan
    return z


def frame_to_grid(x, z, grid):
    if len(x) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(x)
    xs, zs = x[order], z[order]
    valid = ~np.isnan(zs)
    if valid.sum() < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, xs[valid], zs[valid], left=np.nan, right=np.nan)


def fill_interior_nans(x_grid, z_grid):
    z = z_grid.copy()
    has_data = ~np.isnan(z)
    if has_data.sum() < 2:
        return z
    first = np.argmax(has_data)
    last  = len(has_data) - 1 - np.argmax(has_data[::-1])
    interior_gap = np.isnan(z)
    interior_gap[:first] = False
    interior_gap[last + 1:] = False
    if interior_gap.any():
        z[interior_gap] = np.interp(x_grid[interior_gap], x_grid[has_data], z[has_data])
    return z


def measure_step(x, z, left_lo, left_hi, right_lo, right_hi):
    """
    Compute step height as mean(right plateau) - mean(left plateau).
    Returns (step_scanner_units, left_mean, right_mean) or (nan, nan, nan)
    if either plateau has no data.
    """
    left_mask  = (x >= left_lo)  & (x <= left_hi)  & ~np.isnan(z)
    right_mask = (x >= right_lo) & (x <= right_hi) & ~np.isnan(z)
    if left_mask.sum() < 3 or right_mask.sum() < 3:
        return np.nan, np.nan, np.nan
    left_mean  = float(np.mean(z[left_mask]))
    right_mean = float(np.mean(z[right_mask]))
    return right_mean - left_mean, left_mean, right_mean


# ---------------------------------------------------------------
# CONNECT
# ---------------------------------------------------------------
print("Connecting to MTI scanner...")
try:
    mti_client = RRN.ConnectService(MTI_SERVICE_URL)
    mti_client.setExposureTime(EXPOSURE_TIME)
    print("Connected. Starting capture immediately.")
    print(f"Sheet reference: {SHEET_THICKNESS_IN}\" = {SHEET_THICKNESS_MM:.4f} mm")
    print(f"Left plateau:  {LEFT_PLATEAU_LO}–{LEFT_PLATEAU_HI} mm from pen tip")
    print(f"Right plateau: {RIGHT_PLATEAU_LO}–{RIGHT_PLATEAU_HI} mm from pen tip")
    print("Press Enter in plot window to stop and save.")
except Exception as e:
    print(f"ERROR: Could not connect: {e}")
    print("Make sure mti2D_RR.exe is running:")
    print("  .\\mti2D_RR.exe --scanner-ip-address=192.168.60.14")
    exit(1)


# ---------------------------------------------------------------
# LIVE PLOT SETUP
# ---------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(9, 5))
line, = ax.plot([], [], color='steelblue', linewidth=1.4)

# Plateau region markers
ax.axvspan(LEFT_PLATEAU_LO,  LEFT_PLATEAU_HI,  alpha=0.10, color='green',  label='left plateau')
ax.axvspan(RIGHT_PLATEAU_LO, RIGHT_PLATEAU_HI, alpha=0.10, color='orange', label='right plateau')
ax.axvline(0.0, color='red', linestyle=':', linewidth=0.8, alpha=0.6, label='pen tip (X=0)')

info_text = ax.text(
    0.02, 0.95, 'Buffering...', transform=ax.transAxes, fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)

ax.set_title(
    f"Z Calibration  |  Sheet = {SHEET_THICKNESS_IN}\" ({SHEET_THICKNESS_MM:.4f} mm)  |  "
    f"Step = mean(right) - mean(left)",
    fontsize=10
)
ax.set_xlabel(f"Distance from pen tip (mm)  [laser offset = {PEN_TO_LASER_OFFSET_MM:.1f} mm]", fontsize=10)
ax.set_ylabel("Z (scanner units — NOT calibrated mm)", fontsize=10)
ax.set_xlim(X_BIN_MIN, X_BIN_MAX)  # now 7.7–17.7mm
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()

stop_requested = False
def on_key(event):
    global stop_requested
    if event.key == 'enter':
        stop_requested = True
fig.canvas.mpl_connect('key_press_event', on_key)


# ---------------------------------------------------------------
# CAPTURE + PROCESS LOOP
# ---------------------------------------------------------------
bin_edges   = np.linspace(X_BIN_MIN, X_BIN_MAX, NUM_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
frame_buffer  = deque(maxlen=NUM_FRAMES)
raw_frame_log = []
x_plot, z_plot = np.array([]), np.array([])

while not stop_requested and plt.fignum_exists(fig.number):

    try:
        x_data = np.array(mti_client.lineProfile.X_data, dtype=float)
        z_data = np.array(mti_client.lineProfile.Z_data, dtype=float)
    except Exception as e:
        print(f"  Warning: frame capture failed: {e}")
        plt.pause(FRAME_DELAY)
        continue

    if LOG_RAW_FRAMES:
        raw_frame_log.append((x_data.copy(), z_data.copy()))

    z_data[z_data == NODATA_SENTINEL] = np.nan
    valid_mask = (
        ~np.isnan(z_data) &
        (z_data > Z_MIN_THRESHOLD) &
        (x_data > X_MIN) &
        (x_data < X_MAX)
    )
    x_data = x_data[valid_mask]
    z_data = z_data[valid_mask]
    if len(x_data) == 0:
        plt.pause(FRAME_DELAY)
        continue

    x_data = x_data * -1 + PEN_TO_LASER_OFFSET_MM

    pen_mask = np.abs(x_data - PEN_X_CENTER) < PEN_X_HALFWIDTH
    z_data[pen_mask] = np.nan

    z_data = hampel(z_data, window=HAMPEL_WINDOW, k=HAMPEL_K)

    valid_pts = ~np.isnan(z_data)
    if valid_pts.sum() >= 2:
        _slope, _ic = np.polyfit(x_data[valid_pts], z_data[valid_pts], 1)
        _resid_std  = np.std(z_data[valid_pts] - (_slope * x_data[valid_pts] + _ic))
        if _resid_std > FRAME_QUALITY_STD_THRESHOLD:
            print(f"  [quality gate] Frame rejected (std = {_resid_std:.3f})")
            plt.pause(FRAME_DELAY)
            continue

    frame_buffer.append((x_data, z_data))

    stack = np.array([frame_to_grid(f[0], f[1], bin_centers) for f in frame_buffer])
    z_med = np.nanmedian(stack, axis=0)
    z_med = fill_interior_nans(bin_centers, z_med)

    valid  = ~np.isnan(z_med)
    x_plot = bin_centers[valid]
    z_plot = z_med[valid]

    if len(x_plot) < SAVGOL_WINDOW + 1:
        plt.pause(FRAME_DELAY)
        continue

    z_plot = savgol_filter(z_plot, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    # Detrend using LEFT PLATEAU ONLY as the reference flat surface.
    # This avoids the step from biasing the tilt correction.
    left_mask = (x_plot >= LEFT_PLATEAU_LO) & (x_plot <= LEFT_PLATEAU_HI)
    if left_mask.sum() >= 2:
        slope, intercept = np.polyfit(x_plot[left_mask], z_plot[left_mask], 1)
        z_plot = z_plot - (slope * x_plot + intercept)

    # Invert: top-of-sheet appears as positive bump
    z_plot = -z_plot

    # Measure step from plateau means
    step, left_mean, right_mean = measure_step(
        x_plot, z_plot, LEFT_PLATEAU_LO, LEFT_PLATEAU_HI,
        RIGHT_PLATEAU_LO, RIGHT_PLATEAU_HI
    )

    buf_pct = int(100 * len(frame_buffer) / NUM_FRAMES)

    if buf_pct < 100:
        info_str = f"Buffering: {buf_pct}%"
    elif np.isnan(step):
        info_str = (
            f"Buffer: 100%\n"
            f"No data in one or both plateau windows.\n"
            f"Adjust LEFT/RIGHT_PLATEAU_LO/HI in config."
        )
    else:
        z_scale = SHEET_THICKNESS_MM / step if step > 0 else float('nan')
        info_str = (
            f"Buffer: 100%\n"
            f"Left plateau mean:  {left_mean:.4f}  |  Right plateau mean: {right_mean:.4f}\n"
            f"Step (scanner units): {step:.4f}  |  True step: {SHEET_THICKNESS_MM:.4f} mm\n"
            f"Z scale = {z_scale:.4f} mm / scanner unit"
        )

    line.set_data(x_plot, z_plot)
    # Clip Y axis to data percentile range to prevent edge spikes from blowing out the scale
    if len(z_plot) > 10:
        z_lo, z_hi = np.nanpercentile(z_plot, [2, 98])
        margin = max(abs(z_hi - z_lo) * 0.4, 0.05)
        ax.set_ylim(z_lo - margin, z_hi + margin)
    ax.set_xlim(X_BIN_MIN, X_BIN_MAX)
    info_text.set_text(info_str)
    fig.canvas.draw_idle()
    plt.pause(FRAME_DELAY)

plt.ioff()


# ---------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------
if len(x_plot) > 0:
    step_f, left_f, right_f = measure_step(
        x_plot, z_plot, LEFT_PLATEAU_LO, LEFT_PLATEAU_HI,
        RIGHT_PLATEAU_LO, RIGHT_PLATEAU_HI
    )
    print(f"\n--- Z CALIBRATION SUMMARY ---")
    print(f"Sheet thickness        : {SHEET_THICKNESS_MM:.4f} mm  ({SHEET_THICKNESS_IN}\")")
    print(f"Left plateau mean      : {left_f:.4f} scanner units  ({LEFT_PLATEAU_LO}–{LEFT_PLATEAU_HI} mm)")
    print(f"Right plateau mean     : {right_f:.4f} scanner units  ({RIGHT_PLATEAU_LO}–{RIGHT_PLATEAU_HI} mm)")
    print(f"Measured step          : {step_f:.4f} scanner units")
    if not np.isnan(step_f) and step_f > 0:
        zscale = SHEET_THICKNESS_MM / step_f
        print(f"Z scale factor         : {zscale:.4f} mm / scanner unit")
        print(f"  -> real_mm = scanner_z x {zscale:.4f}")
        print(f"  -> Update Z_SCALE in mti_snapshot.py if this differs from 0.2810")
    print(f"-----------------------------\n")

    png_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Plot saved: {png_path}")

plt.close(fig)

if LOG_RAW_FRAMES and len(raw_frame_log) > 0:
    npy_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}_raw_frames.npy")
    np.save(npy_path, np.array(raw_frame_log, dtype=object), allow_pickle=True)
    print(f"Raw frames ({len(raw_frame_log)}) saved: {npy_path}")

print("Done.")

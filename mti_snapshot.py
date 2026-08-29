import numpy as np
import time
import os
import csv
from collections import deque
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from RobotRaconteur.Client import *

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MTI_SERVICE_URL = "rr+tcp://localhost:60830/?service=MTI2D"
EXPOSURE_TIME = "25"           # MTI exposure time in ms
NUM_FRAMES = 50                # rolling window size (frames stacked for median average)
FRAME_DELAY = 0.02             # seconds between frames (20 ms)
OUTPUT_FILENAME = time.strftime("scan_%Y%m%d_%H%M%S")  # e.g. scan_20260626_143022
Z_MIN_THRESHOLD = 40           # discard points with Z < this (hardware noise floor)
X_MIN = -5                     # raw scanner ROI lower bound (mm, scanner frame)
X_MAX = 5                     # scanner FOV ends ~17.7mm in physical frame
PLOT_TITLE = "Live Snapshot"
NUM_BINS = 500                 # X bins for the averaged profile grid

# --- Coordinate frame ---
# The 0.5" pen-to-laser distance is along the Y axis (direction of travel),
# creating kinematic lag — NOT along the scanner's X axis (cross-section).
# So no X offset is applied after mirroring. X = 0 corresponds to the
# scanner's cross-section origin, which aligns with the pen tip.
# ⚠️ PIN: The channel consistently appears at ~1.3 mm from X=0, indicating
#         a small cross-sectional misalignment between scanner and pen tip.
#         Using data-derived channel position for now. Calibrate properly
#         at end of project alongside Z calibration.
PEN_TO_LASER_OFFSET_MM = 0.0   # X-axis offset (corrected: 0.5" offset is in Y/travel axis)

# Display limits (scanner frame after mirror, no X offset)
X_BIN_MIN = X_MIN + PEN_TO_LASER_OFFSET_MM   # -5.0 mm
X_BIN_MAX = X_MAX + PEN_TO_LASER_OFFSET_MM   #  5.0 mm

# --- 1. SDK sentinel (no-data) value ---
# Confirmed from raw frame inspection: the scanner outputs Z = 0.0 for failed
# returns. Z_MIN_THRESHOLD = 40 already catches these, but we also mask
# explicitly here for clarity.
NODATA_SENTINEL = 0.0

# --- 2. Pen-occlusion mask ---
# Pen occlusion appears at X = 0 in the mirrored scanner frame.
PEN_X_CENTER    = 0.0    # mm — pen sits at scanner cross-section origin
PEN_X_HALFWIDTH = 0.75   # mm — half-width of exclusion window

# --- 3. Hampel spike filter ---
HAMPEL_WINDOW  = 7             # total window size in samples (±3 neighbours each side)
HAMPEL_K       = 3.0           # rejection threshold multiplier on MAD

# --- 5. Local detrend / channel search window ---
# Channel observed consistently at ~1.3 mm from pen center in scanner frame.
# Window set from data: using 0.0–3.5 mm to capture full channel profile.
CHANNEL_X_LO   = 0.0    # mm in scanner cross-section frame
CHANNEL_X_HI   = 3.5    # mm in scanner cross-section frame

# --- Frame-quality gate ---
# Frames whose detrended Z std exceeds this threshold are corrupted (e.g. a
# movement jolt or laser dropout) and are discarded before entering the buffer.
# From data: good frames have std ~0.028 mm; Frame 9 had std = 2.04 mm.
FRAME_QUALITY_STD_THRESHOLD = 1.50   # mm — raised for perpendicular orientation: channel dip raises residuals to ~0.575mm; genuine corruption (jolts) is >2mm

# --- 6. Savitzky-Golay smoothing ---
SAVGOL_WINDOW  = 15            # must be odd and > SAVGOL_POLYORDER
SAVGOL_POLYORDER = 3

# --- Noise floor calibration ---
# At startup the script collects NUM_FRAMES frames on the flat surface and
# computes the std of the median profile as the noise floor. Any depth reading
# below NOISE_FLOOR_MULTIPLIER × noise_floor is reported as "below detection"
# rather than a real measurement.
NOISE_FLOOR_MULTIPLIER = 2.0   # detection threshold = 2 × measured noise floor

# --- Z scale calibration ---
# Derived from step-edge scan of a 0.003" (0.0762 mm) aluminum sheet placed on
# the base surface. Plateau mean difference = 0.2712 scanner units.
# Z_SCALE = 0.0762 / 0.2712 = 0.2810 mm / scanner unit.
# Verified 2026-06-26: perpendicular-orientation rescan of 16 channels (Set 1 & 2,
# 3.0–5.0 N) gave clean depths of 0.14–0.26 mm, consistent with this scale.
# ⚠️ PIN: Re-verify with a precision gauge block before final analysis.
Z_SCALE = 0.2810   # mm per scanner unit

# --- 8. Raw frame logging ---
LOG_RAW_FRAMES = True          # save pre-processing frames to .npy on exit

# Enforce odd SAVGOL_WINDOW
if SAVGOL_WINDOW % 2 == 0:
    SAVGOL_WINDOW += 1

# Save to Windows Downloads folder
DOWNLOADS_FOLDER = os.path.join(os.path.expanduser("~"), "Downloads")


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def hampel(z, window=7, k=3.0):
    """
    Hampel outlier filter. Marks samples more than k*MAD from their local
    median as NaN (rather than replacing with median, so the nanmedian stack
    excludes them entirely instead of biasing the average).
    """
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
    """
    Interpolate one cleaned frame onto the common X grid.
    Points outside the frame's X range return NaN (no extrapolation).
    """
    if len(x) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(x)
    xs, zs = x[order], z[order]
    valid = ~np.isnan(zs)
    if valid.sum() < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, xs[valid], zs[valid], left=np.nan, right=np.nan)


def fill_interior_nans(x_grid, z_grid):
    """
    Linearly interpolate across interior NaN gaps (e.g. pen-occlusion mask)
    without extrapolating beyond the first/last valid sample.

    Edge NaNs (outside the scanner's live range) are left as NaN so they
    are excluded from downstream processing.
    """
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
        z[interior_gap] = np.interp(
            x_grid[interior_gap],
            x_grid[has_data],
            z[has_data]
        )
    return z


def collect_and_process_frame(mti_client, bin_centers):
    """
    Read one frame from the scanner and run it through the full pre-processing
    pipeline (sentinel mask → ROI filter → mirror + offset → pen mask → Hampel
    → quality gate → grid interpolation).

    Returns (x_phys, z_clean) on success, or None if the frame is invalid or
    fails the quality gate.
    """
    try:
        x_data = np.array(mti_client.lineProfile.X_data, dtype=float)
        z_data = np.array(mti_client.lineProfile.Z_data, dtype=float)
    except Exception:
        return None

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
        return None

    x_data = x_data * -1 + PEN_TO_LASER_OFFSET_MM
    pen_mask = np.abs(x_data - PEN_X_CENTER) < PEN_X_HALFWIDTH
    z_data[pen_mask] = np.nan
    z_data = hampel(z_data, window=HAMPEL_WINDOW, k=HAMPEL_K)

    # Quality gate
    valid_pts = ~np.isnan(z_data)
    if valid_pts.sum() < 2:
        return None
    _slope, _ic = np.polyfit(x_data[valid_pts], z_data[valid_pts], 1)
    _resid_std = np.std(z_data[valid_pts] - (_slope * x_data[valid_pts] + _ic))
    if _resid_std > FRAME_QUALITY_STD_THRESHOLD:
        return None

    return (x_data, z_data)


def calibrate_noise_floor(mti_client, bin_centers):
    """
    Collect NUM_FRAMES frames on the flat surface and compute the noise floor
    as the std of the median-stacked, detrended profile.

    Returns noise_floor in mm.
    """
    print("Calibrating noise floor — keep laser on flat surface...")
    cal_buffer = []
    attempts = 0
    max_attempts = NUM_FRAMES * 3   # allow extra attempts for rejected frames

    while len(cal_buffer) < NUM_FRAMES and attempts < max_attempts:
        result = collect_and_process_frame(mti_client, bin_centers)
        attempts += 1
        if result is None:
            time.sleep(FRAME_DELAY)
            continue
        x_f, z_f = result
        cal_buffer.append(frame_to_grid(x_f, z_f, bin_centers))
        # Simple progress indicator every 10 frames
        if len(cal_buffer) % 10 == 0:
            print(f"  {len(cal_buffer)}/{NUM_FRAMES} calibration frames collected...")
        time.sleep(FRAME_DELAY)

    if len(cal_buffer) < 10:
        print("  WARNING: fewer than 10 good calibration frames — noise floor unreliable.")

    stack = np.array(cal_buffer)
    z_med = np.nanmedian(stack, axis=0)
    z_med = fill_interior_nans(bin_centers, z_med)

    # Detrend using full profile (no channel present during calibration)
    valid = ~np.isnan(z_med)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(bin_centers[valid], z_med[valid], 1)
        z_detrended = z_med[valid] - (slope * bin_centers[valid] + intercept)
        noise_floor = float(np.std(z_detrended))
    else:
        noise_floor = 0.05   # fallback if calibration fails

    print(f"  Noise floor: {noise_floor*1000:.1f} µm  →  "
          f"Detection threshold: {noise_floor * NOISE_FLOOR_MULTIPLIER * 1000:.1f} µm")
    return noise_floor


def extract_depth_width(x, z, channel_lo, channel_hi, width_frac=0.25):
    """
    From a detrended-and-inverted profile (surface ≈ 0, channel dips negative):
      depth = abs(min Z inside channel window)
      width = X span where Z < -(depth * width_frac)
    Returns (depth_mm, width_mm), or (nan, nan) if the channel window is empty.
    """
    in_ch = (x >= channel_lo) & (x <= channel_hi) & ~np.isnan(z)
    if in_ch.sum() == 0:
        return np.nan, np.nan
    bottom = np.nanmin(z[in_ch])
    depth = -bottom                          # positive value
    if depth <= 0:
        return 0.0, 0.0
    threshold = -(depth * width_frac)        # negative threshold
    below = in_ch & (z < threshold)
    width = float(x[below].max() - x[below].min()) if below.sum() > 0 else 0.0
    return float(depth), width


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
# CALIBRATE NOISE FLOOR (flat surface at startup)
# ---------------------------------------------------------------
bin_edges_cal = np.linspace(X_BIN_MIN, X_BIN_MAX, NUM_BINS + 1)
bin_centers_cal = 0.5 * (bin_edges_cal[:-1] + bin_edges_cal[1:])
noise_floor = calibrate_noise_floor(mti_client, bin_centers_cal)
DETECTION_THRESHOLD = noise_floor * NOISE_FLOOR_MULTIPLIER


# ---------------------------------------------------------------
# LIVE PLOT SETUP
# ---------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], color='steelblue', linewidth=1.2)

# Overlay lines marking the channel search window
ax.axvline(CHANNEL_X_LO, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='channel window')
ax.axvline(CHANNEL_X_HI, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)

# Vertical marker showing pen tip location (X = 0 in physical frame)
ax.axvline(0.0, color='red', linestyle=':', linewidth=0.8, alpha=0.6, label='pen tip (X=0)')

# Text box for live depth / width readout
depth_text = ax.text(
    0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6)
)

ax.set_title(PLOT_TITLE, fontsize=13)
ax.set_xlabel("Scanner cross-section position (mm)  [X=0 = pen tip, channel at ~1.3 mm]", fontsize=10)
ax.set_ylabel("Depth (mm)", fontsize=11)
ax.set_xlim(X_BIN_MIN, X_BIN_MAX)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=9)
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
# ---------------------------------------------------------------
bin_edges   = np.linspace(X_BIN_MIN, X_BIN_MAX, NUM_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
frame_buffer   = deque(maxlen=NUM_FRAMES)
raw_frame_log  = []                          # (8) raw frames before any processing
depth_log      = []                          # per-frame: [timestamp, depth_mm, width_mm, buffer_%]
x_plot, z_plot = np.array([]), np.array([])

while not stop_requested and plt.fignum_exists(fig.number):

    # --- Read one frame from the scanner ---
    try:
        x_data = np.array(mti_client.lineProfile.X_data, dtype=float)
        z_data = np.array(mti_client.lineProfile.Z_data, dtype=float)
    except Exception as e:
        print(f"  Warning: frame capture failed: {e}")
        plt.pause(FRAME_DELAY)
        continue

    # (8) Log truly raw frame before any modification
    if LOG_RAW_FRAMES:
        raw_frame_log.append((x_data.copy(), z_data.copy()))

    # (1) Mask SDK sentinel / no-data values
    z_data[z_data == NODATA_SENTINEL] = np.nan

    # (1 cont.) Mask Z below hardware threshold and X outside ROI (scanner frame)
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

    # Mirror X axis (scanner hardware convention), then shift to physical frame
    # so that X = 0 corresponds to the pen tip.
    x_data = x_data * -1 + PEN_TO_LASER_OFFSET_MM

    # (2) Pen-occlusion mask — NaN out the fixed X window where the pen body
    #     blocks the laser. Interior NaNs are interpolated after the median
    #     stack (see fill_interior_nans below); we do NOT replace here so that
    #     the nanmedian stack ignores these samples entirely.
    pen_mask = np.abs(x_data - PEN_X_CENTER) < PEN_X_HALFWIDTH
    z_data[pen_mask] = np.nan

    # (3) Hampel spike filter on this single frame
    z_data = hampel(z_data, window=HAMPEL_WINDOW, k=HAMPEL_K)

    # Frame-quality gate: reject corrupted frames (movement jolts, laser
    # dropouts) before they enter the buffer. We fit a quick line to the
    # frame and check the std of the residuals. Good frames: ~0.028 mm.
    # Corrupted frames (e.g. Frame 9 in first run): std > 2 mm.
    if len(x_data) >= 2:
        valid_pts = ~np.isnan(z_data)
        if valid_pts.sum() >= 2:
            _slope, _ic = np.polyfit(x_data[valid_pts], z_data[valid_pts], 1)
            _resid_std = np.std(z_data[valid_pts] - (_slope * x_data[valid_pts] + _ic))
            if _resid_std > FRAME_QUALITY_STD_THRESHOLD:
                print(f"  [quality gate] Frame rejected — detrended std = {_resid_std:.3f} mm")
                plt.pause(FRAME_DELAY)
                continue

    frame_buffer.append((x_data, z_data))

    # (4) Median stack: interpolate every buffered frame onto the common grid,
    #     then take nanmedian — simultaneous averaging + spike rejection
    stack  = np.array([frame_to_grid(f[0], f[1], bin_centers) for f in frame_buffer])
    z_med  = np.nanmedian(stack, axis=0)

    # (2 cont.) Pen-occlusion interpolation — fill the interior NaN gap left
    #     by the pen mask with linear interpolation from the neighbouring
    #     shoulder points. This removes the spike artifact cleanly without
    #     leaving a hole in the profile.
    z_med = fill_interior_nans(bin_centers, z_med)

    valid  = ~np.isnan(z_med)
    x_plot = bin_centers[valid]
    z_plot = z_med[valid]

    if len(x_plot) < SAVGOL_WINDOW + 1:
        plt.pause(FRAME_DELAY)
        continue

    # (6) Savitzky-Golay edge-preserving smooth
    z_plot = savgol_filter(z_plot, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

    # (5) Local detrend: fit a line to the shoulders only (outside channel window)
    #     so the channel dip itself doesn't pull the fit down
    shoulder_mask = (x_plot < CHANNEL_X_LO) | (x_plot > CHANNEL_X_HI)
    if shoulder_mask.sum() >= 2:
        try:
            slope, intercept = np.polyfit(x_plot[shoulder_mask], z_plot[shoulder_mask], 1)
            z_plot = z_plot - (slope * x_plot + intercept)
        except (np.linalg.LinAlgError, ValueError):
            pass   # degenerate fit — skip detrend this frame

    # Invert so the channel dips downward in the plot
    z_plot = -z_plot

    # (7) Depth & width extraction + Z scale correction
    depth_su, width = extract_depth_width(x_plot, z_plot, CHANNEL_X_LO, CHANNEL_X_HI)
    depth = depth_su * Z_SCALE if not np.isnan(depth_su) else np.nan   # convert to real mm

    # Log depth/width for this frame
    depth_log.append([
        time.time(),
        float(depth) if not np.isnan(depth) else np.nan,
        float(width) if not np.isnan(width) else np.nan,
        int(100 * len(frame_buffer) / NUM_FRAMES)
    ])

    # Update plot — convert to mm so Y axis matches depth readout
    z_plot_mm = z_plot * Z_SCALE
    line.set_data(x_plot, z_plot_mm)
    if len(z_plot_mm) > 10:
        z_lo, z_hi = np.nanpercentile(z_plot_mm, [2, 98])
        margin = max(abs(z_hi - z_lo) * 0.4, 0.05)
        ax.set_ylim(z_lo - margin, z_hi + margin)
    ax.set_xlim(X_BIN_MIN, X_BIN_MAX)

    buf_pct = int(100 * len(frame_buffer) / NUM_FRAMES)
    detect_thresh_mm = DETECTION_THRESHOLD * Z_SCALE
    if np.isnan(depth):
        depth_text.set_text(f"Depth: —    Width: —    Buffer: {buf_pct}%")
    elif depth < detect_thresh_mm:
        depth_text.set_text(
            f"Depth: below detection (<{detect_thresh_mm*1000:.0f} µm)    Buffer: {buf_pct}%"
        )
    else:
        depth_text.set_text(f"Depth: {depth:.3f} mm    Width: {width:.3f} mm    Buffer: {buf_pct}%")

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

# ---------------------------------------------------------------
# SAVE PROFILE CSV  (same format as line test reference files)
# Row 0: X positions (mm, scanner cross-section frame)
# Row 1: Z values (mm, detrended — channel dips negative, flat surface ≈ 0)
# ---------------------------------------------------------------
if len(x_plot) > 0:
    profile_csv_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}_profile.csv")
    z_mm = z_plot * Z_SCALE   # convert scanner units → mm
    np.savetxt(profile_csv_path, np.array([x_plot, z_mm]), delimiter=' ')
    print(f"Profile CSV saved to: {profile_csv_path}")

# ---------------------------------------------------------------
# SAVE PER-FRAME DEPTH LOG CSV
# ---------------------------------------------------------------
if len(depth_log) > 0:
    log_csv_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}_depth_log.csv")
    with open(log_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'depth_mm', 'width_mm', 'buffer_pct'])
        writer.writerows(depth_log)
    print(f"Depth log ({len(depth_log)} frames) saved to: {log_csv_path}")

# ---------------------------------------------------------------
# (8) SAVE RAW FRAME LOG
# ---------------------------------------------------------------
if LOG_RAW_FRAMES and len(raw_frame_log) > 0:
    npy_path = os.path.join(DOWNLOADS_FOLDER, f"{OUTPUT_FILENAME}_raw_frames.npy")
    np.save(npy_path, np.array(raw_frame_log, dtype=object), allow_pickle=True)
    print(f"Raw frames ({len(raw_frame_log)}) saved to: {npy_path}")

print("Done.")

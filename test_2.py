import numpy as np
import os
from collections import deque
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from RobotRaconteur.Client import *

MTI_SERVICE_URL   = "rr+tcp://localhost:60830/?service=MTI2D"
FRAME_DELAY       = 0.02
NUM_FRAMES        = 50
NUM_BINS          = 500
DOWNLOADS         = os.path.join(os.path.expanduser("~"), "Downloads")

# Pen spike mask — linearly interpolate across this region
PEN_MASK_MIN      = -10
PEN_MASK_MAX      =  -2

# Detrend mode:
#   'auto'  — compute polynomial from live data each frame (use on flat surface to find coefficients)
#   'fixed' — use hardcoded coefficients below (use for channel scans)
DETREND_MODE      = 'auto'
DETREND_COEFFS    = [0.0, 0.0, 0.0]  # [a, b, c] for a*x^2 + b*x + c — paste from flat scan terminal output

# Hampel filter (outlier removal)
HAMPEL_WINDOW     =   7   # half-window size
HAMPEL_K          =   3.0 # threshold in MAD units

# Savitzky-Golay smoothing
SAVGOL_WINDOW     =  11   # must be odd — larger = smoother
SAVGOL_ORDER      =   3   # polynomial order

# Frame quality gate — reject frames where linear residual std exceeds this
# Increase if too many frames are rejected; decrease for stricter filtering
FRAME_QUALITY_STD = 2.0

# Display X range
DISPLAY_X_MIN     = -25
DISPLAY_X_MAX     =   8

print("Connecting to MTI scanner...")
mti_client = RRN.ConnectService(MTI_SERVICE_URL)
mti_client.setExposureTime("25")
print("Connected.")

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], color='steelblue', linewidth=1.2)
ax.set_xlabel("X (scanner units)")
ax.set_ylabel("Z (scanner units, detrended)")
ax.set_title("Live Scanner Profile — 50-frame median, detrended")
ax.grid(True, linestyle='--', alpha=0.5)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.tight_layout()

frame_buffer = deque(maxlen=NUM_FRAMES)
last_coeffs  = None

def hampel(z, window=7, k=3.0):
    z = z.copy()
    h = window // 2
    for i in range(len(z)):
        seg = z[max(0, i-h): i+h+1]
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

while plt.fignum_exists(fig.number):
    x = np.array(mti_client.lineProfile.X_data, dtype=float)
    z = np.array(mti_client.lineProfile.Z_data, dtype=float)

    valid = z != 0.0
    x, z = x[valid], z[valid]

    if len(x) < 2:
        plt.pause(FRAME_DELAY)
        continue

    # Frame quality gate — reject noisy frames before they enter the buffer
    valid_f = ~np.isnan(z)
    if valid_f.sum() >= 2:
        _s, _i = np.polyfit(x[valid_f], z[valid_f], 1)
        _resid_std = np.std(z[valid_f] - (_s * x[valid_f] + _i))
        if _resid_std > FRAME_QUALITY_STD:
            plt.pause(FRAME_DELAY)
            continue

    frame_buffer.append((x, z))

    x_min, x_max = min(f[0].min() for f in frame_buffer), max(f[0].max() for f in frame_buffer)
    grid = np.linspace(x_min, x_max, NUM_BINS)
    stack = np.array([frame_to_grid(f[0], f[1], grid) for f in frame_buffer])
    z_med = np.nanmedian(stack, axis=0)

    # Mask out pen spike and linearly interpolate across it
    pen_mask = (grid >= PEN_MASK_MIN) & (grid <= PEN_MASK_MAX)
    if pen_mask.any():
        left_val  = z_med[grid < PEN_MASK_MIN][-1] if (grid < PEN_MASK_MIN).any() else np.nan
        right_val = z_med[grid > PEN_MASK_MAX][0]  if (grid > PEN_MASK_MAX).any() else np.nan
        if not np.isnan(left_val) and not np.isnan(right_val):
            z_med[pen_mask] = np.interp(grid[pen_mask],
                                         [grid[grid < PEN_MASK_MIN][-1], grid[grid > PEN_MASK_MAX][0]],
                                         [left_val, right_val])

    # Hampel filter — remove outlier spikes, then SG smooth
    z_med = hampel(z_med, window=HAMPEL_WINDOW, k=HAMPEL_K)
    valid_pts = ~np.isnan(z_med)
    if valid_pts.sum() > SAVGOL_WINDOW:
        z_med[valid_pts] = savgol_filter(z_med[valid_pts], SAVGOL_WINDOW, SAVGOL_ORDER)

    # Detrend — 2nd order polynomial fit through all data excluding pen mask region
    if DETREND_MODE == 'fixed':
        coeffs = DETREND_COEFFS
    else:
        fit_mask = ~pen_mask & ~np.isnan(z_med)
        if fit_mask.sum() >= 3:
            coeffs = np.polyfit(grid[fit_mask], z_med[fit_mask], 2)
            last_coeffs = coeffs
            print(f"Detrend coeffs: [{coeffs[0]:.6f}, {coeffs[1]:.6f}, {coeffs[2]:.6f}]", end='\r')
        else:
            coeffs = [0.0, 0.0, 0.0]
    z_med = z_med - np.polyval(coeffs, grid)

    valid_pts = ~np.isnan(z_med)
    display_pts = valid_pts & (grid >= DISPLAY_X_MIN) & (grid <= DISPLAY_X_MAX)
    if display_pts.sum() > 0:
        line.set_data(grid[display_pts], z_med[display_pts])
        ax.set_xlim(DISPLAY_X_MIN, DISPLAY_X_MAX)
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(DISPLAY_X_MIN, DISPLAY_X_MAX)
        fig.canvas.draw_idle()

    plt.pause(FRAME_DELAY)

plt.ioff()

if last_coeffs is not None:
    print(f"\nFinal detrend coeffs (paste into DETREND_COEFFS):")
    print(f"  [{last_coeffs[0]:.6f}, {last_coeffs[1]:.6f}, {last_coeffs[2]:.6f}]")

out = os.path.join(DOWNLOADS, "test_2.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close(fig)
print("Done.")

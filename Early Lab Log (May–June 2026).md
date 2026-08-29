# Early Lab Log — May 25 to June 19, 2026

Historical record of setup, troubleshooting, and early experiments before the main Progress Report began. Useful for understanding decisions made early in the project.

---

## Hardware

### Robot
ABB IRB 1200 Type-B (5/90).

**Battery Pack:** SMB board requires a 7.2V battery. OEM aftermarket packs available for $20–$50; ABB recommends their own replacement parts ($120–$350). Replacement procedure: Owner's Manual Section 3.4.1. Battery pack was ordered during week of 05/25.

**Controller backup in use:** `1200-800636_Backup_20220110` — found to be stable and runs smoothly. Eric also has a 2024 backup; tested and runs similarly, slightly smoother but marginally slower.

**Flex pendant touch calibration reset:** Hold Stop + Prog. 4 on boot.

### End Effector (EE) History
- Original EE had a 3D-printed pen holder with play/wiggle. CAD was created to replace it with an aluminum piece (week of 05/25).
- New 3D-printed piece was collected week of 06/15. Drilled at the processes shop (John). Whole EE removed and rotated 180°.
- New EE is longer than the old one — zcalib had to be updated to account for this (35mm offset added, week of 06/01 and 06/15).
- Pen tip offset for new EE: **Pft XY = 0** (pen on j6 axis). Z component needs re-measurement.

### Scanner
MTI 2D laser scanner. Determined during 06/15 week that scanning perpendicular to the channel (not parallel) is the correct orientation. Laser offset from pen tip is approximately 1 inch (~25mm) at that time — later measured and confirmed as 12.7mm (0.5").

---

## Network / IP Configuration

| Connection | IP Address | Notes |
|---|---|---|
| Lab computer (UCdevice) | 192.168.60.116 | EGM remote host — set in RobotStudio UCdevice config |
| Ahbab's laptop (UCdevice) | 192.168.60.200 | Alternative EGM remote host |
| Robot → RobotStudio (ethernet direct) | 192.168.125.10 | Subnet mask 255.255.255.0 |
| MTI Scanner | 192.168.60.200 | Set on Ethernet 2 adapter |
| ATI F/T sensor | 192.168.60.100 | Used by ATI RobotRaconteur driver |

**Common config issue (06/01):** Controller was looking for `conf1` but task name was set to `default` — controller couldn't find the task. Fixed by correcting the task name. IP was also set to default rather than the lab computer IP — fix by setting UCdevice Remote Address to `192.168.60.116` in RobotStudio → Controller → Configuration → Communication → Transmission Protocol.

---

## Known Issues & Fixes

### `mctrl.stop_egm()` not working
As of 06/01, `stop_egm()` no longer terminates the EGM session. Workaround: press the square (stop) button on the controller pendant to stop the program. Motors shut off automatically if no new program is started. E-stop shuts everything down and requires a motor reset before the next run.

### zcalib infinite loop (06/03)
zcalib requires the robot to hold within `force_epsilon` for 0.2 seconds. Increased sensor noise caused the condition to never be satisfied. Fix: raise `force_epsilon` from 0.3 → 0.5. zcalib now runs correctly.

### zcalib false triggers mid-air (06/02)
Force threshold on zcalib was too sensitive (0.3), causing it to trigger before contact. Raised to 0.8. Also allows higher approach speed so the initial jerk doesn't trigger the threshold.

### zcalib offset (06/03)
ZCalib had an offset of approximately −1.35mm. New offset set to 1.25mm.

### New EE hits rig before pushing down (06/04)
New EE is longer — on approach it contacted the rig frame before reaching the sheet. Fixed by adding a 35mm offset to zcalib approach height.

### Al 3003 high force error (05/27)
Sheet wasn't fully flush against the backing — created a localized high-force point on each pass. Robot triggered force protection after 6 lines. Ensure sheet is flat and properly clamped before running.

### Craft Al pen sticking (05/27)
Pen got stuck during forming, causing the robot to pull harder and spike the measured force. Spikes were consistent and motion-correlated. Root cause: pen play in the holder (fixed with new aluminum EE piece) and material surface interaction.

---

## Material Notes

- **Al 3003:** Tested 05/27. Too stiff for the current force settings — high force errors after 6 lines.
- **Craft aluminum (36 gauge):** Primary test material. Thinner and more compliant. Some pen sticking observed early on, resolved after tightening the EE.
- Forming uses **craft aluminum** for all current ISF work. Al 3003 is a harder alloy and requires different force settings.

---

## 3D Printer Settings (Pen Holder)

**Slicer:** Bambu Lab Studio  
**Printer:** H2D  
**Material:** Generic PLA black  
**Support:** Left extruder → Support for PLA  
CAD was converted from STL to G-code in Bambu Studio (week of 06/12).

---

## FLIR Camera Setup (Separate Sub-Project)

These notes are for the FLIR vision camera used in a separate fabric-detection sub-project, not the MTI laser scanner.

**Software:** Spinnaker SDK (download from FLIR website, account required). Open SpinView → double-click camera to connect.

**Image format:** Default pixel format is BayerRGB (saves greyscale). Change to **BGR8** under Image Format settings for RGB output.

**Python:** Use PySpin. Compatible with **Python 3.11 and earlier only**. Use `pyenv` to switch Python versions easily.

**Fabric detection script:** Takes a reference image of the blank table, then compares against a new image to detect the fabric shape and find its vertices using OpenCV edge detection. Used to locate fabric for pickup.

**Lens note:** Lab lenses are too wide-angle for close-up fabric detection. The lens on the other robot in the lab is more appropriate.

---

## Chronological Summary

| Week | Key Accomplishments |
|---|---|
| 05/25–05/29 | Calibrated robot (dead battery workaround), first Al 3003 and craft Al tests, identified pen play issue, ordered battery, lost EGM/RAPID files after network reset |
| 06/01–06/05 | Fixed controller config (conf1/default + IP), restored from backup, raised force thresholds, fixed zcalib loop, successful test run, connected MTI scanner |
| 06/08–06/12 | New pen holder CAD completed and printed, SMB battery identified (7.2V), FLIR camera setup and fabric detection script written |
| 06/15–06/19 | New EE installed and drilled, zcalib updated for new EE (35mm offset), confirmed perpendicular scan orientation, began learning EGM for scanner-following motion |

---

## Future Steps

1. **Calibrate the scanner using existing data** — use the perpendicular rescan data (16 channels, 3.25–5.0N, Sets 1 & 2) to establish the relationship between scanner readings and true channel geometry. Complete the X-axis scale calibration (tilt stretch factor) so width measurements are accurate in real mm.

2. **Create a regression model** — fit a model to the calibrated depth/width data as a function of Z penetration and applied force. This gives the first D(Z, F) input-output curve and establishes how the material responds under the current setup (craft aluminum, current backing plate).

3. **Create a predictive model for generating consistent geometries** — use the regression model as a feedforward prior. During a forming pass, take real-time scanner depth readings (accounting for the 12.7mm kinematic lag) and compute mid-pass Z corrections to converge the channel to the target geometry (0.320mm deep, 2.236mm wide at top, 0.850mm wide at base). The model should be parameterized by material and backing plate so it transfers across different designs on the same setup.

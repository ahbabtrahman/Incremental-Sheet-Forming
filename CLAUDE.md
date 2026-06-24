# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for robotic incremental sheet forming (ISF): an ABB IRB1200-5/90 arm, fitted with a pen-style forming tool and an ATI force/torque sensor, presses/draws shapes into a sheet (mounted on an iPad-like rig) under closed-loop force control, streamed over ABB EGM (Externally Guided Motion). It is a loose collection of Python research scripts, not a packaged library — there is no `setup.py`/`pyproject.toml`, no top-level `requirements.txt`, and no test suite or linter config. "Running it" means executing an individual script directly against either a real robot or the RobotStudio virtual controller in `Project1/`.

## Version control note

**This directory is not tracked by the git repo present at the drive root.** `git -C R:/ ls-files` shows the repo at `R:\` only tracks an unrelated small ROS2 workspace (`src/object_detection`, `src/tracking_control` — a separate car-tracking project). Nothing under `Research - Incremental Sheet Forming\Projects` is in git history. Don't assume `git log`/`git blame` here will return anything relevant, and don't commit changes in this directory expecting them to land in that ROS2 repo's history unless the user explicitly asks to start tracking this folder.

## Directory layout

- **Top-level `*.py` files** — the live, actively-edited scripts (modification dates run up to the present). This is where changes should be made.
- **`ati_robotraconteur_driver-main/`** — vendored Robot Raconteur driver service for the ATI F/T sensor. Runs as its own standalone process; execution scripts connect to it as a client (see below). Has its own `README.md`/`requirements.txt` (numpy, beautifulsoup4, requests).
- **`Project1/`** — an ABB RobotStudio project (`Project1.rsproj`, `Station/*.rsstnx`, `Controller Data/IRB1200_5_90/`, `Virtual Controllers/`). Opened with the RobotStudio desktop app (Windows) to run a virtual IRB1200_5_90 controller for offline/safe testing before touching the real robot. Don't hand-edit the RAPID/controller backup files unless intentionally modifying the virtual controller.
- **`Sheet-Metal-Deformation-Research-main/`** — a frozen archival snapshot, extracted from the sibling `Sheet-Metal-Deformation-Research-main.zip`. Every file under it carries the same extraction timestamp, confirming it's a point-in-time copy of an earlier repo state (its `SM MV/` subfolder duplicates older versions of many top-level scripts; `Sheet_Metal_Robotics_molly/` has separate MTI laser-scanner scripts). Treat it as reference only — the top-level files are the ones to edit.

## Dependencies (inferred from imports — nothing is pinned)

`numpy`, `scipy`, `matplotlib`, `pyyaml`, `qpsolvers`, `general_robotics_toolbox` (RPI robotics toolbox, including its `general_robotics_toolbox.robotraconteur` submodule for loading robot YAML defs), `RobotRaconteur`, `abb_robot_client` (specifically `abb_robot_client.egm.EGM`), `abb_motion_program_exec`. The ATI driver subprocess additionally needs `beautifulsoup4` and `requests` (its own `requirements.txt`).

## Architecture

### Shared modules (top level)

- **`robot_def.py`** — `robot_obj`/`positioner_obj`: kinematics wrapper around `general_robotics_toolbox`. Loads a robot definition YAML (e.g. `ABB_1200_5_90_robot_default_config.yml`), an optional tool-transform CSV, an optional base-transform CSV, and an optional mocap marker-calibration YAML. Provides `fwd()`/`inv()`/`jacobian()`/`find_curve_js()`.
- **`utils.py`** — shared math/geometry helpers: homogeneous transforms, plane fitting, curve filtering/smoothing/outlier removal, `force_prop`/`adjoint_map` (propagate a wrench between frames), plotting helpers.
- **`lambda_calc.py`** — arc-length ("lambda") parametrization of a Cartesian or joint-space path (`calc_lam_cs`/`calc_lam_js`), used to convert a geometric path into something speed/time can be assigned to.
- **`traj_gen.py`** — synthetic test trajectory generator (`get_trajectory`): a parabolic ISF path with surface normals, used by the simpler example/test scripts instead of a real planned path.
- **`motion_toolbox.py`** — minimal legacy `jog_joint()` velocity helper, superseded by `RobotMotionController.MotionController` for anything nontrivial.

### `RobotMotionController.py` — the central execution engine

`MotionController` wraps either ABB EGM (`abb_robot_client.egm.EGM` + `abb_motion_program_exec.MotionProgramExecClient` talking HTTP to RobotStudio or the real controller) or a Robot Raconteur robot driver (`USE_RR_ROBOT=True`) for joint position streaming, plus a Robot Raconteur wire subscription to the ATI F/T sensor service for real-time force feedback. The constructor blocks until the first force reading arrives, so the ATI driver (see below) must already be running for any non-`simulation` use.

Key methods:
- `position_cmd`/`read_position` — single EGM step send/receive.
- `trajectory_generate` — resamples a joint path onto the EGM timestep grid (`TIMESTEP`, ~4ms) with a trapezoidal velocity/acceleration profile.
- `trajectory_force_PIDcontrol` — streams a path while closing a Z-axis force-impedance loop against per-waypoint force setpoints.
- `force_load_z` — touch-and-hold a target normal force (used to make first contact with the sheet).
- `motion_start_routine`/`motion_end_routine`/`press_button_routine` — rig-specific choreography referencing `ipad_pose`/`H_pentip2ati`, the calibrated frame transforms between the robot, the forming rig, and the F/T sensor.

Everything is built on the EGM control loop cadence (`TIMESTEP`, ~4ms / 250Hz).

### Experiment/execution scripts (top level)

Common pattern: load the robot YAML → build a `robot_obj` → either load a precomputed path from CSV (e.g. `curve_js.csv`, `fullruntraj_q.csv`) or generate one via `traj_gen`/`path_planning_jacobian.py` → instantiate `MotionController` pointed at a robot IP and the ATI Robot Raconteur service → stream the trajectory, logging `(time, force, joint angles)` and/or `(time, force, xyz)` to CSV for offline matplotlib plotting at the bottom of the same script.

Rough groupings by name:
- `drawStar*.py`, `SquareTest.py`, `L-Path.py`, `pointtest.py` — simple path-tracing tests without force control.
- `*FT*.py`, `*egm*.py`, `sheetmetalrobot*.py` — combine motion with closed-loop force sensing/control.
- `calibration.py`, `zcalib.py`/`zcalibration.py`, `improvedzcalibration.py` — compute the rig-to-robot frame transform from probed corner points (`calibrate()` in `calibration.py` is imported by several other scripts, e.g. `rapidmode.py`), producing `rig_pose*.csv`.
- `mti_snapshot.py`, `mtiscanfullsheet.py` — drive an MTI laser scanner to capture sheet/part geometry (most recently active files as of this writing).
- `rS_example.py`, `egm_joint_target_example.py`, `example.py` — minimal reference usage of `abb_motion_program_exec`/EGM, not part of the experiment data pipeline.
- `challengeTask.py` — currently the most actively maintained "main"-style script; a reasonable template for the current calling convention. Note its embedded warnings (e.g. "UPDATE P2 BEFORE ACTUALLY RUNNING") about hand-tuned waypoints that must be checked before any real-robot run.

### ATI F/T sensor driver (`ati_robotraconteur_driver-main/`)

A standalone Robot Raconteur service: start it with `python robotraconteur_ati_driver.py --sensor-ip=<Sensor IP>` before running any script that does force sensing. Execution scripts connect to it as a client at `rr+tcp://localhost:59823?service=ati_sensor`. `rpi_ati_net_ft.py` implements the underlying sensor network protocol.

### ABB RobotStudio project (`Project1/`)

Opened in RobotStudio to run a virtual IRB1200_5_90 controller. `Project1/Controller Data/IRB1200_5_90/RAPID/TASK1/PROGMOD/motion_program_exec_egm.mod` is the RAPID-side EGM routine the Python EGM client talks to; `motion_program_exec.mod` handles non-EGM moves; `TASK2/PROGMOD/motion_program_logger.mod` and `TASK3/PROGMOD/error_reporter.mod` run as separate RAPID tasks. `Virtual Controllers/` holds saved virtual-controller state/backups — these are RobotStudio-managed, not meant for manual editing.

### Data flow

Robot YAML + calibration CSVs → planned path (`curve_js.csv`, `lam_planned.csv`, etc.) → `MotionController.trajectory_generate` resamples to the EGM grid → real-time streaming with optional force feedback → telemetry CSV (`time, force, joint angles` / `time, force, xyz`) → ad hoc matplotlib plotting inline at the end of the same script (no separate analysis pipeline/notebook).

## Things to check before running a script against hardware

- **Robot IP** is hardcoded per-script (several scripts use `192.168.60.101`, sometimes flagged with a comment like "Change IP at Line ..."). Confirm it points where you intend (real IRB1200_5_90 vs. RobotStudio virtual controller) before executing.
- **ATI sensor driver must already be running** and reachable at `rr+tcp://localhost:59823?service=ati_sensor` — `MotionController.__init__` blocks indefinitely waiting on the first force reading otherwise.
- **Hand-tuned waypoints/tool offsets** (e.g. `Pft`, corner points `c1`/`c2`/`c3`, `P2` in `challengeTask.py`) are calibrated per physical rig setup and must be re-verified, not assumed, before a real-robot run — several scripts have explicit comments to this effect.
- A few scripts (e.g. `challengeTask.py`) carry a `sys.path.append("/home/fusing-ubuntu/Sheet-Metal-Deformation-Research/SM MV/")` left over from the lab's Linux control-PC deployment path; it's a harmless no-op when running from this Windows directory since the needed modules already sit alongside the script.

## Conventions

- Indentation is inconsistent file-to-file (tabs in older modules like `robot_def.py`/`utils.py`, spaces in newer ones like `RobotMotionController.py`) — match the existing file's style rather than normalizing it.
- No automated tests exist. "Verification" means running the relevant script against the RobotStudio virtual controller (`Project1/`) first, then cautiously against the real robot.

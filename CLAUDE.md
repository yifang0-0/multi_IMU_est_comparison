# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-IMU joint angle estimation comparison project for knee and ankle angles from IMU sensor data. Implements and benchmarks multiple estimation methods (VQF+Olsson, Heading Correction, Kalman Filter with gravity frame) against OpenSense algorithms and motion capture ground truth.

## Commands

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download dataset (~5.5GB from SimTK)
python download_simtk_dataset.py

# Run estimation
python run_estimation.py --joint knee --method all --subject Subject08
python run_estimation.py --joint ankle --method vqf_olsson
python run_estimation.py --joint knee --method kf_gframe_olsson --no-plot
```

Available methods: `vqf_olsson`, `vqf_olsson_heading_correction`, `opensense`, `kf_gframe_olsson`, `kf_gframe_optimized`, `vqf_ik`, `all`

## Architecture

### Core Modules

- **`run_estimation.py`** - Unified entry point for knee/ankle estimation with `--joint` and `--method` args
- **`utils.py`** - Data loading (`load_imu_data`), orientation estimation (`estimate_orientations`), signal alignment (`find_best_shift`, `compute_raw_signal_offset`), OpenSense result loading
- **`calTools.py`** - Mathematical utilities: quaternion operations (`quatmultiply`, `quatconj`, `integrateGyr`), Jacobians (`dLnk`, `dMotion`), filters (`lowpass_filter`), derivatives (`approx_derivative`)
- **`constants.py`** - Physical constants (G, FS) and KF covariance defaults
- **`plotting.py`** - Plotting functions (`plot_time_series_error`, `plot_error_comparison`)

### Estimation Methods (`methods/` module)

Each method is a standalone function returning angle in degrees:

- **`methods/vqf_olsson.py`** - VQF orientation + Olsson hinge joint axis estimation; also contains `run_vqf_olsson_heading_corrected` which adds qmt heading correction
- **`methods/kf_gframe.py`** - Kalman filter with gravity frame constraints (quaternion helpers, Jacobian computation, EKF implementation, lever arm estimation)
- **`methods/shared.py`** - Common utilities: `load_mot()` for OpenSim files, `olsson_estimate_hinge_joint_axes()`, `calculate_joint_angle()`

**Method Categories:**
- **IMU methods** (`vqf_olsson`, `vqf_olsson_heading_correction`, `kf_gframe_olsson`, `kf_gframe_optimized`): Compute from raw sensor data
- **Precomputed methods** (`opensense`, `vqf_ik`): Load results from `IMU/{algo}/IKResults/`

## Key Libraries

- **qmt** - Quaternion math toolbox: `qmt.oriEstVQF`, `qmt.jointAxisEstHingeOlsson`, `qmt.headingCorrection`, `qmt.qmult`, `qmt.qinv`

## Data Conventions

- **Quaternion format**: w, x, y, z (qmt convention)
- **Array shapes**: Often (3, N) for sensor data, (N, 4) for quaternions
- **Sampling rate**: 100 Hz throughout
- **Angles**: Degrees for display/RMSE, radians for computation
- **Cross-correlation alignment**: Used for phase correction before error calculation
- **Offset caching**: `offsets.json` stores computed IMU-Mocap alignment offsets; regenerates on first run if missing

## Data Structure

Expected in `data/Subject08/`:
- `walking/IMU/xsens/LowerExtremity/*.txt` - Raw IMU data (TSV at 100 Hz)
- `walking/IMU/myIMUMappings_walking.xml` - Sensor-to-segment mappings
- `walking/Mocap/ikResults/walking_IK.mot` - Ground truth from motion capture

## Code Style

- **Concise over verbose**: Minimize boilerplate; use libraries (e.g., `tqdm`) over manual implementations
- **Readable over clever**: Prefer `1024 * 1024` over `1 << 20`
- **Self-descriptive names**: Function names should convey purpose (e.g., `download_and_extract_simtk_dataset`)
- **One-liner docstrings**: Brief documentation is preferred
- **Importable design**: Write functions that can be called from other modules

## Excluded Subjects

The following subjects are excluded from analysis due to data issues:

- **Subject01**: Poor data quality (96° RMSE); foot sensor not mapped
- **Subject06**: IMU data at 40 Hz instead of 100 Hz
- **Subject09**: Inverted IMU attachment; foot sensor (calcn_r_imu) not mapped
- **Subject10**: IMU data at 40 Hz instead of 100 Hz
- **Subject11**: Foot sensor (calcn_r_imu) not mapped; malformed XML mapping file

Valid subjects: Subject02, Subject03, Subject04, Subject07, Subject08

## Git Preferences

- Do NOT add `Co-Authored-By` lines to commit messages

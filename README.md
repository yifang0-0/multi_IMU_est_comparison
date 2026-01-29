# Multi-IMU Joint Angle Estimation Comparison

A benchmarking framework for comparing IMU-based joint angle estimation methods against motion capture ground truth. Implements and evaluates multiple algorithms for knee and ankle angle estimation from dual-IMU sensor configurations.

## Methods Implemented

| Method | Description |
|--------|-------------|
| `vqf_olsson` | VQF orientation estimation + Olsson hinge joint axis estimation |
| `vqf_olsson_heading_correction` | VQF+Olsson with qmt heading drift correction |
| `kf_gframe_olsson` | Extended Kalman Filter with gravity frame constraints + Olsson joint axis |
| `kf_gframe_optimized` | KF with gravity frame + optimized joint axis (uses ground truth for calibration) |
| `opensense` | OpenSense IK results (Xsens, Madgwick, Mahony algorithms) |
| `vqf_opensim` | Precomputed VQF-OpenSim results |

## Results Summary

**Knee Joint RMSE (degrees)** - Subject08 representative results:

| Method | RMSE |
|--------|------|
| vqf_olsson_heading_correction | **2.20** |
| kf_gframe_optimized | 2.54 |
| Madgwick | 4.75 |
| kf_gframe_olsson | 5.01 |
| Mahony | 5.57 |
| Xsens | 5.99 |
| VQF-OpenSim | 13.06 |

The heading-corrected VQF+Olsson method achieves excellent performance after proper quaternion handling.

## Setup

```bash
# Create environment (using uv)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download dataset (~5.5GB from SimTK)
python download_simtk_dataset.py
```

## Usage

```bash
# Run all methods on a single subject
python run_estimation.py --joint knee --subject Subject08

# Run specific method
python run_estimation.py --joint ankle --method kf_gframe_olsson

# Run on all valid subjects with parallel processing
python run_estimation.py --joint knee --method all --subject all --workers 4

# Disable interactive plots
python run_estimation.py --joint knee --no-plot
```

**Arguments:**
- `--joint`: `knee` or `ankle`
- `--method`: `vqf_olsson`, `vqf_olsson_heading_correction`, `kf_gframe_olsson`, `kf_gframe_optimized`, `opensense`, `vqf_opensim`, or `all`
- `--subject`: `Subject01`-`Subject11` or `all`
- `--workers`: Number of parallel workers (default: CPU count)
- `--no-plot`: Save plots without displaying

## Data Structure

Data is expected in `data/SubjectXX/`:
```
data/Subject08/
├── walking/
│   ├── IMU/
│   │   ├── xsens/LowerExtremity/*.txt    # Raw IMU data (100 Hz)
│   │   ├── myIMUMappings_walking.xml     # Sensor-to-segment mappings
│   │   ├── xsens/IKResults/              # OpenSense Xsens results
│   │   ├── madgwick/IKResults/           # OpenSense Madgwick results
│   │   └── mahony/IKResults/             # OpenSense Mahony results
│   └── Mocap/
│       └── ikResults/walking_IK.mot      # Motion capture ground truth
```

## Project Structure

```
├── run_estimation.py      # Unified entry point
├── utils.py               # Data loading, signal alignment
├── calTools.py            # Quaternion operations, Jacobians, filters
├── constants.py           # Physical constants
├── plotting.py            # Visualization
├── methods/
│   ├── vqf_olsson.py      # VQF+Olsson and heading correction
│   ├── kf_gframe.py       # Kalman filter implementation
│   └── shared.py          # Common utilities (Olsson, angle calculation)
├── plots/                 # Generated comparison plots
└── results/               # RMSE summary CSVs
```

## Valid Subjects

5 subjects are valid for analysis: Subject02, Subject03, Subject04, Subject07, Subject08

**Excluded subjects:**
- Subject01, Subject11: Missing data files
- Subject05: Malformed XML (nested comments)
- Subject06, Subject10: IMU data at 40 Hz (expected 100 Hz)
- Subject09: Inverted IMU attachment

## Key Libraries

- [qmt](https://github.com/dlaidig/qmt) - Quaternion math toolbox (VQF, Olsson, heading correction)
- numpy, scipy - Numerical processing
- matplotlib - Visualization

## Implementation Notes

- **Heading Correction**: Uses `qmt.headingCorrection` with the Olsson-estimated joint axis and swing-twist decomposition (`qmt.quatProject`) for proper angle extraction
- **KF Gravity Frame**: Constrains both IMU orientations to share a common gravity direction, with auto-estimated lever arms
- **Signal Alignment**: Cross-correlation used to align IMU estimates with motion capture ground truth

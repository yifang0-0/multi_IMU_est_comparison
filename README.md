# Multi-IMU Joint Angle Estimation Comparison

A benchmarking framework for comparing IMU-based joint angle estimation methods against motion capture ground truth. Implements and evaluates multiple algorithms for knee and ankle angle estimation from dual-IMU sensor configurations.

## Methods Implemented

| Method | Description |
|--------|-------------|
| `kf_gframe` | Extended Kalman Filter with gravity frame constraints and auto-estimated lever arms |
| `vqf_olsson` | VQF orientation estimation + Olsson hinge joint axis estimation |
| `vqf_olsson_heading_correction` | VQF+Olsson with qmt heading drift correction (experimental) |
| `opensense` | OpenSense IK results (Xsens, Madgwick, Mahony algorithms) |
| `vqf_opensim` | Precomputed VQF-OpenSim results |

## Results Summary

**Knee Joint RMSE (degrees)** across 7 subjects:

| Method | Mean RMSE |
|--------|-----------|
| kf_gframe | **3.15** |
| vqf_opensim | 5.50 |
| Mahony | 5.98 |
| Madgwick | 6.42 |
| Xsens | 18.26 |
| vqf+olsson+heading_correction | 19.98 |
| vqf+olsson | 86.40 |

The Kalman Filter method (`kf_gframe`) achieves the best performance for knee angle estimation.

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
python run_estimation.py --joint ankle --method kf_gframe

# Run on all valid subjects with parallel processing
python run_estimation.py --joint knee --method all --subject all --workers 4

# Disable interactive plots
python run_estimation.py --joint knee --no-plot
```

**Arguments:**
- `--joint`: `knee` or `ankle`
- `--method`: `vqf_olsson`, `vqf_olsson_heading_correction`, `opensense`, `kf_gframe`, `vqf_opensim`, or `all`
- `--subject`: `Subject01`-`Subject11` or `all`
- `--workers`: Number of parallel workers (default: 4)
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
│   ├── vqf_olsson.py      # VQF+Olsson implementation
│   ├── kf_gframe.py       # Kalman filter implementation
│   └── shared.py          # Common utilities
├── plots/                 # Generated comparison plots
└── results/               # RMSE summary CSVs
```

## Valid Subjects

7 subjects are valid for analysis: Subject01, Subject02, Subject03, Subject04, Subject07, Subject08, Subject11

**Excluded subjects:**
- Subject05, Subject09: Malformed XML
- Subject06, Subject10: IMU data at 40 Hz (expected 100 Hz)

## Key Libraries

- [qmt](https://github.com/dlaidig/qmt) - Quaternion math toolbox (VQF, Olsson, heading correction)
- numpy, scipy - Numerical processing
- matplotlib - Visualization

## Known Issues

- **VQF+Olsson**: Poor performance, especially on ankle joint; under investigation
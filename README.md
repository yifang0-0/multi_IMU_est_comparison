# Joint Angle Estimation Scripts

This project contains Python scripts for estimating knee and ankle joint angles using IMU data. It implements various methods including VQF orientation estimation, Olsson joint axis estimation, and Heading Correction. It also allows comparison with OpenSense results.

## Environment Setup

The scripts are designed to run in a Python 3 environment. A virtual environment is recommended.

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate  # On Windows
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Structure

The scripts expect the following data structure in the root directory:

-   `Subject08/`: Contains the subject data.
    -   `walking/IMU/xsens/LowerExtremity/`: Raw IMU data (`.txt` files).
    -   `walking/IMU/myIMUMappings_walking.xml`: Sensor mappings.
    -   `walking/Mocap/ikResults/walking_IK.mot`: Ground truth motion capture data.
    -   `walking/IMU/[algo]/IKResults/...`: OpenSense results for comparison.

## Scripts

### 1. `run_estimation_knee.py`
Estimates the **knee** joint angle.

**Key Functions:**
-   `estimate_joint_axes`: Estimates the knee joint axis using the Olsson method.
-   `run_vqf_olsson`: Calculates knee angle using VQF orientation and the estimated axis.
-   `run_vqf_olsson_heading_correction`: Applies heading correction to the VQF+Olsson estimate (currently experimental/tuning).
-   `run_opensense`: Loads and plots OpenSense results for comparison.

### 2. `run_estimation_ankle.py`
Estimates the **ankle** joint angle.

**Key Functions:**
-   Similar structure to the knee script, but adapted for Tibia and Calcaneus segments.

## Usage

Run the scripts from the command line. You can specify which method(s) to run and whether to show plots.

**Basic Usage:**
```bash
python run_estimation_knee.py
python run_estimation_ankle.py
```

**Options:**
-   `--method`: Choose the estimation method.
    -   `vqf_olsson`: Run VQF + Olsson.
    -   `heading_correction`: Run Heading Correction (experimental).
    -   `opensense`: Run OpenSense comparison.
    -   `vqf_olsson_heading_correction`: Run VQF + Olsson + Heading Correction.
    -   `all`: Run all methods (default).
-   `--no-plot`: Disable interactive plotting (plots are still saved to `plots/`).

**Examples:**
```bash
# Run only VQF + Olsson method for knee
python run_estimation_knee.py --method vqf_olsson

# Run all methods for ankle without opening plot windows
python run_estimation_ankle.py --no-plot
```

## Current Status & Known Issues

-   **Heading Correction**: The `heading_correction` method is implemented but may require parameter tuning (`tauDelta`, `tauBias`) for optimal performance. It currently might show over-correction or drift in some cases.
-   **Olsson Method**: The joint axis estimation (Olsson) is working but performance depends on the quality of the motion data (excitation).
    -   **TODO**: The Olsson method currently does not perform well for the **ankle** joint. This needs further evaluation or a potential fix.
-   **Synchronization**: There is an unknown time offset between the IMU and Ground Truth (Mocap) data. Currently, a cross-correlation approach is used to align the signals, but a systematic synchronization method using timestamps is missing.


"""Utility functions for IMU joint angle estimation (data loading, signal alignment)."""
import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.signal import correlate, butter, filtfilt
import qmt

from methods.shared import load_mot
from constants import FS


# =============================================================================
# Data Loading
# =============================================================================

def load_imu_data(file_path):
    """Load raw IMU data from .txt file (Xsens format)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('PacketCounter'):
            header_line = i
            data_start = i + 1
            break

    if header_line is None:
        raise ValueError(f"Could not find header in {file_path}")

    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=data_start,
        names=lines[header_line].strip().split('\t')
    )
    return df


def get_sensor_mappings(xml_path):
    """Parse XML to get sensor mappings {body_part: sensor_id}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mappings = {}
    sensors = root.find('.//ExperimentalSensors')
    if sensors is not None:
        for sensor in sensors.findall('ExperimentalSensor'):
            name = sensor.get('name')
            model_name = sensor.find('name_in_model').text
            mappings[model_name] = name

    return mappings


def load_opensense_results(subject_path, gt_column='knee_angle_r', algorithm=None):
    """Load pre-calculated results from OpenSense algorithms.

    Args:
        subject_path: Path to subject data directory
        gt_column: Column name to extract (default: 'knee_angle_r')
        algorithm: Single algorithm ('xsens', 'madgwick', 'mahony') or None for all

    Returns:
        dict mapping algorithm name to angle values (np.ndarray)
    """
    from pathlib import Path
    subject_path = Path(subject_path)
    algos = [algorithm] if algorithm else ['xsens', 'madgwick', 'mahony']
    results = {}

    for algo in algos:
        if algo not in ('xsens', 'madgwick', 'mahony'):
            print(f"Unknown algorithm: {algo}")
            continue

        path = subject_path / 'IMU' / algo / 'IKResults' / 'IKWithErrorsUniformWeights' / 'walking_IK.mot'
        if not path.exists():
            path = subject_path / 'IMU' / algo / 'IKResults' / 'IKWithErrorsExtremeLowFeetWeights' / 'walking_IK.mot'

        if not path.exists():
            print(f"OpenSense results not found for {algo}")
            continue

        df = load_mot(path)
        if gt_column not in df.columns:
            print(f"Column '{gt_column}' not found in {path}")
            continue

        results[algo] = df[gt_column].values

    return results


# =============================================================================
# Orientation Estimation
# =============================================================================

def estimate_orientations(acc, gyr, fs):
    """Estimate orientation using VQF without magnetometer."""
    q = qmt.oriEstVQF(gyr, acc, mag=None, params={'Ts': 1.0/fs})
    return q[0] if isinstance(q, tuple) else q


# =============================================================================
# Signal Alignment
# =============================================================================

def find_best_shift(est_signal, gt_signal):
    """Find best alignment offset between two signals using cross-correlation.

    Returns (offset, correlation) where offset is for use with align_signals
    (positive = trim gt, negative = trim est).
    """
    if np.std(est_signal) == 0 or np.std(gt_signal) == 0:
        return 0, 0.0
    if np.isnan(est_signal).any() or np.isnan(gt_signal).any():
        return 0, 0.0

    sig1 = (est_signal - np.mean(est_signal)) / (np.std(est_signal) + 1e-6)
    sig2 = (gt_signal - np.mean(gt_signal)) / (np.std(gt_signal) + 1e-6)

    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig2) + 1, len(sig1))
    best_idx = np.argmax(corr)
    peak_corr = corr[best_idx] / min(len(sig1), len(sig2))
    # Negate to match align_signals convention (positive = trim gt)
    return -int(lags[best_idx]), float(peak_corr)


def align_signals(est_signal, gt_signal, shift):
    """Align signals based on calculated shift.

    Note: Uses legacy sign convention where negative shift means est starts before gt.
    """
    if shift > 0:
        common_len = min(len(est_signal), len(gt_signal) - shift)
        return est_signal[:common_len], gt_signal[shift:shift+common_len]
    elif shift < 0:
        s = -shift
        common_len = min(len(est_signal) - s, len(gt_signal))
        return est_signal[s:s+common_len], gt_signal[:common_len]
    else:
        common_len = min(len(est_signal), len(gt_signal))
        return est_signal[:common_len], gt_signal[:common_len]


def _lowpass_filter(data, cutoff=5.0, fs=FS, order=4):
    """Low-pass filter for alignment signals (internal helper)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def _approx_derivative(y, fs=FS):
    """Compute numerical derivative using 5-point central difference (internal helper)."""
    dy = np.zeros_like(y)
    dy[:, 2:-2] = (y[:, :-4] - 8 * y[:, 1:-3] + 8 * y[:, 3:-1] - y[:, 4:]) * (fs / 12)
    return dy


def validate_offset(offset, imu_len, mocap_len, min_overlap_fraction=0.5):
    """Check if offset produces physically plausible overlap.

    Args:
        offset: Computed alignment offset (positive = IMU starts before mocap)
        imu_len: Length of IMU signal in samples
        mocap_len: Length of mocap signal in samples
        min_overlap_fraction: Minimum required overlap as fraction of shorter signal

    Returns:
        tuple: (is_valid, overlap_samples, message)
    """
    if offset > 0:  # IMU starts before mocap
        overlap = min(imu_len - offset, mocap_len)
    else:  # Mocap starts before IMU
        overlap = min(imu_len, mocap_len + offset)

    min_overlap = min_overlap_fraction * min(imu_len, mocap_len)

    if overlap < 0:
        return False, 0, "Offset produces no overlap"
    if overlap < min_overlap:
        return False, overlap, f"Overlap ({overlap}) below minimum ({min_overlap:.0f})"

    return True, overlap, f"Valid overlap: {overlap} samples ({overlap/FS:.1f} sec)"


def compute_raw_signal_offset(subject_path, fs=FS, method='pelvis_gyr_z'):
    """Compute alignment offset using pelvis gyroscope (joint-independent).

    This method correlates IMU pelvis gyroscope with mocap pelvis angular velocity.
    Uses full cross-correlation to search all possible lags.

    Args:
        subject_path: Path to subject data directory (e.g., 'data/Subject08/walking')
        fs: Sampling frequency
        method: Signal pair to use:
            - 'pelvis_gyr_z': pelvis_Gyr_Z vs d/dt[pelvis_rotation] (best correlation ~0.87)
            - 'pelvis_gyr_mag': ||pelvis_gyr|| vs ||d/dt[pelvis_euler]|| (original)

    Returns:
        tuple: (offset, correlation, error_message)
            offset: Alignment offset (negative = IMU starts before mocap in legacy convention)
            correlation: Peak normalized correlation value
            error_message: None if successful, error string otherwise
    """
    subject_path = Path(subject_path)
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    mocap_path = subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot'
    mapping_path = subject_path / 'IMU' / 'myIMUMappings_walking.xml'

    # Load sensor mappings
    try:
        mappings = get_sensor_mappings(mapping_path)
    except Exception as e:
        return None, 0.0, f"Failed to load mappings: {e}"

    # Get pelvis sensor ID
    pelvis_sensor = mappings.get('pelvis_imu', '').lstrip('_')
    if not pelvis_sensor:
        return None, 0.0, "pelvis_imu not found in mappings"

    # Load pelvis IMU data
    try:
        pelvis_files = list(imu_dir.glob(f"*{pelvis_sensor}.txt"))
        if not pelvis_files:
            return None, 0.0, f"Pelvis IMU file not found for sensor {pelvis_sensor}"
        pelvis_df = load_imu_data(pelvis_files[0])
    except Exception as e:
        return None, 0.0, f"Failed to load pelvis IMU: {e}"

    # Load mocap data
    try:
        mocap_df = load_mot(mocap_path)
    except Exception as e:
        return None, 0.0, f"Failed to load mocap: {e}"

    pelvis_gyr = pelvis_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

    if method == 'pelvis_gyr_z':
        # Best method: pelvis_Gyr_Z vs d/dt[pelvis_rotation] (correlation ~0.87)
        imu_signal = _lowpass_filter(pelvis_gyr[:, 2], cutoff=5.0, fs=fs)

        if 'pelvis_rotation' not in mocap_df.columns:
            return None, 0.0, "pelvis_rotation column not found in mocap"

        pelvis_rot = np.deg2rad(mocap_df['pelvis_rotation'].values)
        mocap_signal = _lowpass_filter(np.gradient(pelvis_rot) * fs, cutoff=5.0, fs=fs)

    else:  # pelvis_gyr_mag (original method)
        imu_signal = _lowpass_filter(np.linalg.norm(pelvis_gyr, axis=1), cutoff=5.0, fs=fs)

        euler_cols = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
        if not all(c in mocap_df.columns for c in euler_cols):
            return None, 0.0, "Pelvis Euler columns not found in mocap"

        euler = np.deg2rad(mocap_df[euler_cols].values.T)  # (3, N)
        d_euler = _approx_derivative(euler, fs=fs)  # (3, N)
        mocap_signal = _lowpass_filter(np.linalg.norm(d_euler, axis=0), cutoff=5.0, fs=fs)

    # Compute offset with full correlation (searches all lags)
    offset, corr = find_best_shift(imu_signal, mocap_signal)

    # Validate offset (use negated offset for validation since find_best_shift already negates)
    is_valid, overlap, msg = validate_offset(-offset, len(imu_signal), len(mocap_signal))
    if not is_valid:
        return None, corr, f"Invalid offset: {msg}"

    return offset, corr, None


# =============================================================================
# Metrics
# =============================================================================

def calculate_rmse(estimated, ground_truth):
    """Calculate root mean squared error."""
    n = min(len(estimated), len(ground_truth))
    return np.sqrt(np.mean((ground_truth[:n] - estimated[:n])**2))


def gyro_magnitude(gyr):
    """Compute gyroscope magnitude, handling (3, N) or (N, 3) shapes."""
    return np.linalg.norm(gyr, axis=0) if gyr.shape[0] == 3 else np.linalg.norm(gyr, axis=1)


# =============================================================================
# VQF-OpenSim Results
# =============================================================================

def find_vqf_opensim_file(subject_id):
    """Find VQF-OpenSim .mot file for subject. Returns Path or None."""
    # Extract number from subject_id (e.g., 'Subject02' -> '2', 'Subject08' -> '8')
    import re
    match = re.search(r'(\d+)', subject_id)
    if not match:
        return None
    num = str(int(match.group(1)))  # Remove leading zeros
    pattern = f'subject{num}'
    for f in Path('vqf_opensim_results').glob('*.mot'):
        if f.stem.lower().startswith(pattern):
            return f
    return None


def load_offset(method, subject_id, gt_column):
    """Load cached offset from JSON. Returns int or None."""
    path = Path('offsets.json')
    if path.exists():
        offsets = json.loads(path.read_text())
        return offsets.get(f"{method}_{subject_id}_{gt_column}")
    return None


def save_offset(method, subject_id, gt_column, offset):
    """Save offset to JSON cache."""
    path = Path('offsets.json')
    offsets = json.loads(path.read_text()) if path.exists() else {}
    offsets[f"{method}_{subject_id}_{gt_column}"] = int(offset)  # Convert numpy int to Python int
    path.write_text(json.dumps(offsets, indent=2))
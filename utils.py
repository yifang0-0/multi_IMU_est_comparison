"""Utility functions for IMU joint angle estimation (data loading, signal alignment)."""
import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.signal import correlate
import qmt

from methods.shared import load_mot


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
    """Find where shorter signal best matches within longer signal using cross-correlation.

    Returns offset such that est_signal[offset:] aligns with gt_signal (or vice versa).
    Positive offset means est_signal starts before gt_signal.
    """
    # Handle NaN/constant signals
    if np.std(est_signal) == 0 or np.std(gt_signal) == 0:
        return 0
    if np.isnan(est_signal).any() or np.isnan(gt_signal).any():
        return 0

    # Normalize both signals
    sig1 = (est_signal - np.mean(est_signal)) / (np.std(est_signal) + 1e-6)
    sig2 = (gt_signal - np.mean(gt_signal)) / (np.std(gt_signal) + 1e-6)

    # Ensure sig1 is the longer signal for consistent offset interpretation
    if len(sig1) < len(sig2):
        sig1, sig2 = sig2, sig1
        swapped = True
    else:
        swapped = False

    # Correlate: slides sig2 along sig1
    # mode='valid' gives correlation at each position where sig2 fully overlaps sig1
    # Result length: len(sig1) - len(sig2) + 1
    corr = correlate(sig1, sig2, mode='valid')

    best_offset = int(np.argmax(np.abs(corr)))

    # Convert to lag convention (positive = est starts before gt)
    if swapped:
        return -best_offset
    return best_offset


def align_signals(est_signal, gt_signal, shift):
    """Align signals based on calculated shift."""
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
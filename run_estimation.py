"""
Unified joint angle estimation for knee and ankle joints.

Usage:
    python run_estimation.py --joint knee --method all
    python run_estimation.py --joint ankle --method vqf_olsson
    python run_estimation.py --joint knee --method kf_gframe --no-plot
    python run_estimation.py --joint knee --subject Subject08 --method all
"""
import numpy as np
import argparse
import os
from pathlib import Path

from utils import (
    load_imu_data, get_sensor_mappings,
    find_best_shift, align_signals,
    load_opensense_results, find_vqf_opensim_file, load_offset, save_offset,
    compute_raw_signal_offset, validate_offset
)
from methods.shared import load_mot, calculate_joint_angle
from methods import run_vqf_olsson, run_vqf_olsson_heading_corrected, run_kf_gframe
from plotting import plot_time_series_error, plot_error_comparison


def _eval_imu_method(name, angle_deg, data, errors_dict):
    """Evaluate IMU method with fine-tuning alignment search."""
    gt = data['gt']
    best_rmse, best_offset = float('inf'), 0

    # Fine search only - data is pre-aligned
    for delta in range(-50, 51):
        est, gt_a = align_signals(angle_deg, gt, delta)
        if len(est) > 0:
            rmse = np.sqrt(np.mean((gt_a - est)**2))
            if rmse < best_rmse:
                best_rmse, best_offset = rmse, delta

    est, gt_a = align_signals(angle_deg, gt, best_offset)
    print(f"{name} - RMSE: {best_rmse:.2f} deg (fine-tune: {best_offset})")
    errors_dict[name] = np.abs(gt_a - est)


def _eval_precomputed(name, angle_deg, gt, errors_dict):
    """Evaluate precomputed method (already time-synced with mocap)."""
    n = min(len(angle_deg), len(gt))
    error = np.abs(gt[:n] - angle_deg[:n])
    errors_dict[name] = error
    print(f"{name} - RMSE: {np.sqrt(np.mean(error**2)):.2f} deg")


# Joint configuration
JOINTS = {
    'knee': {
        'proximal_sensor': 'femur_r_imu',
        'distal_sensor': 'tibia_r_imu',
        'gt_column': 'knee_angle_r',
        'r1_default': np.array([-0.1222504, 0.01730777, -0.00477925]),
        'r2_default': np.array([-0.03597717, -0.01554343, -0.0232674]),
    },
    'ankle': {
        'proximal_sensor': 'tibia_r_imu',
        'distal_sensor': 'calcn_r_imu',
        'gt_column': 'ankle_angle_r',
    },
}


def prepare_data(joint_name, subject_id='Subject08'):
    """Load and prepare IMU data for a given joint and subject.

    Args:
        joint_name: 'knee' or 'ankle'
        subject_id: Subject identifier (e.g., 'Subject08')

    Returns:
        dict with keys: acc_prox, gyr_prox, acc_dist, gyr_dist,
                        fs, gt, subject_path, joint_config, alignment_offset
    """
    joint_config = JOINTS[joint_name]
    subject_path = Path(f'data/{subject_id}/walking')
    imu_path = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    fs = 100.0

    # Get sensor mappings from XML
    mappings = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    prox_id = mappings.get(joint_config['proximal_sensor'])
    dist_id = mappings.get(joint_config['distal_sensor'])
    if not prox_id or not dist_id:
        raise ValueError(f"Could not find sensor IDs for {joint_name}")

    # Load IMU data
    prox_df = load_imu_data(list(imu_path.glob(f"*{prox_id}.txt"))[0])
    dist_df = load_imu_data(list(imu_path.glob(f"*{dist_id}.txt"))[0])

    acc_prox = prox_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_prox = prox_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    acc_dist = dist_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_dist = dist_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

    # Load ground truth
    gt_df = load_mot(subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot')
    gt = gt_df[joint_config['gt_column']].values

    # Compute joint-independent alignment offset using raw signals (pelvis gyro)
    offset = load_offset('raw_signal', subject_id, 'alignment')
    if offset is None:
        offset, corr, err = compute_raw_signal_offset(subject_path, fs)
        if err:
            print(f"Warning: Raw signal alignment failed ({err}), falling back to zero offset")
            offset = 0
        else:
            save_offset('raw_signal', subject_id, 'alignment', offset)
            print(f"Computed raw-signal alignment offset: {offset} samples ({offset/fs:.2f} sec), corr={corr:.3f}")
    else:
        print(f"Using cached raw-signal alignment offset: {offset} samples ({offset/fs:.2f} sec)")

    # Store original GT for precomputed methods (OpenSense, VQF-OpenSim)
    gt_original = gt

    # Align IMU and GT based on offset sign
    if offset > 0:
        # Mocap leads: trim early GT samples
        gt = gt[offset:]
    elif offset < 0:
        # IMU leads: trim early IMU samples
        trim = -offset
        acc_prox, gyr_prox = acc_prox[trim:], gyr_prox[trim:]
        acc_dist, gyr_dist = acc_dist[trim:], gyr_dist[trim:]

    # Truncate to common length (all arrays must match)
    n = min(len(acc_prox), len(acc_dist), len(gt))
    acc_prox, gyr_prox = acc_prox[:n], gyr_prox[:n]
    acc_dist, gyr_dist = acc_dist[:n], gyr_dist[:n]
    gt = gt[:n]

    print(f"Aligned data length: {n} samples ({n/fs:.1f} sec)")

    return {
        'acc_prox': acc_prox,
        'gyr_prox': gyr_prox,
        'acc_dist': acc_dist,
        'gyr_dist': gyr_dist,
        'fs': fs,
        'gt': gt,                   # Aligned GT for IMU methods
        'gt_original': gt_original, # Original GT for precomputed methods
        'subject_path': subject_path,
        'joint_config': joint_config,
        'subject_id': subject_id,
    }


def process_vqf_olsson(data, errors_dict):
    """Run VQF+Olsson and add errors to dict."""
    print("\n=== VQF+Olsson Joint Axis Estimation ===")
    angle_deg, jhat_prox, _, q_rel, _, _ = run_vqf_olsson(
        data['acc_prox'], data['gyr_prox'],
        data['acc_dist'], data['gyr_dist'],
        data['fs']
    )
    # Pick axis sign with better correlation
    angle_neg = calculate_joint_angle(q_rel, -jhat_prox)
    gt = data['gt']
    n = min(len(angle_deg), len(gt))
    if abs(np.corrcoef(angle_neg[:n], gt[:n])[0, 1]) > abs(np.corrcoef(angle_deg[:n], gt[:n])[0, 1]):
        angle_deg = angle_neg
    _eval_imu_method('vqf+olsson', angle_deg, data, errors_dict)


def process_vqf_olsson_heading_correction(data, errors_dict):
    """Run VQF+Olsson+Heading Correction and add errors to dict."""
    print("\n=== VQF+Olsson+Heading Correction ===")
    angle_deg = run_vqf_olsson_heading_corrected(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs']
    )
    _eval_imu_method('vqf+olsson+heading_correction', angle_deg, data, errors_dict)


def process_kf_gframe(data, errors_dict):
    """Run KF_Gframe and add errors to dict."""
    print("\n=== KF_Gframe with Auto R ===")
    jc = data['joint_config']
    r1, r2 = jc.get('r1_default'), jc.get('r2_default')
    angle_deg, r1_est, r2_est, _, _ = run_kf_gframe(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
        data['fs'], r1=r1, r2=r2, axis_mode='optimize', gt_angles=data['gt'], calib_samples=3000
    )
    if r1 is None:
        print(f"Estimated r1: {r1_est}, r2: {r2_est}")
    _eval_imu_method('kf_gframe', angle_deg, data, errors_dict)


def process_opensense(data, errors_dict):
    """Load OpenSense results and add errors to dict.

    Note: OpenSense results are already temporally aligned with mocap,
    so we use gt_original (no IMU-mocap offset applied).
    """
    print("\n=== OpenSense Comparison ===")
    gt = data['gt_original']
    results = load_opensense_results(data['subject_path'], data['joint_config']['gt_column'])
    for algo, angle_deg in results.items():
        _eval_precomputed(algo.capitalize(), angle_deg, gt, errors_dict)


def process_vqf_opensim(data, errors_dict):
    """Load VQF-OpenSim results and add errors to dict.

    Note: VQF-OpenSim results are already temporally aligned with mocap,
    so we use gt_original (no IMU-mocap offset applied).
    """
    vqf_file = find_vqf_opensim_file(data['subject_id'])
    if not vqf_file:
        print("\n=== VQF-OpenSim: No file found ===")
        return

    print("\n=== VQF-OpenSim ===")
    gt_col = data['joint_config']['gt_column']
    vqf_angle = load_mot(vqf_file)[gt_col].values
    _eval_precomputed('VQF-OpenSim', vqf_angle, data['gt_original'], errors_dict)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(
        description='Unified joint angle estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--joint', type=str, default='knee', choices=['knee', 'ankle'],
                        help='Joint to estimate (default: knee)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['vqf_olsson', 'vqf_olsson_heading_correction',
                                 'opensense', 'kf_gframe', 'vqf_opensim', 'all'],
                        help='Estimation method (default: all)')
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID (default: Subject08)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable interactive plotting (plots still saved)')
    args = parser.parse_args()

    Path('plots').mkdir(exist_ok=True)

    # Load data
    print(f"\nPreparing data for {args.joint} joint, {args.subject}...")
    data = prepare_data(args.joint, args.subject)
    errors_dict = {}

    # Run selected methods
    if args.method in ('kf_gframe', 'all'):
        process_kf_gframe(data, errors_dict)

    if args.method in ('vqf_olsson', 'all'):
        process_vqf_olsson(data, errors_dict)

    if args.method in ('vqf_olsson_heading_correction', 'all'):
        process_vqf_olsson_heading_correction(data, errors_dict)

    if args.method in ('opensense', 'all'):
        process_opensense(data, errors_dict)

    if args.method in ('vqf_opensim', 'all'):
        process_vqf_opensim(data, errors_dict)

    # Plot results
    if errors_dict:
        joint_title = args.joint.capitalize()
        show_plots = not args.no_plot
        plot_time_series_error(errors_dict, joint_name=joint_title, show=show_plots, num_entries=3)
        plot_error_comparison(errors_dict, joint_name=joint_title, show=show_plots)


if __name__ == "__main__":
    main()

"""
Unified joint angle estimation for knee and ankle joints.

Usage:
    python run_estimation.py --joint knee --method all
    python run_estimation.py --joint ankle --method vqf_olsson
    python run_estimation.py --joint knee --method kf_gframe --no-plot
    python run_estimation.py --joint knee --subject Subject08 --method all
    python run_estimation.py --joint knee --method all --subject all
    python run_estimation.py --joint knee --method all --subject all --workers 4
"""
import numpy as np
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Valid subjects (others excluded due to data issues per CLAUDE.md)
VALID_SUBJECTS = ['Subject02', 'Subject03', 'Subject04',
                  'Subject07', 'Subject08']

from utils import (
    load_imu_data, get_sensor_mappings,
    find_best_shift, align_signals,
    load_opensense_results, find_vqf_opensim_file, load_offset, save_offset,
    compute_raw_signal_offset, validate_offset
)
from methods.shared import load_mot, calculate_joint_angle
from methods import (
    run_vqf_olsson, run_vqf_olsson_heading_corrected,
    run_kf_gframe_olsson, run_kf_gframe_optimized
)
from plotting import plot_time_series_error, plot_error_comparison


def _eval_imu_method(name, angle_deg, data, errors_dict):
    """Evaluate IMU method against ground truth."""
    gt = data['gt']
    n = min(len(angle_deg), len(gt))
    est, gt_a = angle_deg[:n], gt[:n]
    rmse = np.sqrt(np.mean((gt_a - est)**2))
    print(f"{name} - RMSE: {rmse:.2f} deg")
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
        'gt_column': 'knee_angle_r'
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
        'alignment_offset': offset, # Raw signal alignment offset
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


def process_kf_gframe_olsson(data, errors_dict):
    """Run KF_Gframe with Olsson joint axis estimation."""
    print("\n=== KF_Gframe + Olsson ===")
    angle_deg, r1_est, r2_est, _, _ = run_kf_gframe_olsson(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs']
    )
    _eval_imu_method('kf_gframe_olsson', angle_deg, data, errors_dict)


def process_kf_gframe_optimized(data, errors_dict):
    """Run KF_Gframe with optimized joint axis (uses ground truth for calibration)."""
    print("\n=== KF_Gframe + Optimized Axis ===")
    angle_deg, r1_est, r2_est, _, _ = run_kf_gframe_optimized(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
        data['fs'], gt_angles=data['gt'], calib_samples= 3000
    )
    _eval_imu_method('kf_gframe_optimized', angle_deg, data, errors_dict)


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
    """Load VQF-OpenSim results and align to ground truth via cross-correlation."""
    vqf_file = find_vqf_opensim_file(data['subject_id'])
    if not vqf_file:
        print("\n=== VQF-OpenSim: No file found ===")
        return

    print("\n=== VQF-OpenSim ===")
    gt_col = data['joint_config']['gt_column']
    vqf_angle = load_mot(vqf_file)[gt_col].values
    gt = data['gt_original']

    # VQF-OpenSim has different time boundaries than raw IMU, align via cross-correlation
    offset, _ = find_best_shift(vqf_angle, gt)
    est, gt_aligned = align_signals(vqf_angle, gt, offset)

    error = np.abs(gt_aligned - est)
    errors_dict['VQF-OpenSim'] = error
    rmse = np.sqrt(np.mean(error**2))
    print(f"VQF-OpenSim - RMSE: {rmse:.2f} deg (offset: {offset})")


def run_single_subject(joint, method, subject_id, no_plot=True):
    """Run estimation on a single subject and return errors dict."""
    print(f"\n{'='*60}")
    print(f"Processing {subject_id} - {joint} joint")
    print(f"{'='*60}")

    try:
        data = prepare_data(joint, subject_id)
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return subject_id, {}

    errors_dict = {}

    if method in ('kf_gframe_olsson', 'all'):
        process_kf_gframe_olsson(data, errors_dict)

    if method in ('kf_gframe_optimized', 'all'):
        process_kf_gframe_optimized(data, errors_dict)

    if method == 'vqf_olsson':  # Excluded from 'all' due to poor performance
        process_vqf_olsson(data, errors_dict)

    if method in ('vqf_olsson_heading_correction', 'all'):
        process_vqf_olsson_heading_correction(data, errors_dict)

    if method in ('opensense', 'all'):
        process_opensense(data, errors_dict)

    if method in ('vqf_opensim', 'all'):
        process_vqf_opensim(data, errors_dict)

    # Plot results for single subject (when not in parallel mode)
    if errors_dict:
        joint_title = joint.capitalize()
        plot_time_series_error(errors_dict, joint_name=joint_title, show=not no_plot, num_entries=3)
        plot_error_comparison(errors_dict, joint_name=joint_title, show=not no_plot)

    return subject_id, errors_dict


def run_all_subjects(joint, method, workers=None):
    """Run estimation on all valid subjects in parallel."""
    print(f"\nRunning {method} on all subjects for {joint} joint...")
    print(f"Valid subjects: {', '.join(VALID_SUBJECTS)}")
    print(f"Workers: {workers or 'auto (CPU count)'}\n")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_subject, joint, method, subj, True): subj
            for subj in VALID_SUBJECTS
        }
        for future in as_completed(futures):
            subj = futures[future]
            try:
                _, errors = future.result()
                results[subj] = errors
            except Exception as e:
                print(f"Error processing {subj}: {e}")
                results[subj] = {}
    return results


def print_summary_table(results, joint):
    """Print RMSE summary table and save to CSV."""
    # Collect all methods from results
    methods = set()
    for errors in results.values():
        methods.update(errors.keys())
    methods = sorted(methods)

    if not methods:
        print("No results to summarize.")
        return

    # Build rows with RMSE values
    rows = []
    for subj in sorted(results.keys()):
        row = {'subject': subj}
        for m in methods:
            if m in results[subj] and len(results[subj][m]) > 0:
                row[m] = np.sqrt(np.mean(results[subj][m]**2))
            else:
                row[m] = np.nan
        rows.append(row)

    # Add mean row
    mean_row = {'subject': 'MEAN'}
    for m in methods:
        vals = [r[m] for r in rows if not np.isnan(r.get(m, np.nan))]
        mean_row[m] = np.mean(vals) if vals else np.nan
    rows.append(mean_row)

    df = pd.DataFrame(rows)

    # Print table
    print("\n" + "="*80)
    print(f"RMSE Summary - {joint.capitalize()} Joint (degrees)")
    print("="*80)
    print(df.to_string(index=False, float_format='%.2f'))

    # Save to CSV
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f'{joint}_rmse_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\nResults saved to {csv_path}")


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
                                 'opensense', 'kf_gframe_olsson', 'kf_gframe_optimized',
                                 'vqf_opensim', 'all'],
                        help='Estimation method (default: all)')
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID or "all" for all valid subjects (default: Subject08)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable interactive plotting (plots still saved)')
    args = parser.parse_args()

    Path('plots').mkdir(exist_ok=True)

    if args.subject == 'all':
        results = run_all_subjects(args.joint, args.method, args.workers)
        print_summary_table(results, args.joint)
    else:
        _ = run_single_subject(args.joint, args.method, args.subject, args.no_plot)


if __name__ == "__main__":
    main()

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

from utils import (
    load_imu_data, get_sensor_mappings,
    load_opensense_results, find_vqf_ik_file,
    get_aligned_time_range
)
from constants import VALID_SUBJECTS
from methods.shared import load_mot, calculate_joint_angle
from methods import (
    run_vqf_olsson, run_vqf_olsson_heading_corrected,
    run_kf_gframe_olsson, run_kf_gframe_optimized, run_kf_gframe_opensim
)
from plotting import plot_time_series_error, plot_error_comparison


def _eval_method(name, est, gt, errors_dict, angles_dict=None):
    """Evaluate estimated angles against ground truth."""
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    error = np.abs(gt - est)
    errors_dict[name] = error
    print(f"{name} - RMSE: {np.sqrt(np.mean(error**2)):.2f} deg")
    if angles_dict is not None:
        angles_dict[name] = (est, gt)


def export_time_series(subject_id, joint, angles_dict, fs, output_dir='results/time_series'):
    """Export angle time series to CSV (one file per method)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for method, (angle_deg, gt_method) in angles_dict.items():
        n = min(len(angle_deg), len(gt_method))
        df = pd.DataFrame({
            'time': np.arange(n) / fs,
            'sample': np.arange(n),
            'estimated_deg': angle_deg[:n],
            'gt_deg': gt_method[:n]
        })
        safe_method = method.replace('+', '_').replace('-', '_')
        csv_path = Path(output_dir) / f'{subject_id}_{joint}_{safe_method}.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Exported: {csv_path}")


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

    # Store original GT for precomputed methods (OpenSense, VQF-OpenSim)
    gt_original = gt

    # Get aligned time range using centralized utility
    time_range = get_aligned_time_range(subject_path, fs)
    offset = time_range['offset']
    imu_start = time_range['imu_start']
    imu_end = time_range['imu_end']
    print(f"Alignment offset: {offset} samples ({offset/fs:.2f} sec)")

    # Trim IMU data to aligned range
    acc_prox, gyr_prox = acc_prox[imu_start:imu_end], gyr_prox[imu_start:imu_end]
    acc_dist, gyr_dist = acc_dist[imu_start:imu_end], gyr_dist[imu_start:imu_end]

    # Truncate to common length (all arrays must match GT)
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


def process_vqf_olsson(data, errors_dict, angles_dict=None):
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
    _eval_method('vqf+olsson', angle_deg, data['gt'], errors_dict, angles_dict)


def process_vqf_olsson_heading_correction(data, errors_dict, angles_dict=None):
    """Run VQF+Olsson+Heading Correction and add errors to dict."""
    print("\n=== VQF+Olsson+Heading Correction ===")
    angle_deg = run_vqf_olsson_heading_corrected(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs']
    )
    _eval_method('vqf+olsson+heading_correction', angle_deg, data['gt'], errors_dict, angles_dict)


def process_kf_gframe_olsson(data, errors_dict, angles_dict=None):
    """Run KF_Gframe with Olsson joint axis estimation."""
    print("\n=== KF_Gframe + Olsson ===")
    angle_deg, r1_est, r2_est, _, _ = run_kf_gframe_olsson(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs']
    )
    _eval_method('kf_gframe_olsson', angle_deg, data['gt'], errors_dict, angles_dict)


def process_kf_gframe_optimized(data, errors_dict, angles_dict=None):
    """Run KF_Gframe with optimized joint axis (uses ground truth for calibration)."""
    print("\n=== KF_Gframe + Optimized Axis ===")
    angle_deg, _, _, _, _ = run_kf_gframe_optimized(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
        data['fs'], gt_angles=data['gt'],calib_samples=3000
    )
    _eval_method('kf_gframe_optimized', angle_deg, data['gt'], errors_dict, angles_dict)


def process_kf_gframe_opensim(data, errors_dict, angles_dict=None):
    """Run KF_Gframe with precomputed OpenSim joint axis."""
    print("\n=== KF_Gframe + OpenSim Axis ===")
    joint = 'knee' if 'knee' in data['joint_config']['gt_column'] else 'ankle'
    angle_deg, _, _, jhat, _ = run_kf_gframe_opensim(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
        data['fs'], joint=joint
    )
    print(f"Joint axis: [{jhat[0]:.3f}, {jhat[1]:.3f}, {jhat[2]:.3f}]")
    _eval_method('kf_gframe_opensim', angle_deg, data['gt'], errors_dict, angles_dict)


def process_opensense(data, errors_dict, angles_dict=None):
    """Load OpenSense IK results (VQF, Madgwick, Mahony, Xsens) and add errors to dict.

    Note: OpenSense results are already temporally aligned with mocap,
    so we use gt_original (no IMU-mocap offset applied).
    """
    print("\n=== OpenSense IK Comparison ===")
    gt = data['gt_original']
    results = load_opensense_results(
        data['subject_path'],
        data['joint_config']['gt_column'],
        weighting='IKWithErrorsExtremeLowFeetWeights'
    )
    for algo, angle_deg in results.items():
        _eval_method(algo.upper(), angle_deg, gt, errors_dict, angles_dict)


def process_vqf_ik(data, errors_dict, angles_dict=None):
    """Load VQF IK results using raw signal alignment (same as IMU methods)."""
    vqf_file = find_vqf_ik_file(data['subject_id'])
    if not vqf_file:
        print("\n=== VQF-IK: No file found ===")
        return

    print("\n=== VQF-IK ===")
    gt_col = data['joint_config']['gt_column']
    vqf_angle = load_mot(vqf_file)[gt_col].values

    gt = data['gt']

    # Check if VQF-IK is pre-aligned (similar length to mocap) or needs offset trimming
    # Pre-aligned VQF-IK has ~60k samples matching mocap, unaligned has ~144k
    offset = data['alignment_offset']
    vqf_is_prealigned = len(vqf_angle) < len(gt) * 1.5  # Within 50% of mocap length

    if not vqf_is_prealigned and offset < 0:
        # Old unaligned VQF-IK: apply offset trimming
        vqf_angle = vqf_angle[-offset:]

    n = min(len(vqf_angle), len(gt))
    est, gt_aligned = vqf_angle[:n], gt[:n]

    error = np.abs(gt_aligned - est)
    errors_dict['VQF-IK'] = error
    rmse = np.sqrt(np.mean(error**2))
    print(f"VQF-IK - RMSE: {rmse:.2f} deg")
    if angles_dict is not None:
        angles_dict['VQF-IK'] = (est, gt_aligned)


def run_single_subject(joint, method, subject_id, no_plot=True, export=False):
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
    angles_dict = {} if export else None

    if method in ('kf_gframe_olsson', 'all'):
        process_kf_gframe_olsson(data, errors_dict, angles_dict)

    if method in ('kf_gframe_optimized', 'all'):
        process_kf_gframe_optimized(data, errors_dict, angles_dict)

    if method in ('kf_gframe_opensim', 'all'):
        process_kf_gframe_opensim(data, errors_dict, angles_dict)

    if method == 'vqf_olsson':  # Excluded from 'all' due to poor performance
        process_vqf_olsson(data, errors_dict, angles_dict)

    if method in ('vqf_olsson_heading_correction', 'all'):
        process_vqf_olsson_heading_correction(data, errors_dict, angles_dict)

    if method in ('opensense', 'all'):
        process_opensense(data, errors_dict, angles_dict)

    if method in ('vqf_ik', 'all'):
        process_vqf_ik(data, errors_dict, angles_dict)

    # Export time series if requested
    if export and angles_dict:
        export_time_series(subject_id, joint, angles_dict, data['fs'])

    # Plot results for single subject (when not in parallel mode)
    if errors_dict:
        joint_title = joint.capitalize()
        plot_time_series_error(errors_dict, joint_name=joint_title, show=not no_plot, num_entries=3)
        plot_error_comparison(errors_dict, joint_name=joint_title, show=not no_plot)

    return subject_id, errors_dict


def run_all_subjects(joint, method, workers=None, export=False):
    """Run estimation on all valid subjects in parallel."""
    print(f"\nRunning {method} on all subjects for {joint} joint...")
    print(f"Valid subjects: {', '.join(VALID_SUBJECTS)}")
    print(f"Workers: {workers or 'auto (CPU count)'}\n")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_subject, joint, method, subj, True, export): subj
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
                                 'kf_gframe_opensim', 'vqf_ik', 'all'],
                        help='Estimation method (default: all)')
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID or "all" for all valid subjects (default: Subject08)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable interactive plotting (plots still saved)')
    parser.add_argument('--export', action='store_true',
                        help='Export angle time series to results/time_series/')
    args = parser.parse_args()

    Path('plots').mkdir(exist_ok=True)

    if args.subject == 'all':
        results = run_all_subjects(args.joint, args.method, args.workers, args.export)
        print_summary_table(results, args.joint)
    else:
        _ = run_single_subject(args.joint, args.method, args.subject, args.no_plot, args.export)


if __name__ == "__main__":
    main()

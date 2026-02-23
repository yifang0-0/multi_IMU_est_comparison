"""Alignment sensitivity analysis: sweep est vs GT shift by +-50 samples for all methods."""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so imports work from evaluations/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import VALID_SUBJECTS
from run_estimation import prepare_data, _find_calibrated_model
from utils import load_opensense_results, find_vqf_ik_file
from methods.shared import load_mot, calculate_joint_angle
from methods import (
    run_vqf_olsson, run_vqf_olsson_heading_corrected,
    run_kf_gframe_all_variants, run_rnno_all_variants,
)

MAX_SHIFT = 50


def compute_shifted_rmse(est, gt, delta):
    """Compute RMSE between est and gt shifted by delta samples."""
    if delta > 0:
        return np.sqrt(np.mean((est[delta:] - gt[:-delta]) ** 2))
    elif delta < 0:
        return np.sqrt(np.mean((est[:delta] - gt[-delta:]) ** 2))
    return np.sqrt(np.mean((est - gt) ** 2))


def sweep_shifts(est, gt, method, subject, joint, rows):
    """Truncate to common length then sweep shifts, appending to rows."""
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    for delta in range(-MAX_SHIFT, MAX_SHIFT + 1):
        rows.append({
            'method': method,
            'subject': subject,
            'joint': joint,
            'delta': delta,
            'rmse': compute_shifted_rmse(est, gt, delta),
        })


def fix_axis_sign(angle_deg, jhat, q_rel, gt):
    """Flip axis sign if negative correlation gives better match."""
    angle_neg = calculate_joint_angle(q_rel, -jhat)
    n = min(len(angle_deg), len(gt))
    if abs(np.corrcoef(angle_neg[:n], gt[:n])[0, 1]) > abs(np.corrcoef(angle_deg[:n], gt[:n])[0, 1]):
        return angle_neg
    return angle_deg


def collect_methods(joint, subject, data):
    """Run all methods once, return list of (method_name, est_angles, gt_angles)."""
    gt = data['gt']
    pairs = []

    # --- KF_Gframe (shared orientation) ---
    model_path = _find_calibrated_model(data['subject_path'])
    model_kwargs = {}
    if model_path is not None:
        model_kwargs = {
            'model_path': model_path,
            'prox_imu': data['joint_config']['proximal_sensor'],
            'dist_imu': data['joint_config']['distal_sensor'],
        }
    kf = run_kf_gframe_all_variants(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
        data['fs'], gt_angles=gt, calib_samples=3000, joint=joint, **model_kwargs,
    )
    # olsson & pca need sign correction
    for key, name in [('olsson', 'kf_gframe_olsson'), ('pca', 'kf_gframe_pca')]:
        angle_deg, jhat, q_rel = kf[key]
        pairs.append((name, fix_axis_sign(angle_deg, jhat, q_rel, gt), gt))
    for key, name in [('optimized', 'kf_gframe_optimize')]:
        if key in kf:
            pairs.append((name, kf[key][0], gt))
    if 'model' in kf:
        pairs.append(('kf_gframe_model', kf['model'][0], gt))

    # --- RNNO (shared orientation) ---
    try:
        rnno = run_rnno_all_variants(
            data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'],
            data['fs'], gt_angles=gt, calib_samples=None, joint=joint, **model_kwargs,
        )
        for key, name in [('olsson', 'rnno_olsson'), ('pca', 'rnno_pca')]:
            if key in rnno:
                angle_deg, jhat, q_rel = rnno[key]
                pairs.append((name, fix_axis_sign(angle_deg, jhat, q_rel, gt), gt))
        for key, name in [('optimized', 'rnno_optimized')]:
            if key in rnno:
                pairs.append((name, rnno[key][0], gt))
        if 'model' in rnno:
            pairs.append(('rnno_model', rnno['model'][0], gt))
    except Exception as e:
        print(f"  RNNO skipped: {e}")

    # --- VQF+Olsson ---
    angle_deg, jhat_prox, _, q_rel, _, _ = run_vqf_olsson(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs'],
    )
    pairs.append(('vqf_olsson', fix_axis_sign(angle_deg, jhat_prox, q_rel, gt), gt))

    # --- VQF+Olsson+Heading Correction ---
    angle_deg = run_vqf_olsson_heading_corrected(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs'],
    )
    pairs.append(('vqf_olsson_hc', angle_deg, gt))

    # --- OpenSense (uses gt_original, own alignment) ---
    gt_orig = data['gt_original']
    opensense = load_opensense_results(
        data['subject_path'], data['joint_config']['gt_column'],
        weighting='IKWithErrorsExtremeLowFeetWeights',
    )
    for algo, angle_deg in opensense.items():
        pairs.append((f'opensense_{algo}', angle_deg, gt_orig))

    # --- VQF-IK ---
    vqf_file = find_vqf_ik_file(data['subject_id'])
    if vqf_file:
        vqf_angle = load_mot(vqf_file)[data['joint_config']['gt_column']].values
        offset = data['alignment_offset']
        if len(vqf_angle) >= len(gt) * 1.5 and offset < 0:
            vqf_angle = vqf_angle[-offset:]
        pairs.append(('vqf_ik', vqf_angle, gt))

    return pairs


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    rows = []

    for joint in ['knee', 'ankle']:
        for subject in VALID_SUBJECTS:
            print(f"\n--- {subject} / {joint} ---")
            data = prepare_data(joint, subject)
            pairs = collect_methods(joint, subject, data)
            print(f"  {len(pairs)} methods collected")
            for method, est, gt in pairs:
                sweep_shifts(est, gt, method, subject, joint, rows)

    df = pd.DataFrame(rows)

    # Save CSV
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / 'alignment_sensitivity.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved {csv_path} ({len(df)} rows)")

    # Print summary table
    methods = df['method'].unique()
    header = f"{'Subject':<12} {'Joint':<8} " + " ".join(f"{m:>22}" for m in methods)
    print(f"\n{'='*len(header)}")
    print("Optimal delta per method (RMSE@0 -> best RMSE @ delta)")
    print(f"{'='*len(header)}")
    print(header)
    print('-' * len(header))

    for (subject, joint), sj_group in df.groupby(['subject', 'joint'], sort=False):
        parts = [f"{subject:<12} {joint:<8}"]
        for m in methods:
            mg = sj_group[sj_group['method'] == m]
            if mg.empty:
                parts.append(f"{'—':>22}")
                continue
            r0 = mg.loc[mg['delta'] == 0, 'rmse'].values[0]
            best_idx = mg['rmse'].idxmin()
            br = mg.loc[best_idx, 'rmse']
            bd = int(mg.loc[best_idx, 'delta'])
            parts.append(f"{r0:.2f}->{br:.2f}@{bd:+d}".rjust(22))
        print(" ".join(parts))


if __name__ == '__main__':
    main()

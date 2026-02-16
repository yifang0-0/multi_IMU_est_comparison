"""Validate absolute orientation from optical marker pipeline."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
import qmt

# Add project root to path so imports work from evaluations/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import (
    load_trc, orientation_from_marker_triad, compute_q_rel_optical,
    load_imu_data, get_sensor_mappings, read_orientations_sto,
    get_aligned_time_range, estimate_orientations,
)
from run_estimation import prepare_data
from methods import run_kf_gframe, run_rnno
from methods.shared import load_mot, calculate_joint_angle
from constants import FS

# IMU marker triads: body_part -> (O, X, Y) marker names
IMU_TRIADS = {
    'pelvis':  ('Pelvis_IMU_O',   'Pelvis_IMU_X',   'Pelvis_IMU_Y'),
    'femur':   ('R.Femur_IMU_O',  'R.Femur_IMU_X',  'R.Femur_IMU_Y'),
    'tibia':   ('R.Tibia_IMU_O',  'R.Tibia_IMU_X',  'R.Tibia_IMU_Y'),
    'foot':    ('R.Foot_IMU_O',   'R.Foot_IMU_X',   'R.Foot_IMU_Y'),
}

# Xsens .sto sensor names corresponding to each body part
STO_SENSOR_NAMES = {
    'pelvis': 'pelvis_imu',
    'femur':  'femur_r_imu',
    'tibia':  'tibia_r_imu',
    'foot':   'calcn_r_imu',
}


def _all_triad_markers():
    """Return flat list of all O/X/Y marker names across all triads."""
    markers = []
    for o, x, y in IMU_TRIADS.values():
        markers.extend([o, x, y])
    return markers


def _compute_optical_orientations(trc_path):
    """Compute per-IMU orientations from TRC marker triads, returns {body_part: (N, 4)}."""
    all_markers = _all_triad_markers()
    _, markers = load_trc(trc_path, markers=all_markers)

    orientations = {}
    for part, (o, x, y) in IMU_TRIADS.items():
        q = orientation_from_marker_triad(markers[o], markers[x], markers[y])
        orientations[part] = qmt.quatUnwrap(q)
    return orientations


def _procrustes_q_L(q_xs, q_opt, min_angle_deg=5):
    """Find rotation q_L such that q_rel_xs ≈ q_L * q_rel_opt * qinv(q_L)."""
    q_rel_xs = qmt.qmult(q_xs, qmt.qinv(q_xs[0]))
    q_rel_opt = qmt.qmult(q_opt, qmt.qinv(q_opt[0]))

    axes_xs = qmt.quatAxis(q_rel_xs)
    axes_opt = qmt.quatAxis(q_rel_opt)
    angles = np.abs(np.degrees(qmt.quatAngle(q_rel_xs)))
    big = angles > min_angle_deg

    H = axes_opt[big].T @ axes_xs[big]
    U, _, Vt = np.linalg.svd(H)
    R_L = Vt.T @ U.T
    if np.linalg.det(R_L) < 0:
        Vt[-1] *= -1
        R_L = Vt.T @ U.T
    return qmt.quatFromRotMat(R_L.reshape(1, 3, 3))[0]


# ============================================================================
# Validation 1: VQF 6D inclination comparison (tilt, 2 DOF)
# ============================================================================

def validate_inclination(subject_id='Subject08'):
    """Compare optical inclination vs VQF 6D (accel+gyro) inclination during walking."""
    subject_path = ROOT / f'data/{subject_id}/walking'
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    mapping_path = subject_path / 'IMU' / 'myIMUMappings_walking.xml'

    mappings = get_sensor_mappings(mapping_path)
    q_optical = _compute_optical_orientations(str(subject_path / 'Mocap' / 'walking.trc'))
    alignment = get_aligned_time_range(subject_path)
    imu_start, imu_end, gt_samples = alignment['imu_start'], alignment['imu_end'], alignment['gt_samples']

    G = 9.81

    print(f"VQF 6D inclination validation ({subject_id}):")
    print(f"{'Sensor':<10} {'Mean (deg)':>10} {'Median':>8} {'P95':>8}  {'Frames':>8}")
    print('-' * 50)

    results = {}
    for part, sto_name in STO_SENSOR_NAMES.items():
        sensor_id = mappings.get(sto_name, '').lstrip('_')
        if not sensor_id:
            print(f"{part:<10}  SKIPPED (no sensor mapping)")
            continue

        imu_files = list(imu_dir.glob(f"*{sensor_id}.txt"))
        if not imu_files:
            print(f"{part:<10}  SKIPPED (no IMU file)")
            continue

        df = load_imu_data(imu_files[0])
        acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
        gyr = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

        # Run VQF 6D (no magnetometer) → sensor-to-Earth quaternions
        q_vqf_full = estimate_orientations(acc, gyr, FS)
        q_vqf = qmt.quatUnwrap(q_vqf_full[imu_start:imu_end])

        q_opt = q_optical[part]
        n = min(len(q_vqf), len(q_opt), gt_samples)
        q_vqf = q_vqf[:n]
        q_opt = q_opt[:n]

        valid = ~np.isnan(q_opt).any(axis=1)
        if valid.sum() < 100:
            print(f"{part:<10}  SKIPPED (too few valid frames)")
            continue

        # Gravity in each body frame
        g_vqf = qmt.rotate(qmt.qinv(q_vqf), np.array([0, 0, G]))    # NED: gravity = [0,0,G]
        g_opt = qmt.rotate(qmt.qinv(q_opt), np.array([0, -G, 0]))    # Y-up: gravity = [0,-G,0]

        # Normalize
        g_vqf_n = g_vqf / np.linalg.norm(g_vqf, axis=1, keepdims=True)
        g_opt_n = g_opt / np.linalg.norm(g_opt, axis=1, keepdims=True)

        # Procrustes SVD to find constant marker-to-sensor rotation for gravity vectors
        H = g_opt_n[valid].T @ g_vqf_n[valid]
        U, _, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T
        if np.linalg.det(R_align) < 0:
            Vt[-1] *= -1
            R_align = Vt.T @ U.T

        g_opt_aligned = (R_align @ g_opt_n.T).T

        # Per-frame inclination error
        cos_err = np.clip(np.sum(g_vqf_n * g_opt_aligned, axis=1), -1, 1)
        incl_err = np.degrees(np.arccos(cos_err))

        errs_valid = incl_err[valid]
        mean_err = np.mean(errs_valid)
        median_err = np.median(errs_valid)
        p95_err = np.percentile(errs_valid, 95)

        results[part] = {
            'mean': mean_err, 'median': median_err, 'p95': p95_err,
            'errors': errs_valid, 'n_frames': len(errs_valid),
        }
        print(f"{part:<10} {mean_err:>10.2f} {median_err:>8.2f} {p95_err:>8.2f}  {len(errs_valid):>8}")

    # Plot
    if results:
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (part, r) in zip(axes, results.items()):
            time = np.arange(r['n_frames']) / FS
            ax.plot(time, r['errors'], alpha=0.6, linewidth=0.5)
            ax.axhline(r['mean'], color='r', linestyle='--', label=f"mean={r['mean']:.1f}°")
            ax.set_ylabel('Error (deg)')
            ax.set_title(f'{part} — mean={r["mean"]:.1f}°, median={r["median"]:.1f}°, P95={r["p95"]:.1f}°')
            ax.legend(loc='upper right')
            ax.set_ylim(0, min(10, r['p95'] * 2))
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(f'Optical vs VQF 6D Inclination Error ({subject_id})')
        plt.tight_layout()
        plt.savefig(ROOT / f'results/inclination_validation_{subject_id}.png', dpi=150)
        print(f"Saved: results/inclination_validation_{subject_id}.png")

    return results


# ============================================================================
# Validation 1b: KF gframe q_rel vs optical q_rel (3 DOF)
# ============================================================================

def validate_qrel_method(method_fn, method_name, subject_id='Subject08'):
    """Compare a method's relative orientation against optical ground truth."""
    trc_path = str(ROOT / f'data/{subject_id}/walking/Mocap/walking.trc')

    print(f"{method_name} q_rel validation ({subject_id}):")
    print(f"{'Joint':<10} {'Mean (deg)':>10} {'Median':>8} {'P95':>8}  {'Frames':>8}")
    print('-' * 50)

    results = {}
    for joint in ['knee', 'ankle']:
        data = prepare_data(joint, subject_id)
        _, _, _, _, q_rel_est = method_fn(
            data['acc_prox'], data['gyr_prox'],
            data['acc_dist'], data['gyr_dist'],
            data['fs'], axis_mode='olsson',
        )
        q_rel_est = qmt.quatUnwrap(q_rel_est)

        q_rel_opt = compute_q_rel_optical(trc_path, joint)
        q_rel_opt = qmt.quatUnwrap(q_rel_opt)

        n = min(len(q_rel_est), len(q_rel_opt))
        q_rel_est = q_rel_est[:n]
        q_rel_opt = q_rel_opt[:n]

        valid = ~np.isnan(q_rel_opt).any(axis=1)
        if valid.sum() < 100:
            print(f"{joint:<10}  SKIPPED (too few valid frames)")
            continue

        # Changes relative to first frame
        dq_est = qmt.qmult(q_rel_est, qmt.qinv(q_rel_est[0]))
        dq_opt = qmt.qmult(q_rel_opt, qmt.qinv(q_rel_opt[0]))

        # Find alignment: dq_est ≈ q_L * dq_opt * qinv(q_L)
        q_L = _procrustes_q_L(q_rel_est[valid], q_rel_opt[valid])

        # Apply conjugation alignment
        dq_opt_aligned = qmt.qmult(q_L, qmt.qmult(dq_opt, qmt.qinv(q_L)))

        # Per-frame angular error
        q_err = qmt.qmult(qmt.qinv(dq_est), dq_opt_aligned)
        angle_err = np.abs(np.degrees(qmt.quatAngle(q_err)))

        errs_valid = angle_err[valid]
        mean_err = np.mean(errs_valid)
        median_err = np.median(errs_valid)
        p95_err = np.percentile(errs_valid, 95)

        results[joint] = {
            'mean': mean_err, 'median': median_err, 'p95': p95_err,
            'errors': errs_valid, 'n_frames': len(errs_valid),
        }
        print(f"{joint:<10} {mean_err:>10.2f} {median_err:>8.2f} {p95_err:>8.2f}  {len(errs_valid):>8}")

    # Plot
    if results:
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (joint, r) in zip(axes, results.items()):
            time = np.arange(r['n_frames']) / FS
            ax.plot(time, r['errors'], alpha=0.6, linewidth=0.5)
            ax.axhline(r['mean'], color='r', linestyle='--', label=f"mean={r['mean']:.1f}°")
            ax.set_ylabel('Error (deg)')
            ax.set_title(f'{joint} — mean={r["mean"]:.1f}°, median={r["median"]:.1f}°, P95={r["p95"]:.1f}°')
            ax.legend(loc='upper right')
            ax.set_ylim(0, min(30, r['p95'] * 2))
        axes[-1].set_xlabel('Time (s)')
        safe_name = method_name.lower().replace(' ', '_')
        plt.suptitle(f'{method_name} q_rel vs Optical q_rel ({subject_id})')
        plt.tight_layout()
        plt.savefig(ROOT / f'results/qrel_{safe_name}_validation_{subject_id}.png', dpi=150)
        print(f"Saved: results/qrel_{safe_name}_validation_{subject_id}.png")

    return results


# ============================================================================
# Validation 2: Xsens orientation comparison (3 DOF)
# ============================================================================

def validate_xsens_comparison(subject_id='Subject08'):
    """Compare optical marker orientations vs Xsens AHRS quaternions."""
    subject_path = ROOT / f'data/{subject_id}/walking'
    walking_trc_path = subject_path / 'Mocap' / 'walking.trc'
    sto_path = subject_path / 'IMU' / 'xsens' / 'walking_orientations.sto'

    q_optical = _compute_optical_orientations(str(walking_trc_path))
    _, q_xsens_dict, _ = read_orientations_sto(sto_path)

    # Note: .sto quaternions and TRC are already temporally synchronized (both at 100Hz).
    # No temporal offset needed — the .sto is produced by Xsens's OpenSense pipeline
    # which uses the same time base as the mocap TRC.

    print(f"\nXsens orientation comparison ({subject_id}):")
    print(f"{'Sensor':<10} {'Mean (deg)':>10} {'Median':>8} {'P95':>8} {'ω corr':>7}  {'Frames':>8}")
    print('-' * 58)

    results = {}
    for part, sto_name in STO_SENSOR_NAMES.items():
        q_opt = q_optical[part]
        q_xs = q_xsens_dict.get(sto_name)
        if q_xs is None:
            print(f"{part:<10}  SKIPPED (not in .sto)")
            continue

        q_xs = qmt.quatUnwrap(q_xs)
        n = min(len(q_xs), len(q_opt))
        q_xs_a = q_xs[:n]
        q_opt_a = q_opt[:n]

        valid = ~np.isnan(q_opt_a).any(axis=1) & ~np.isnan(q_xs_a).any(axis=1)
        if valid.sum() < 100:
            print(f"{part:<10}  SKIPPED (too few valid frames)")
            continue

        # Angular velocity magnitude correlation (frame-independent motion check)
        q_diff_xs = qmt.qmult(qmt.qinv(q_xs_a[:-1]), q_xs_a[1:])
        q_diff_opt = qmt.qmult(qmt.qinv(q_opt_a[:-1]), q_opt_a[1:])
        omega_xs = np.abs(np.degrees(qmt.quatAngle(q_diff_xs))) * FS
        omega_opt = np.abs(np.degrees(qmt.quatAngle(q_diff_opt))) * FS
        v2 = valid[:-1] & valid[1:]
        b, a = butter(4, 5 / (0.5 * FS), 'low')
        omega_corr = np.corrcoef(
            filtfilt(b, a, omega_xs[v2]),
            filtfilt(b, a, omega_opt[v2]),
        )[0, 1]

        # Find q_L via Procrustes on rotation axes
        q_L = _procrustes_q_L(q_xs_a[valid], q_opt_a[valid])

        # Relative orientations (referenced to first frame)
        q_rel_xs = qmt.qmult(q_xs_a, qmt.qinv(q_xs_a[0]))
        q_rel_opt = qmt.qmult(q_opt_a, qmt.qinv(q_opt_a[0]))

        # Apply conjugation alignment: q_opt_aligned = q_L * q_rel_opt * qinv(q_L)
        q_opt_aligned = qmt.qmult(q_L, qmt.qmult(q_rel_opt, qmt.qinv(q_L)))

        # Per-frame angular error
        q_err = qmt.qmult(qmt.qinv(q_opt_aligned), q_rel_xs)
        angle_err = np.abs(np.degrees(qmt.quatAngle(q_err)))

        errs_valid = angle_err[valid]
        mean_err = np.mean(errs_valid)
        median_err = np.median(errs_valid)
        p95_err = np.percentile(errs_valid, 95)

        results[part] = {
            'mean': mean_err, 'median': median_err, 'p95': p95_err,
            'omega_corr': omega_corr,
            'errors': errs_valid, 'n_frames': len(errs_valid),
        }
        print(f"{part:<10} {mean_err:>10.2f} {median_err:>8.2f} {p95_err:>8.2f}"
              f" {omega_corr:>7.3f}  {len(errs_valid):>8}")

    # Plot
    if results:
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (part, r) in zip(axes, results.items()):
            time = np.arange(r['n_frames']) / FS
            ax.plot(time, r['errors'], alpha=0.6, linewidth=0.5)
            ax.axhline(r['mean'], color='r', linestyle='--', label=f"mean={r['mean']:.1f}°")
            ax.set_ylabel('Error (deg)')
            ax.set_title(f'{part} — mean={r["mean"]:.1f}°, median={r["median"]:.1f}°, P95={r["p95"]:.1f}°')
            ax.legend(loc='upper right')
            ax.set_ylim(0, min(30, r['p95'] * 2))
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(f'Optical vs Xsens Orientation Error ({subject_id})')
        plt.tight_layout()
        plt.savefig(ROOT / f'results/xsens_comparison_{subject_id}.png', dpi=150)
        print(f"Saved: results/xsens_comparison_{subject_id}.png")

    return results


# ============================================================================
# Validation 3: Marker triad geometric consistency
# ============================================================================

def validate_geometry(subject_id='Subject08'):
    """Check marker triad rigidity, orthogonality, and static stability."""
    subject_path = ROOT / f'data/{subject_id}/walking'

    print(f"\nMarker triad geometric consistency ({subject_id}):")

    results = {}
    for trc_name, label in [('static_walking.trc', 'static'), ('walking.trc', 'walking')]:
        trc_path = subject_path / 'Mocap' / trc_name
        all_markers = _all_triad_markers()
        _, markers = load_trc(str(trc_path), markers=all_markers)

        print(f"\n  [{label}] {trc_name}")
        print(f"  {'Sensor':<10} {'|O→X| mm':>12} {'|O→Y| mm':>12} {'Orth dev (deg)':>14}")
        print(f"  {'':<10} {'mean±std':>12} {'mean±std':>12} {'from 90°':>14}")
        print('  ' + '-' * 52)

        for part, (o, x, y) in IMU_TRIADS.items():
            origin = markers[o]
            vec_ox = markers[x] - origin
            vec_oy = markers[y] - origin

            dist_ox = np.linalg.norm(vec_ox, axis=1)
            dist_oy = np.linalg.norm(vec_oy, axis=1)

            valid = ~(np.isnan(dist_ox) | np.isnan(dist_oy))
            dist_ox_v = dist_ox[valid]
            dist_oy_v = dist_oy[valid]

            dot = np.sum(vec_ox[valid] * vec_oy[valid], axis=1)
            cos_angle = dot / (dist_ox_v * dist_oy_v + 1e-10)
            angle_xy = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            orth_dev = angle_xy - 90.0

            print(f"  {part:<10} {dist_ox_v.mean():>7.2f}±{dist_ox_v.std():>4.2f}"
                  f" {dist_oy_v.mean():>7.2f}±{dist_oy_v.std():>4.2f}"
                  f" {np.mean(orth_dev):>+8.2f}±{np.std(orth_dev):.2f}")

            results[f'{part}_{label}'] = {
                'dist_ox_mean': dist_ox_v.mean(), 'dist_ox_std': dist_ox_v.std(),
                'dist_oy_mean': dist_oy_v.mean(), 'dist_oy_std': dist_oy_v.std(),
                'orth_dev_mean': np.mean(orth_dev), 'orth_dev_std': np.std(orth_dev),
            }

    # Static orientation stability
    static_trc = subject_path / 'Mocap' / 'static_walking.trc'
    q_static = _compute_optical_orientations(str(static_trc))

    print(f"\n  Static orientation stability:")
    print(f"  {'Sensor':<10} {'Std (deg)':>10}")
    print('  ' + '-' * 22)

    for part, q in q_static.items():
        valid = ~np.isnan(q).any(axis=1)
        if valid.sum() < 2:
            continue
        q_valid = q[valid]
        q_mean = qmt.averageQuat(q_valid)
        q_dev = qmt.qmult(qmt.qinv(q_mean), q_valid)
        dev_angles = np.abs(np.degrees(qmt.quatAngle(q_dev)))
        ori_std = np.std(dev_angles)
        print(f"  {part:<10} {ori_std:>10.3f}")
        results[f'{part}_static_std'] = ori_std

    return results


# ============================================================================
# Validation 4: Knee q_rel vs IK (sanity check)
# ============================================================================

def validate_knee_angle(subject_id='Subject08'):
    """Compare knee angle from optical q_rel vs IK ground truth."""
    trc_path = str(ROOT / f'data/{subject_id}/walking/Mocap/walking.trc')
    mot_path = str(ROOT / f'data/{subject_id}/walking/Mocap/ikResults/walking_IK.mot')

    q_rel_optical = compute_q_rel_optical(trc_path, joint='knee')

    mot_df = load_mot(mot_path)
    gt_angle = mot_df['knee_angle_r'].values

    n = min(len(q_rel_optical), len(gt_angle))
    q_rel_optical = q_rel_optical[:n]
    gt_angle = gt_angle[:n]

    # PCA to find dominant rotation axis
    rot_vecs = qmt.quatToRotVec(q_rel_optical)
    valid = ~np.isnan(rot_vecs).any(axis=1)
    rot_centered = rot_vecs[valid] - rot_vecs[valid].mean(axis=0)
    _, _, Vt = np.linalg.svd(rot_centered, full_matrices=False)
    pca_axis = Vt[0]

    optical_angle = calculate_joint_angle(q_rel_optical, pca_axis)

    corr = np.corrcoef(optical_angle[valid], gt_angle[valid])[0, 1]
    if corr < 0:
        optical_angle = -optical_angle
        pca_axis = -pca_axis
        corr = -corr

    optical_shifted = optical_angle - np.nanmean(optical_angle[valid]) + np.mean(gt_angle[valid])
    rmse = np.sqrt(np.nanmean((optical_shifted[valid] - gt_angle[valid])**2))

    print(f"\nKnee angle validation ({subject_id}):")
    print(f"  Correlation: {corr:.4f}")
    print(f"  RMSE (after offset): {rmse:.2f} deg")
    print(f"  PCA axis: [{pca_axis[0]:.3f}, {pca_axis[1]:.3f}, {pca_axis[2]:.3f}]")

    # Plot
    time = np.arange(n) / FS
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, gt_angle, label='IK GT', alpha=0.8)
    ax.plot(time, optical_shifted, label='Optical q_rel', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Knee angle (deg)')
    ax.set_title(f'Knee Angle: Optical q_rel vs IK ({subject_id}) — r={corr:.4f}, RMSE={rmse:.1f}°')
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f'results/knee_optical_validation_{subject_id}.png', dpi=150)
    print(f"Saved: results/knee_optical_validation_{subject_id}.png")

    return corr, rmse


if __name__ == '__main__':
    (ROOT / 'results').mkdir(exist_ok=True)

    print("=" * 60)
    print("VALIDATION 1: VQF 6D inclination comparison (tilt, 2 DOF)")
    print("=" * 60)
    inclination_results = validate_inclination()

    print("\n" + "=" * 60)
    print("VALIDATION 1b: Method q_rel vs optical q_rel (3 DOF)")
    print("=" * 60)
    qrel_results = {}
    for method_fn, method_name in [(run_kf_gframe, 'KF gframe'), (run_rnno, 'RNNO')]:
        qrel_results[method_name] = validate_qrel_method(method_fn, method_name)
        print()

    print("\n" + "=" * 60)
    print("VALIDATION 2: Xsens orientation comparison (3 DOF)")
    print("=" * 60)
    xsens_results = validate_xsens_comparison()

    print("\n" + "=" * 60)
    print("VALIDATION 3: Marker triad geometric consistency")
    print("=" * 60)
    geo_results = validate_geometry()

    print("\n" + "=" * 60)
    print("VALIDATION 4: Knee q_rel vs IK (sanity check)")
    print("=" * 60)
    knee_corr, knee_rmse = validate_knee_angle()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if inclination_results:
        max_incl = max(r['mean'] for r in inclination_results.values())
        pelvis_incl = inclination_results.get('pelvis', {}).get('mean', float('inf'))
        print(f"  Max inclination error: {max_incl:.2f}° mean (target < 3°) "
              f"{'PASS' if max_incl < 3 else 'FAIL'}")
        print(f"  Pelvis inclination:    {pelvis_incl:.1f}° mean (target < 2°) "
              f"{'PASS' if pelvis_incl < 2 else 'FAIL'}")
    if qrel_results:
        for method_name, joints in qrel_results.items():
            for joint, r in joints.items():
                print(f"  {method_name} q_rel {joint:<6} err:  {r['mean']:.1f}° mean, P95={r['p95']:.1f}°")
    if xsens_results:
        pelvis_mean = xsens_results.get('pelvis', {}).get('mean', float('inf'))
        min_omega_corr = min(r['omega_corr'] for r in xsens_results.values())
        print(f"  Pelvis Xsens error:    {pelvis_mean:.1f}° (target < 8°) "
              f"{'PASS' if pelvis_mean < 8 else 'FAIL'}")
        print(f"  Min ω correlation:     {min_omega_corr:.3f} (target > 0.7) "
              f"{'PASS' if min_omega_corr > 0.7 else 'FAIL'}")
    print(f"  Knee q_rel vs IK:      r={knee_corr:.3f} (target > 0.95) "
          f"{'PASS' if knee_corr > 0.95 else 'FAIL'}")

"""RNNO-based joint angle estimation using the imt library."""
import numpy as np
from imt import methods as imt_methods

from .shared import calculate_joint_angle, parse_osim_calibration, calculate_joint_angle_model
from .axis import estimate_joint_axis


def compute_rnno_orientation(
    acc_prox,  # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,  # proximal gyroscope (N, 3) or (3, N)
    acc_dist,  # distal accelerometer (N, 3) or (3, N)
    gyr_dist,  # distal gyroscope (N, 3) or (3, N)
    fs,        # sampling frequency in Hz (must be 100 Hz)
):
    """Compute relative orientation using RNNO, returns q_rel (N, 4)."""
    if abs(fs - 100.0) > 0.1:
        raise ValueError(f"RNNO requires 100 Hz, got {fs} Hz")

    # Ensure shape is (N, 3)
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    N = len(acc_prox)

    rnno = imt_methods.RNNO_rO()
    rnno.setTs(1.0 / fs)
    rnno.reset()

    q_rel, _ = rnno.apply(
        T=N,
        acc1=acc_prox,
        acc2=acc_dist,
        gyr1=gyr_prox,
        gyr2=gyr_dist,
        mag1=None,
        mag2=None,
    )
    return q_rel


def run_rnno(
    acc_prox,                  # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,                  # proximal gyroscope (N, 3) or (3, N)
    acc_dist,                  # distal accelerometer (N, 3) or (3, N)
    gyr_dist,                  # distal gyroscope (N, 3) or (3, N)
    fs,                        # sampling frequency in Hz (must be 100 Hz)
    axis_mode='olsson',        # 'olsson', 'optimize', or 'opensim'
    gt_angles=None,            # ground truth for 'optimize' mode (degrees)
    calib_samples=None,        # samples for axis optimization (default: full dataset)
    joint=None,                # joint name for 'opensim'/'model' mode ('knee' or 'ankle')
    q_rel=None,                # pre-computed relative quaternion (skips RNNO)
    model_path=None,           # path to calibrated .osim for 'model' mode
    prox_imu=None,             # proximal IMU frame name for 'model' mode
    dist_imu=None,             # distal IMU frame name for 'model' mode
):
    """Estimate joint angle using RNNO, returns (angle_deg, r1, r2, jhat, q_rel)."""
    # Ensure shape is (N, 3) for axis estimation
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # Compute orientation if not provided
    if q_rel is None:
        q_rel = compute_rnno_orientation(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)

    # Model-based angle calculation bypasses axis estimation entirely
    if axis_mode == 'model':
        joint_name = joint or 'knee_r'
        if '_r' not in joint_name and '_l' not in joint_name:
            joint_name += '_r'
        calib = parse_osim_calibration(model_path, joint_name, prox_imu, dist_imu)
        angle_deg = calculate_joint_angle_model(q_rel, calib)
        jhat = calib['R_prox_proxIMU'].T @ calib['R_parent_offset'] @ calib['rot_axis']
        jhat = jhat / np.linalg.norm(jhat)
        return angle_deg, None, None, jhat, q_rel

    # Map legacy axis_mode names to unified method names
    method_map = {'optimize': 'optimized', 'pca_omega': 'pca_rotvec'}
    axis_method = method_map.get(axis_mode, axis_mode)

    # Joint axis estimation using unified API
    jhat = estimate_joint_axis(
        q_rel, axis_method=axis_method, gt_angles=gt_angles,
        acc_prox=acc_prox, gyr_prox=gyr_prox, acc_dist=acc_dist, gyr_dist=gyr_dist,
        correct_sign=True, joint=joint, calib_samples=calib_samples
    )
    angle_deg = calculate_joint_angle(q_rel, jhat)

    return angle_deg, None, None, jhat, q_rel


def run_rnno_all_variants(
    acc_prox,             # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,             # proximal gyroscope (N, 3) or (3, N)
    acc_dist,             # distal accelerometer (N, 3) or (3, N)
    gyr_dist,             # distal gyroscope (N, 3) or (3, N)
    fs,                   # sampling frequency in Hz (must be 100 Hz)
    gt_angles=None,       # ground truth for 'optimized' mode
    calib_samples=None,   # samples for optimization
    joint=None,           # joint name for 'opensim'/'model' mode
    model_path=None,      # path to calibrated .osim for 'model' mode
    prox_imu=None,        # proximal IMU frame name for 'model' mode
    dist_imu=None,        # distal IMU frame name for 'model' mode
):
    """Run all RNNO variants with shared orientation, returns dict of (angle_deg, jhat, q_rel)."""
    # Compute orientation once (expensive)
    q_rel = compute_rnno_orientation(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)

    # Ensure shape is (N, 3) for axis estimation
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    results = {}

    # Olsson (always computed)
    angle_deg, _, _, jhat, _ = run_rnno(
        acc_prox, gyr_prox, acc_dist, gyr_dist, fs,
        axis_mode='olsson', q_rel=q_rel
    )
    results['olsson'] = (angle_deg, jhat, q_rel)

    # Optimized (if ground truth provided)
    if gt_angles is not None:
        angle_deg, _, _, jhat, _ = run_rnno(
            acc_prox, gyr_prox, acc_dist, gyr_dist, fs,
            axis_mode='optimize', gt_angles=gt_angles, calib_samples=calib_samples, q_rel=q_rel
        )
        results['optimized'] = (angle_deg, jhat, q_rel)

    # OpenSim (if joint provided)
    if joint is not None:
        angle_deg, _, _, jhat, _ = run_rnno(
            acc_prox, gyr_prox, acc_dist, gyr_dist, fs,
            axis_mode='opensim', joint=joint, q_rel=q_rel
        )
        results['opensim'] = (angle_deg, jhat, q_rel)

    # PCA (always computed, requires only q_rel)
    angle_deg, _, _, jhat, _ = run_rnno(
        acc_prox, gyr_prox, acc_dist, gyr_dist, fs,
        axis_mode='pca_omega', q_rel=q_rel
    )
    results['pca'] = (angle_deg, jhat, q_rel)

    # Model (if calibrated model provided)
    if model_path is not None and prox_imu is not None and dist_imu is not None:
        joint_name = (joint or 'knee') + '_r'
        calib = parse_osim_calibration(model_path, joint_name, prox_imu, dist_imu)
        angle_deg = calculate_joint_angle_model(q_rel, calib)
        jhat = calib['R_prox_proxIMU'].T @ calib['R_parent_offset'] @ calib['rot_axis']
        jhat = jhat / np.linalg.norm(jhat)
        results['model'] = (angle_deg, jhat, q_rel)

    return results

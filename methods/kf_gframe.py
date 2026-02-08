"""Kalman filter with gravity frame constraints for dual-IMU joint angle estimation."""
import numpy as np
import qmt

from dfjimu.mekf_acc import mekf_acc
from dfjimu import estimate_lever_arms
from .shared import calculate_joint_angle, parse_osim_calibration, calculate_joint_angle_model
from .axis import estimate_joint_axis

# Noise parameters matching the original process_orientation_KF_Gframe
_Q_COV = np.ones(6) * 1e-2              # process noise diagonal
_R_DIAG = 2.0 * 0.35**2 * 10           # measurement noise (2 * cov_lnk_scale)
_P_INIT_DIAG = 1.0                      # initial covariance diagonal




def run_kf_gframe(
    acc_prox,          # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,          # proximal gyroscope (N, 3) or (3, N)
    acc_dist,          # distal accelerometer (N, 3) or (3, N)
    gyr_dist,          # distal gyroscope (N, 3) or (3, N)
    fs,                # sampling frequency in Hz
    r1=None,           # lever arm 1, auto-estimated if None
    r2=None,           # lever arm 2, auto-estimated if None
    axis_mode='fixed',    # 'fixed', 'olsson', 'optimize', 'opensim', or 'model'
    euler_axes='zyx',     # Euler axes for 'fixed' mode
    gt_angles=None,       # ground truth for 'optimize' mode (degrees)
    calib_samples=3000,   # samples for axis optimization
    joint=None,           # 'knee' or 'ankle' for 'opensim' mode
    model_path=None,      # path to calibrated .osim for 'model' mode
    prox_imu=None,        # proximal IMU frame name for 'model' mode
    dist_imu=None,        # distal IMU frame name for 'model' mode
):
    """Estimate joint angle using KF with gravity constraints, returns (angle_deg, r1, r2, jhat, q_rel)."""
    # Ensure shape is (N, 3)
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # Estimate lever arms if not provided
    if r1 is None or r2 is None:
        r1, r2 = estimate_lever_arms(gyr_prox, gyr_dist, acc_prox, acc_dist, fs)

    # Run MEKF-acc
    q1_all, q2_all = mekf_acc(
        gyr_prox, gyr_dist, acc_prox, acc_dist, r1, r2, fs,
        np.array([1.0, 0, 0, 0]), _Q_COV, _R_DIAG, _P_INIT_DIAG,
    )

    # Compute relative quaternion
    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)

    # Model-based angle calculation bypasses axis estimation entirely
    if axis_mode == 'model':
        joint_name = joint or 'knee_r'
        if '_r' not in joint_name and '_l' not in joint_name:
            joint_name += '_r'
        calib = parse_osim_calibration(model_path, joint_name, prox_imu, dist_imu)
        angle_deg = calculate_joint_angle_model(q_rel, calib)
        jhat = calib['R_prox_proxIMU'].T @ calib['R_parent_offset'] @ calib['rot_axis']
        jhat = jhat / np.linalg.norm(jhat)
        return angle_deg, r1, r2, jhat, q_rel

    # Map legacy axis_mode names to unified method names
    method_map = {'optimize': 'optimized', 'pca_omega': 'pca_rotvec'}
    axis_method = method_map.get(axis_mode, axis_mode)

    # Joint axis estimation using unified API
    if axis_mode == 'fixed':
        # Fixed mode uses Euler angles directly
        angle_deg = np.degrees(qmt.eulerAngles(q_rel, axes=euler_axes)[:, 0])  # type: ignore[index]
        jhat = estimate_joint_axis(q_rel, axis_method='fixed', euler_axes=euler_axes)
    else:
        # All other modes use estimate_joint_axis with (N, 3) shaped IMU data
        jhat = estimate_joint_axis(
            q_rel, axis_method=axis_method, gt_angles=gt_angles,
            acc_prox=acc_prox, gyr_prox=gyr_prox, acc_dist=acc_dist, gyr_dist=gyr_dist,
            correct_sign=True, joint=joint, calib_samples=calib_samples
        )
        angle_deg = calculate_joint_angle(q_rel, jhat)

    return angle_deg, r1, r2, jhat, q_rel


def run_kf_gframe_all_variants(
    acc_prox,             # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,             # proximal gyroscope (N, 3) or (3, N)
    acc_dist,             # distal accelerometer (N, 3) or (3, N)
    gyr_dist,             # distal gyroscope (N, 3) or (3, N)
    fs,                   # sampling frequency in Hz
    r1=None,              # lever arm 1, auto-estimated if None
    r2=None,              # lever arm 2, auto-estimated if None
    gt_angles=None,       # ground truth for 'optimized' mode
    calib_samples=3000,   # samples for optimization
    joint=None,           # joint name for 'opensim' mode
    model_path=None,      # path to calibrated .osim for 'model' variant
    prox_imu=None,        # proximal IMU frame name for 'model' variant
    dist_imu=None,        # distal IMU frame name for 'model' variant
):
    """Run all KF_Gframe variants with shared KF computation, returns dict of (angle_deg, jhat, q_rel)."""
    # Ensure shape is (N, 3)
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # Estimate lever arms once
    if r1 is None or r2 is None:
        r1, r2 = estimate_lever_arms(gyr_prox, gyr_dist, acc_prox, acc_dist, fs)

    # Run MEKF-acc once (expensive)
    q1_all, q2_all = mekf_acc(
        gyr_prox, gyr_dist, acc_prox, acc_dist, r1, r2, fs,
        np.array([1.0, 0, 0, 0]), _Q_COV, _R_DIAG, _P_INIT_DIAG,
    )

    # Compute relative quaternion once
    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)

    results = {}

    # Olsson (always computed)
    jhat = estimate_joint_axis(
        q_rel, axis_method='olsson',
        acc_prox=acc_prox, gyr_prox=gyr_prox,
        acc_dist=acc_dist, gyr_dist=gyr_dist, correct_sign=True
    )
    angle_deg = calculate_joint_angle(q_rel, jhat)
    results['olsson'] = (angle_deg, jhat, q_rel)

    # Optimized (if ground truth provided)
    if gt_angles is not None:
        jhat = estimate_joint_axis(
            q_rel, axis_method='optimized', gt_angles=gt_angles,
            acc_prox=acc_prox, gyr_prox=gyr_prox,
            acc_dist=acc_dist, gyr_dist=gyr_dist,
            correct_sign=True, calib_samples=calib_samples
        )
        angle_deg = calculate_joint_angle(q_rel, jhat)
        results['optimized'] = (angle_deg, jhat, q_rel)

    # PCA (always computed, requires only q_rel)
    jhat = estimate_joint_axis(q_rel, axis_method='pca_rotvec', correct_sign=True)
    angle_deg = calculate_joint_angle(q_rel, jhat)
    results['pca'] = (angle_deg, jhat, q_rel)

    # OpenSim (if joint provided)
    if joint is not None:
        jhat = estimate_joint_axis(q_rel, axis_method='opensim', joint=joint, correct_sign=True)
        angle_deg = calculate_joint_angle(q_rel, jhat)
        results['opensim'] = (angle_deg, jhat, q_rel)

    # Model-based (if model_path provided)
    if model_path is not None and prox_imu is not None and dist_imu is not None:
        joint_name = (joint or 'knee') + '_r'
        calib = parse_osim_calibration(model_path, joint_name, prox_imu, dist_imu)
        angle_deg = calculate_joint_angle_model(q_rel, calib)
        jhat = calib['R_prox_proxIMU'].T @ calib['R_parent_offset'] @ calib['rot_axis']
        jhat = jhat / np.linalg.norm(jhat)
        results['model'] = (angle_deg, jhat, q_rel)

    return results




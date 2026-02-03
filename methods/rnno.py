"""RNNO-based joint angle estimation using the imt library."""
import numpy as np
import imt
from scipy.optimize import minimize

from .shared import olsson_estimate_hinge_joint_axes, calculate_joint_angle

OPENSIM_JOINT_AXES = {
    'knee': np.array([0, 0, 1]),
    'ankle': np.array([-0.6, -0.3, -0.75]),
}


def compute_rnno_orientation(acc_prox, gyr_prox, acc_dist, gyr_dist, fs):
    """Compute relative orientation using RNNO (expensive - call once, reuse q_rel).

    Args:
        acc_prox: Proximal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_prox: Proximal IMU gyroscope data, shape (N, 3) or (3, N)
        acc_dist: Distal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_dist: Distal IMU gyroscope data, shape (N, 3) or (3, N)
        fs: Sampling frequency in Hz (must be 100 Hz)

    Returns:
        q_rel: Relative quaternion (N, 4)
    """
    if abs(fs - 100.0) > 0.1:
        raise ValueError(f"RNNO requires 100 Hz, got {fs} Hz")

    # Ensure shape is (N, 3)
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    N = len(acc_prox)

    rnno = imt.methods.RNNO_rO()
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


def run_rnno(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, axis_mode='olsson',
             gt_angles=None, calib_samples=None, joint=None, q_rel=None):
    """Estimate joint angle using RNNO relative orientation estimation.

    Args:
        acc_prox: Proximal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_prox: Proximal IMU gyroscope data, shape (N, 3) or (3, N)
        acc_dist: Distal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_dist: Distal IMU gyroscope data, shape (N, 3) or (3, N)
        fs: Sampling frequency in Hz (must be 100 Hz)
        axis_mode: Joint axis estimation mode: 'olsson', 'optimize', or 'opensim'.
        gt_angles: Ground truth angles for 'optimize' mode (degrees).
        calib_samples: Number of samples for axis optimization (default: full dataset).
        joint: Joint name ('knee' or 'ankle') - required for 'opensim' mode.
        q_rel: Pre-computed relative quaternion (optional, skips RNNO computation).

    Returns:
        (angle_deg, r1, r2, jhat, q_rel): Joint angle in degrees, None for lever arms,
        axis vector, and relative quaternion.
    """
    # Ensure shape is (N, 3) for axis estimation
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # Compute orientation if not provided
    if q_rel is None:
        q_rel = compute_rnno_orientation(acc_prox, gyr_prox, acc_dist, gyr_dist, fs)

    # Joint axis estimation based on mode
    if axis_mode == 'olsson':
        jhat, _ = olsson_estimate_hinge_joint_axes(acc_prox, acc_dist, gyr_prox, gyr_dist)
        angle_deg = calculate_joint_angle(q_rel, jhat)

    elif axis_mode == 'optimize':
        if gt_angles is None:
            raise ValueError("axis_mode='optimize' requires gt_angles")
        jhat = _optimize_joint_axis(q_rel, gt_angles, calib_samples)
        angle_deg = calculate_joint_angle(q_rel, jhat)

    elif axis_mode == 'opensim':
        if joint is None:
            raise ValueError("axis_mode='opensim' requires joint parameter")
        if joint not in OPENSIM_JOINT_AXES:
            raise ValueError(f"Unknown joint: {joint}")
        jhat = OPENSIM_JOINT_AXES[joint].copy()
        angle_deg = calculate_joint_angle(q_rel, jhat)

    else:
        raise ValueError(f"Unknown axis_mode: {axis_mode}")

    return angle_deg, None, None, jhat, q_rel


def _optimize_joint_axis(q_rel, gt_angles, calib_samples):
    """Find joint axis minimizing RMSE against ground truth."""
    if calib_samples is None:
        calib_samples = len(gt_angles)
    n = min(calib_samples, len(gt_angles), len(q_rel))
    q_calib, gt_calib = q_rel[:n], gt_angles[:n]

    def spherical_to_cart(theta, phi):
        return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    def objective(params):
        jhat = spherical_to_cart(*params)
        angle_est = calculate_joint_angle(q_calib, jhat)
        return np.sqrt(np.mean((gt_calib - angle_est)**2))

    # 4x8 grid over spherical coordinates for better coverage
    init_points = [(theta, phi)
                   for theta in np.linspace(0.01, np.pi - 0.01, 4)
                   for phi in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    best = min(
        (minimize(objective, init, method='L-BFGS-B',
                  bounds=[(0, np.pi), (-np.pi, np.pi)])
         for init in init_points),
        key=lambda r: r.fun
    )
    jhat = spherical_to_cart(*best.x)

    # Sign check via correlation - pick sign with better correlation to GT
    angle_pos = calculate_joint_angle(q_calib, jhat)
    angle_neg = calculate_joint_angle(q_calib, -jhat)
    if np.corrcoef(angle_neg, gt_calib)[0, 1] > np.corrcoef(angle_pos, gt_calib)[0, 1]:
        jhat = -jhat

    return jhat


def run_rnno_olsson(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, q_rel=None):
    """RNNO with Olsson joint axis estimation."""
    return run_rnno(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, axis_mode='olsson', q_rel=q_rel)


def run_rnno_optimized(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, gt_angles, calib_samples=None, q_rel=None):
    """RNNO with optimized joint axis (requires ground truth).

    By default uses full dataset for optimization to avoid overfitting to short windows.
    """
    if calib_samples is None:
        calib_samples = len(gt_angles)
    return run_rnno(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, axis_mode='optimize',
                    gt_angles=gt_angles, calib_samples=calib_samples, q_rel=q_rel)


def run_rnno_opensim(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, joint, q_rel=None):
    """RNNO with precomputed OpenSim joint axis."""
    return run_rnno(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, axis_mode='opensim', joint=joint, q_rel=q_rel)


def run_rnno_all_variants(acc_prox, gyr_prox, acc_dist, gyr_dist, fs,
                          gt_angles=None, calib_samples=None, joint=None):
    """Run all RNNO variants with shared orientation computation.

    Computes RNNO orientation once and reuses for all axis modes.

    Args:
        acc_prox, gyr_prox, acc_dist, gyr_dist: IMU data
        fs: Sampling frequency (must be 100 Hz)
        gt_angles: Ground truth angles for optimized mode (optional)
        calib_samples: Samples for optimization (optional)
        joint: Joint name for opensim mode (optional)

    Returns:
        dict with keys 'olsson', 'optimized' (if gt_angles), 'opensim' (if joint).
        Each value: (angle_deg, jhat, q_rel)
    """
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

    return results

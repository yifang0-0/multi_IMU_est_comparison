"""Joint axis estimation methods registry."""
import numpy as np
from scipy.optimize import minimize

from constants import FS
from .shared import olsson_estimate_hinge_joint_axes, calculate_joint_angle

AXIS_METHODS = {}


def register_axis_method(
    name,          # method name for lookup
    requires_gt=False,   # needs gt_angles parameter
    requires_imu=False,  # needs raw IMU data (acc/gyr)
):
    """Decorator to register axis estimation methods."""
    def decorator(func):
        AXIS_METHODS[name] = {'func': func, 'requires_gt': requires_gt, 'requires_imu': requires_imu}
        return func
    return decorator


def get_axis_method(name):
    """Get a registered axis method by name."""
    if name not in AXIS_METHODS:
        raise ValueError(f"Unknown axis method: {name}. Available: {list(AXIS_METHODS.keys())}")
    return AXIS_METHODS[name]


def list_axis_methods():
    """List all registered axis method names."""
    return list(AXIS_METHODS.keys())


def estimate_joint_axis(
    q_rel,              # relative quaternion array (N, 4)
    axis_method='olsson',
    gt_angles=None,     # ground truth angles in degrees
    acc_prox=None,      # proximal accelerometer (N, 3)
    gyr_prox=None,      # proximal gyroscope (N, 3)
    acc_dist=None,      # distal accelerometer (N, 3)
    gyr_dist=None,      # distal gyroscope (N, 3)
    correct_sign=True,  # flip axis to maximize correlation with gt_angles
    joint=None,         # 'knee' or 'ankle'
    **kwargs,
):
    """Returns normalized joint axis vector (3,)."""
    method = get_axis_method(axis_method)
    call_kwargs = {'q_rel': q_rel, 'joint': joint, **kwargs}

    if method['requires_gt']:
        if gt_angles is None:
            raise ValueError(f"axis_method='{axis_method}' requires gt_angles")
        call_kwargs['gt_angles'] = gt_angles

    if method['requires_imu']:
        if any(x is None for x in [acc_prox, gyr_prox, acc_dist, gyr_dist]):
            raise ValueError(f"axis_method='{axis_method}' requires IMU data")
        call_kwargs.update(acc_prox=acc_prox, gyr_prox=gyr_prox, acc_dist=acc_dist, gyr_dist=gyr_dist)

    jhat = method['func'](**call_kwargs)
    jhat = jhat / np.linalg.norm(jhat)

    if correct_sign and gt_angles is not None:
        angle_pos = calculate_joint_angle(q_rel, jhat)
        angle_neg = calculate_joint_angle(q_rel, -jhat)
        n = min(len(angle_pos), len(gt_angles))
        if np.corrcoef(angle_neg[:n], gt_angles[:n])[0, 1] > np.corrcoef(angle_pos[:n], gt_angles[:n])[0, 1]:
            jhat = -jhat

    return jhat


# =============================================================================
# Built-in Methods
# =============================================================================

@register_axis_method('olsson', requires_imu=True)
def _axis_olsson(acc_prox, gyr_prox, acc_dist, gyr_dist, **kwargs):
    """Olsson hinge joint axis estimation from raw IMU data."""
    jhat, _ = olsson_estimate_hinge_joint_axes(acc_prox, acc_dist, gyr_prox, gyr_dist)
    return jhat


@register_axis_method('optimized', requires_gt=True)
def _axis_optimized(
    q_rel,                # relative quaternions (N, 4)
    gt_angles,            # ground truth angles in degrees
    calib_samples=None,   # samples for calibration (None = all)
    **kwargs,
):
    """Optimize joint axis by minimizing RMSE against ground truth."""
    n = min(calib_samples or len(gt_angles), len(gt_angles), len(q_rel))
    q_calib, gt_calib = q_rel[:n], gt_angles[:n]

    def spherical_to_cart(theta, phi):
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    def objective(params):
        angle_est = calculate_joint_angle(q_calib, spherical_to_cart(*params))
        return np.sqrt(np.mean((gt_calib - angle_est)**2))

    # 4x8 grid search over sphere
    init_points = [(theta, phi)
                   for theta in np.linspace(0.01, np.pi - 0.01, 4)
                   for phi in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    best = min((minimize(objective, init, method='L-BFGS-B', bounds=[(0, np.pi), (-np.pi, np.pi)])
                for init in init_points), key=lambda r: r.fun)
    jhat = spherical_to_cart(*best.x)

    # Sign correction for direct calls (also done in estimate_joint_axis)
    angle_pos = calculate_joint_angle(q_calib, jhat)
    angle_neg = calculate_joint_angle(q_calib, -jhat)
    if np.corrcoef(angle_neg, gt_calib)[0, 1] > np.corrcoef(angle_pos, gt_calib)[0, 1]:
        jhat = -jhat

    return jhat


@register_axis_method('fixed')
def _axis_fixed(
    euler_axes='zyx',  # euler convention determining primary axis
    **kwargs,
):
    """Return fixed axis based on Euler angle convention."""
    axis_map = {
        'zyx': [0, 0, 1], 'xyz': [1, 0, 0], 'yxz': [0, 1, 0],
        'zxy': [0, 0, 1], 'xzy': [1, 0, 0], 'yzx': [0, 1, 0],
    }
    return np.array(axis_map.get(euler_axes, [0, 0, 1]), dtype=float)


@register_axis_method('z_axis')
def _axis_z(**kwargs):
    """Return [0, 0, 1] baseline axis."""
    return np.array([0.0, 0.0, 1.0])


# =============================================================================
# PCA-Based Methods
# =============================================================================

def _quaternion_to_rotvec(q_array):
    """Convert quaternions (N, 4) to rotation vectors (N, 3)."""
    q = np.atleast_2d(q_array).copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True) + np.finfo(float).eps

    # Ensure positive scalar part for consistent representation
    q[q[:, 0] < 0] *= -1

    w, xyz = q[:, 0], q[:, 1:4]
    theta = np.arccos(np.clip(w, -1.0, 1.0))  # half-angle
    sin_theta = np.sin(theta)

    # Taylor expansion for small angles: theta/sin(theta) ≈ 1 + theta²/6
    scale = np.where(np.abs(sin_theta) < 1e-8,
                     1.0 + theta**2 / 6.0,
                     theta / (sin_theta + np.finfo(float).eps))

    return xyz * scale[:, np.newaxis]


def _pca_dominant_axis(data):  # (N, 3) array
    """Extract dominant direction via PCA."""
    centered = data - np.mean(data, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    dominant = eigenvectors[:, -1]  # largest eigenvalue is last
    return dominant / np.linalg.norm(dominant)


@register_axis_method('pca_rotvec')
def _axis_pca_rotvec(q_rel, **kwargs):
    """PCA on rotation vectors to find dominant rotation axis."""
    return _pca_dominant_axis(_quaternion_to_rotvec(q_rel))


@register_axis_method('pca_omega')
def _axis_pca_omega(
    q_rel,       # relative quaternions (N, 4)
    fs=FS,       # sampling frequency in Hz
    **kwargs,
):
    """PCA on angular velocity derived from quaternion differences."""
    if hasattr(fs, '__len__'):  # handle numpy array from cache
        fs = float(fs)
    rotvec = _quaternion_to_rotvec(q_rel)
    omega = (rotvec[2:] - rotvec[:-2]) * (fs / 2.0)  # central difference
    return _pca_dominant_axis(omega)

"""Kalman filter with gravity frame constraints for dual-IMU joint angle estimation."""
import numpy as np
import qmt
from scipy.optimize import minimize

from constants import FS, T as DT, ACC_OUTLIER_THRESHOLD
from calTools import (
    integrateGyr, quatmultiply, EXPq, quat2matrix, crossM,
    approx_derivative, calc_acc_at_center
)
from .shared import olsson_estimate_hinge_joint_axes, calculate_joint_angle




def run_kf_gframe(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, r1=None, r2=None,
                  cov_w_scale=1e-2, cov_lnk_scale=0.35**2 * 10,
                  axis_mode='fixed', euler_axes='zyx',
                  gt_angles=None, calib_samples=3000):
    """Estimate joint angle using Kalman filter with gravity frame constraints.

    Args:
        acc_prox: Proximal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_prox: Proximal IMU gyroscope data, shape (N, 3) or (3, N)
        acc_dist: Distal IMU accelerometer data, shape (N, 3) or (3, N)
        gyr_dist: Distal IMU gyroscope data, shape (N, 3) or (3, N)
        fs: Sampling frequency in Hz
        r1, r2: Lever arms. If None, auto-estimated from data.
        cov_w_scale: Scale for process noise covariance (default: 1e-2).
        cov_lnk_scale: Scale for measurement noise covariance (default: 0.35**2 * 10).
        axis_mode: Joint axis estimation mode: 'fixed', 'olsson', or 'optimize'.
        euler_axes: Euler angle extraction axes for 'fixed' mode (default: 'zyx').
        gt_angles: Ground truth angles for 'optimize' mode (degrees).
        calib_samples: Number of samples for axis optimization (default: 3000).

    Returns:
        (angle_deg, r1, r2, jhat, q_rel): Joint angle in degrees, lever arms, axis vector, and relative quaternion.
    """
    # Ensure shape is (3, N) for KF processing
    if acc_prox.shape[0] != 3:
        acc1, gyr1 = acc_prox.T, gyr_prox.T
        acc2, gyr2 = acc_dist.T, gyr_dist.T
    else:
        acc1, gyr1 = acc_prox, gyr_prox
        acc2, gyr2 = acc_dist, gyr_dist

    # Estimate lever arms if not provided
    if r1 is None or r2 is None:
        r1, r2 = estimate_lever_arms(acc1, gyr1, acc2, gyr2, fs)

    # Covariance matrices
    cov_w = np.eye(6) * cov_w_scale
    cov_lnk = np.eye(3) * cov_lnk_scale

    # Run KF with explicit parameters
    q1_all, q2_all, _ = process_orientation_KF_Gframe(
        data={
            'gyr_1': gyr1, 'gyr_2': gyr2,
            'acc_1': acc1, 'acc_2': acc2,
            'r1': r1, 'r2': r2
        },
        cov_w=cov_w,
        cov_lnk=cov_lnk,
    )

    # Compute relative quaternion
    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)

    # Joint axis estimation based on mode
    if axis_mode == 'fixed':
        angle_deg = np.degrees(qmt.eulerAngles(q_rel, axes=euler_axes)[:, 0])
        axis_map = {'zyx': [0, 0, 1], 'xyz': [1, 0, 0], 'yxz': [0, 1, 0],
                    'zxy': [0, 0, 1], 'xzy': [1, 0, 0], 'yzx': [0, 1, 0]}
        jhat = np.array(axis_map.get(euler_axes, [0, 0, 1]), dtype=float)

    elif axis_mode == 'olsson':
        # Olsson expects shape (N, 3), so transpose from (3, N)
        jhat, _ = olsson_estimate_hinge_joint_axes(acc1.T, acc2.T, gyr1.T, gyr2.T)
        angle_deg = calculate_joint_angle(q_rel, jhat)

    elif axis_mode == 'optimize':
        if gt_angles is None:
            raise ValueError("axis_mode='optimize' requires gt_angles")
        jhat = _optimize_joint_axis(q_rel, gt_angles, calib_samples)
        angle_deg = calculate_joint_angle(q_rel, jhat)

    else:
        raise ValueError(f"Unknown axis_mode: {axis_mode}")

    return angle_deg, r1, r2, jhat, q_rel


def run_kf_gframe_olsson(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, r1=None, r2=None):
    """KF with gravity frame using Olsson joint axis estimation."""
    return run_kf_gframe(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, r1=r1, r2=r2, axis_mode='olsson')


def run_kf_gframe_optimized(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, gt_angles, r1=None, r2=None, calib_samples=None):
    """KF with gravity frame using optimized joint axis (requires ground truth).

    By default uses full dataset for optimization to avoid overfitting to short windows.
    """
    if calib_samples is None:
        calib_samples = len(gt_angles)  # Use full dataset by default
    return run_kf_gframe(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, r1=r1, r2=r2,
                         axis_mode='optimize', gt_angles=gt_angles, calib_samples=calib_samples)


def _optimize_joint_axis(q_rel, gt_angles, calib_samples):
    """Find joint axis minimizing RMSE against ground truth."""
    n = min(calib_samples, len(gt_angles), len(q_rel))
    q_calib, gt_calib = q_rel[:n], gt_angles[:n]

    def spherical_to_cart(theta, phi):
        return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    def objective(params):
        jhat = spherical_to_cart(*params)
        angle_est = calculate_joint_angle(q_calib, jhat)
        return np.sqrt(np.mean((gt_calib - angle_est)**2))

    # Multi-start optimization from X, Y, Z axis initializations
    init_points = [(0.01, 0), (np.pi/2, 0), (np.pi/2, np.pi/2)]
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



def estimate_lever_arms(acc1, gyr1, acc2, gyr2, fs, iterations=25, step=0.7):
    """Estimate lever arms from dual-IMU data using Gauss-Newton optimization.

    Args:
        acc1, gyr1: Proximal IMU data, shape (3, N) or (N, 3)
        acc2, gyr2: Distal IMU data, shape (3, N) or (N, 3)
        fs: Sampling frequency
        iterations: Number of Gauss-Newton iterations
        step: Step size for updates

    Returns:
        r1, r2: Estimated lever arm vectors (3,)
    """
    # Ensure shape is (3, N)
    if acc1.shape[0] != 3:
        acc1, gyr1 = acc1.T, gyr1.T
        acc2, gyr2 = acc2.T, gyr2.T

    def get_dgyr(y, f):
        dy = np.zeros_like(y)
        dy[:, 2:-2] = (y[:, :-4] - 8*y[:, 1:-3] + 8*y[:, 3:-1] - y[:, 4:]) * (f/12)
        return dy

    def get_K(g, dg):
        num = g.shape[1]
        K_mat = np.zeros((3, 3, num))
        for i in range(num):
            w, alpha = g[:, i], dg[:, i]
            Sw = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            Sa = np.array([[0, -alpha[2], alpha[1]], [alpha[2], 0, -alpha[0]], [-alpha[1], alpha[0], 0]])
            K_mat[:, :, i] = Sw @ Sw + Sa
        return K_mat

    dg1, dg2 = get_dgyr(gyr1, fs), get_dgyr(gyr2, fs)
    K1, K2 = get_K(gyr1, dg1), get_K(gyr2, dg2)

    x = 0.1 * np.ones(6)
    num = gyr1.shape[1]

    for _ in range(iterations):
        e1 = acc1 - np.array([K1[:,:,i] @ x[0:3] for i in range(num)]).T
        e2 = acc2 - np.array([K2[:,:,i] @ x[3:6] for i in range(num)]).T

        n1 = np.linalg.norm(e1, axis=0)
        n2 = np.linalg.norm(e2, axis=0)
        eps = n1 - n2

        J = np.zeros((num, 6))
        for i in range(num):
            J[i, 0:3] = -(K1[:,:,i].T @ e1[:,i]) / (n1[i] + 1e-9)
            J[i, 3:6] = (K2[:,:,i].T @ e2[:,i]) / (n2[i] + 1e-9)

        G = J.T @ eps
        H = J.T @ J
        try:
            x -= step * np.linalg.solve(H + 1e-8*np.eye(6), G)
        except np.linalg.LinAlgError:
            break

    return x[0:3], x[3:6]


def process_orientation_KF_Gframe(
    data,
    q1_init=None,
    cov_w=None,
    cov_lnk=None,
    run_dynamic_update=True,
    run_measurement_update=True,
    use_raw_gyro=False,
):
    """Process orientation using Extended Kalman Filter with gravity frame constraints.

    Args:
        data: Dictionary containing sensor data (gyr_1, gyr_2, acc_1, acc_2, r1, r2).
        q1_init: Initial orientation quaternion (default: [1, 0, 0, 0]).
        cov_w: Process noise covariance (6x6). Default: np.eye(6) * 1e-2.
        cov_lnk: Measurement noise covariance (3x3). Default: np.eye(3) * 0.35**2 * 10.
        run_dynamic_update: Whether to run prediction step (default: True).
        run_measurement_update: Whether to run measurement update step (default: True).
        use_raw_gyro: Use raw gyro data instead of filtered (default: False).

    Returns:
        orientation_s1, orientation_s2, P_list: Estimated orientations and covariance history.
    """
    if q1_init is None:
        q1_init = np.array([1, 0, 0, 0])
    if cov_w is None:
        cov_w = np.eye(6) * 1e-2
    if cov_lnk is None:
        cov_lnk = np.eye(3) * 0.35**2 * 10

    T = DT
    gyr_1 = data['gyr_1']
    N = gyr_1.shape[1]
    gyr_2 = data['gyr_2']
    acc_1 = data['acc_1']
    acc_2 = data['acc_2']

    dgyr_1 = approx_derivative(gyr_1, FS)
    dgyr_2 = approx_derivative(gyr_2, FS)

    q_lin_s1_t = integrateGyr(gyr_1.T, q1_init)
    q_lin_s2_t = integrateGyr(gyr_2.T, q1_init)
    r_s1 = data['r1']
    r_s2 = data['r2']

    Q = np.eye(6) * cov_w[0][0]
    Pq_init = np.eye(6)
    R = np.eye(3) * 2 * cov_lnk[0][0]

    orientation_s1 = np.zeros((N, 4))
    orientation_s2 = np.zeros((N, 4))
    orientation_s1[0] = q1_init
    orientation_s2[0] = q1_init

    P_list = []
    P_list.append(Pq_init.copy())

    num_rejected = 0
    q_lin_s1_t = q1_init.copy()
    q_lin_s2_t = q1_init.copy()
    P_local = np.zeros((6, 6))
    P_local[0:6, 0:6] = Pq_init.copy()
    x0 = np.zeros((6, 1))
    x_local = x0.copy()

    for t in range(1, N):
        gyr_1_t = gyr_1[:, t-1:t]
        gyr_2_t = gyr_2[:, t-1:t]
        acc_1_t = acc_1[:, t:t+1]
        acc_2_t = acc_2[:, t:t+1]
        dgyr_1_t = dgyr_1[:, t:t+1]
        dgyr_2_t = dgyr_2[:, t:t+1]

        eta = np.zeros((2, 3))
        x_local[0:3, 0] = eta[0, 0:3]
        x_local[3:6, 0] = eta[1, 0:3]

        if np.any(acc_1_t > ACC_OUTLIER_THRESHOLD) or np.any(acc_2_t > ACC_OUTLIER_THRESHOLD):
            run_acc_inlimit = False
            num_rejected += 1
        else:
            run_acc_inlimit = True

        if run_dynamic_update:
            F = np.eye(6)
            q_lin_s1_t = quatmultiply(q_lin_s1_t, EXPq(T/2 * gyr_1_t))
            q_lin_s2_t = quatmultiply(q_lin_s2_t, EXPq(T/2 * gyr_2_t))

            if use_raw_gyro:
                gyr_1_t = data['raw_gyr_1'][:, t:t+1]
                gyr_2_t = data['raw_gyr_2'][:, t:t+1]
            else:
                gyr_1_t = gyr_1[:, t:t+1]
                gyr_2_t = gyr_2[:, t:t+1]

            AccG1_t, Cr1_t = calc_acc_at_center(gyr_1_t, dgyr_1_t, acc_1_t, r_s1)
            AccG2_t, Cr2_t = calc_acc_at_center(gyr_2_t, dgyr_2_t, acc_2_t, r_s2)

            G = np.zeros((6, 6))
            G[:3, :3] = T * quat2matrix(q_lin_s1_t)
            G[3:6, 3:6] = T * quat2matrix(q_lin_s2_t)
            P_local[0:6, 0:6] = F @ P_local[0:6, 0:6] @ F.T + G @ Q[0:6, 0:6] @ G.T
        else:
            AccG1_t, Cr1_t = calc_acc_at_center(gyr_1_t, dgyr_1_t, acc_1_t, r_s1)
            AccG2_t, Cr2_t = calc_acc_at_center(gyr_2_t, dgyr_2_t, acc_2_t, r_s2)

        if run_measurement_update and run_acc_inlimit:
            H = np.zeros((3, 6))
            vec1 = quat2matrix(q_lin_s1_t) @ AccG1_t
            vec2 = quat2matrix(q_lin_s2_t) @ AccG2_t
            H[0:3, 0:3] = crossM(vec1)
            H[0:3, 3:6] = -crossM(vec2)

            e = (quat2matrix(q_lin_s1_t) @ AccG1_t) - (quat2matrix(q_lin_s2_t) @ AccG2_t)

            S = H @ P_local @ H.T + R
            K = (P_local @ H.T) @ np.linalg.inv(S)
            x_local = x_local + K @ e
            eta[0, 0:3] = x_local[0:3, 0]
            eta[1, 0:3] = x_local[3:6, 0]

            q_lin_s1_t = quatmultiply(EXPq(eta[0, 0:3]/2), q_lin_s1_t)
            q_lin_s2_t = quatmultiply(EXPq(eta[1, 0:3]/2), q_lin_s2_t)

            P_local = P_local - K @ H @ P_local

        P_list.append(P_local.copy())
        orientation_s1[t] = q_lin_s1_t
        orientation_s2[t] = q_lin_s2_t

    return orientation_s1, orientation_s2, P_list


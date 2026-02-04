"""Kalman filter with gravity frame constraints for dual-IMU joint angle estimation."""
import numpy as np
import qmt

from constants import FS, T as DT, ACC_OUTLIER_THRESHOLD
from calTools import (
    integrateGyr, quatmultiply, EXPq, quat2matrix, crossM,
    approx_derivative, calc_acc_at_center
)
from .shared import calculate_joint_angle
from .axis import estimate_joint_axis, OPENSIM_JOINT_AXES




def run_kf_gframe(
    acc_prox,          # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,          # proximal gyroscope (N, 3) or (3, N)
    acc_dist,          # distal accelerometer (N, 3) or (3, N)
    gyr_dist,          # distal gyroscope (N, 3) or (3, N)
    fs,                # sampling frequency in Hz
    r1=None,           # lever arm 1, auto-estimated if None
    r2=None,           # lever arm 2, auto-estimated if None
    cov_w_scale=1e-2,
    cov_lnk_scale=0.35**2 * 10,
    axis_mode='fixed',    # 'fixed', 'olsson', 'optimize', or 'opensim'
    euler_axes='zyx',     # Euler axes for 'fixed' mode
    gt_angles=None,       # ground truth for 'optimize' mode (degrees)
    calib_samples=3000,   # samples for axis optimization
    joint=None,           # 'knee' or 'ankle' for 'opensim' mode
):
    """Estimate joint angle using KF with gravity constraints, returns (angle_deg, r1, r2, jhat, q_rel)."""
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
            acc_prox=acc1.T, gyr_prox=gyr1.T, acc_dist=acc2.T, gyr_dist=gyr2.T,
            correct_sign=True, joint=joint, calib_samples=calib_samples
        )
        angle_deg = calculate_joint_angle(q_rel, jhat)

    return angle_deg, r1, r2, jhat, q_rel


def estimate_lever_arms(
    acc1, gyr1,        # proximal IMU data (3, N) or (N, 3)
    acc2, gyr2,        # distal IMU data (3, N) or (N, 3)
    fs,                # sampling frequency
    iterations=25,     # Gauss-Newton iterations
    step=0.7,          # step size for updates
):
    """Estimate lever arms from dual-IMU data, returns (r1, r2) vectors."""
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
    data,                        # dict with gyr_1, gyr_2, acc_1, acc_2, r1, r2
    q1_init=None,                # initial quaternion, default [1, 0, 0, 0]
    cov_w=None,                  # process noise covariance (6x6)
    cov_lnk=None,                # measurement noise covariance (3x3)
    run_dynamic_update=True,     # run prediction step
    run_measurement_update=True, # run measurement update step
    use_raw_gyro=False,          # use raw gyro instead of filtered
):
    """EKF with gravity frame constraints, returns (orientation_s1, orientation_s2, P_list)."""
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


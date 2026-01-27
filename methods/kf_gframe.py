"""Kalman filter with gravity frame constraints for dual-IMU joint angle estimation.

This module contains the complete KF implementation including Jacobian computation.
"""
import numpy as np
import qmt
from scipy.sparse import lil_matrix, csc_matrix

from constants import FS, T as DT
from calTools import (
    integrateGyr, quatmultiply, quatconj, EXPq, LOGq,
    quatR, quatL, quat2matrix, crossM, approx_derivative,
    calc_acc_at_center, dMotion, dInit, dMotiontm1, dLnk,
    dLnkdr, dLnk_etaG, dInit_etaG, dMotion_t_etaG, dMotion_tp1_etaG
)


# =============================================================================
# Jacobian Computation Functions (from Jacob.py)
# =============================================================================

def calc_jac2(row_num, col_num, n, q_lin, q_1, cov_i, cov_w, cov_a, N):
    """Calculation of Jacobian matrix block for one sensor (No link)."""
    J = lil_matrix((row_num, col_num), dtype=np.float64)

    for t in range(N):
        if t == 0:
            J[0:3, 0:3] = np.linalg.inv(cov_i) ** 0.5 @ dInit(q_1, q_lin[0])

        if t > 0:
            pos_wt1_r = 3 + (t - 1) * 3
            pos_wt1_c = (t - 1) * 3
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c:pos_wt1_c+3] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotiontm1(q_lin[t], q_lin[t-1])
            )
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c+3:pos_wt1_c+6] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion(q_lin[t], q_lin[t-1])
            )

    return csc_matrix(J)


def calc_jac2_etaG(row_num, col_num, n, q_lin, q_1, cov_i, cov_w, cov_a, N):
    """Calculation of Jacobian matrix block for one sensor (eta-G parameterization)."""
    J = lil_matrix((row_num, col_num), dtype=np.float64)

    for t in range(N):
        if t == 0:
            J[0:3, 0:3] = np.linalg.inv(cov_i) ** 0.5 @ dInit_etaG(q_1, q_lin[0])

        if t > 0:
            pos_wt1_r = 3 + (t - 1) * 3
            pos_wt1_c = (t - 1) * 3
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c:pos_wt1_c+3] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion_t_etaG(q_lin[t], q_lin[t-1])
            )
            J[pos_wt1_r:pos_wt1_r+3, pos_wt1_c+3:pos_wt1_c+6] = (
                np.linalg.inv(cov_w) ** 0.5 @ dMotion_tp1_etaG(q_lin[t], q_lin[t-1])
            )

    return csc_matrix(J)


def calcJac_Link(m, n, q_lin_s1, q_lin_s2, AccG1, AccG2, cov_lnk, N):
    """Calculation of Jacobian matrix block for the link part."""
    J = lil_matrix((m, n))

    for t in range(N):
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])
        pos_link_r = t * 3
        pos_link_c = t * 3

        J[pos_link_r:pos_link_r + 3, pos_link_c:pos_link_c + 3] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnk(Rn1, AccG1[:, t]))
        )
        J[pos_link_r:pos_link_r + 3, (3 * N) + pos_link_c:(3 * N) + pos_link_c + 3] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnk(Rn2, AccG2[:, t]))
        )

    return csc_matrix(J)


def calcJac_Link_etaG(m, n, q_lin_s1, q_lin_s2, AccG1, AccG2, cov_lnk, N):
    """Calculation of Jacobian matrix block for the link part (eta-G parameterization)."""
    J = lil_matrix((m, n))

    for t in range(N):
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])
        pos_link_r = t * 3
        pos_link_c = t * 3

        J[pos_link_r:pos_link_r + 3, pos_link_c:pos_link_c + 3] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnk_etaG(Rn1, AccG1[:, t]))
        )
        J[pos_link_r:pos_link_r + 3, (3 * N) + pos_link_c:(3 * N) + pos_link_c + 3] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnk_etaG(Rn2, AccG2[:, t]))
        )

    return J


def calcJac_Link_r(m, n, q_lin_s1, q_lin_s2, gyr_1, gyr_2, cov_lnk, Fs, N):
    """Calculation of Jacobian matrix block for lever arm optimization."""
    J = lil_matrix((m, n))

    dgyr_1 = approx_derivative(gyr_1, Fs)
    dgyr_2 = approx_derivative(gyr_2, Fs)

    for t in range(N):
        Rn1 = quat2matrix(q_lin_s1[t, :])
        Rn2 = quat2matrix(q_lin_s2[t, :])

        C1 = crossM(gyr_1[:, t]) @ crossM(gyr_1[:, t]) + crossM(dgyr_1[:, t])
        C2 = crossM(gyr_2[:, t]) @ crossM(gyr_2[:, t]) + crossM(dgyr_2[:, t])

        pos_link_row = t * 3
        pos_link_col = 0

        J[pos_link_row:pos_link_row + 3, pos_link_col:pos_link_col + 3] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (-dLnkdr(Rn1, C1))
        )
        J[pos_link_row:pos_link_row + 3, pos_link_col+3:pos_link_col + 6] = (
            (np.linalg.inv(cov_lnk) ** 0.5) @ (dLnkdr(Rn2, C2))
        )

    return J


# =============================================================================
# Extended Kalman Filter (from KF_Gframe.py)
# =============================================================================


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

    print("T in KF_Gframe:", T)
    print("N in KF_Gframe:", N)
    print("cov_w in KF_Gframe:", cov_w)
    print("cov_lnk in KF_Gframe:", cov_lnk)

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

        if np.any(acc_1_t > 300) or np.any(acc_2_t > 300):
            run_acc_inlimit = False
            num_rejected += 1
            print(f"detected uncommon {t} acc_1_t, acc_2_t: ", acc_1_t, acc_2_t)
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
            try:
                vec1 = quat2matrix(q_lin_s1_t) @ AccG1_t
                vec2 = quat2matrix(q_lin_s2_t) @ AccG2_t
                H[0:3, 0:3] = crossM(vec1)
                H[0:3, 3:6] = -crossM(vec2)
            except Exception as e:
                print(f"\n!!! Error detected at sample t={t} !!!")
                print(f"Error Type: {type(e).__name__}: {e}")
                print("N: ", N)
                print(f"DEBUG SHAPES:")
                print(f"  AccG1_t: {AccG1_t.shape if hasattr(AccG1_t, 'shape') else 'no shape'}")
                print(f"  q_lin_s1_t: {q_lin_s1_t.shape}")
                print(f"  vec1 (result of @): {vec1.shape if 'vec1' in locals() else 'N/A'}")
                print(f"DEBUG VALUES (NaN check):")
                print(f"  AccG1_t has NaN: {np.isnan(AccG1_t).any() if hasattr(AccG1_t, 'shape') else 'N/A'}")
                print(f"  AccG1_t content: {AccG1_t}")
                print(f"RAW DATA at t={t}:")
                print(f"  acc_1_t: {acc_1_t.flatten()}")
                print(f"  gyr_1_t: {gyr_1_t.flatten()}")
                raise e

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


# =============================================================================
# Public API
# =============================================================================

def run_kf_gframe(acc_prox, gyr_prox, acc_dist, gyr_dist, fs, r1=None, r2=None,
                  cov_w_scale=1e-2, cov_lnk_scale=0.35**2 * 10):
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

    Returns:
        (angle_deg, r1, r2): Joint angle in degrees and lever arm vectors
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

    # Compute relative angle from quaternions
    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)
    angle_deg = np.degrees(np.unwrap(qmt.eulerAngles(q_rel, axes='zyx')[:, 0]))

    return angle_deg, r1, r2

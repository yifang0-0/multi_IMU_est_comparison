"""Shared utility functions for estimation methods."""
import numpy as np
import pandas as pd
import qmt


def load_mot(file_path):
    """Load OpenSim motion (.mot) file."""
    header_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            header_lines += 1
            if line.strip().startswith('time'):
                break
    df = pd.read_csv(file_path, sep='\t', skiprows=header_lines-1)
    df.columns = df.columns.str.strip()
    return df


def estimate_joint_axes(acc_femur, acc_tibia, gyr_femur, gyr_tibia):
    """Estimate joint axes using Olsson method. Returns flattened axis vectors."""
    jhat_femur, jhat_tibia = qmt.jointAxisEstHingeOlsson(
        acc_femur, acc_tibia, gyr_femur, gyr_tibia
    )
    return jhat_femur.flatten(), jhat_tibia.flatten()


def calculate_joint_angle(q_prox, q_dist, jhat_prox):
    """Calculate joint angle using quaternion projection onto estimated axis."""
    q_rel = qmt.qmult(qmt.qinv(q_prox), q_dist)
    q_twist = qmt.quatProject(q_rel, jhat_prox)['projQuat']
    angle_mag = qmt.quatAngle(q_twist)
    twist_axis = qmt.quatAxis(q_twist)
    signs = np.sign(np.sum(twist_axis * jhat_prox, axis=1))
    return np.degrees(angle_mag * signs)


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
